#!/usr/bin/env python
"""Export every investigation's self-contained HTML report.

The report is generated SERVER-SIDE by vivarium-workbench's deterministic,
data-only generator (``lib.investigation_report``), the same code path the
workbench's own static-bundle publisher uses and the same one backing
``GET /api/investigation-report/<slug>`` in the live app. Every panel is read
from existing ``investigation.yaml`` / ``study.yaml`` / loop-trajectory JSON,
with figures inlined as data-URIs — so the output is fully self-contained.

History: this script used to drive the workbench SPA headlessly through
Playwright, clicking a "Generate report" button and capturing the download.
That client-side builder was deleted upstream in vivarium-workbench #878
(after #873 moved generation server-side and #876 rewired the SPA), which
removed ``window._generateInvestigationReport`` and left this script waiting
30 s for a symbol that no longer exists — all 14 investigations timed out and
the deploy published nothing from 2026-08-18 onward. Calling the library
directly is what #878's own rewiring did for workbench's in-repo consumers;
this is the same move for the one that lives out-of-repo.

Consequences of the migration: no browser, no headless server, no port
juggling, and the output is deterministic rather than dependent on SPA boot
timing.

Usage:
  .venv/bin/python scripts/publish_investigation_reports.py \
      --workspace . --out reports/published

Exit code is non-zero only when NOTHING published, so one stubborn
investigation cannot red the whole deploy.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

# A report with figures must embed them; a report that lost its embeds has
# none. We treat "claims figures but embeds none" as a hard failure rather than
# silently publishing a stripped report — a stripped report once clobbered a
# good gh-pages page, so a failed report must leave NO file behind and let the
# copy step preserve the last-good one.
MIN_REPORT_BYTES = 20_000


def discover_investigations(ws_root: Path) -> list[str]:
    inv_root = ws_root / "workspace" / "investigations"
    if not inv_root.is_dir():
        inv_root = ws_root / "investigations"  # flat-layout fallback
    return sorted(
        d.name for d in inv_root.iterdir()
        if d.is_dir() and (d / "investigation.yaml").is_file()
    )


def _load_investigation_yaml(ws_root: Path, slug: str) -> dict:
    """Read an investigation's investigation.yaml (nested or flat layout)."""
    for base in (ws_root / "workspace" / "investigations", ws_root / "investigations"):
        p = base / slug / "investigation.yaml"
        if p.is_file():
            return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {}


def build_index_fragment(ws_root: Path, slugs: list[str]) -> str:
    """Build the ``<div class="invest">`` blocks for the gh-pages landing page.

    Generated from each investigation.yaml (title, status, a short description,
    study list) so the root gallery lists EVERY investigation that has a
    published report — no hand-curation. Injected between the
    ``auto-investigations`` markers in ``index.html`` by the publish workflow.
    """
    import html as _h

    blocks: list[str] = []
    for slug in slugs:
        spec = _load_investigation_yaml(ws_root, slug)
        title = str(spec.get("title") or slug)
        status_raw = str(spec.get("status") or "").strip()
        status_label = status_raw.replace("_", " ") or "—"
        status_class = status_raw or "in_progress"

        # Description: prefer executive.what_is_this, else the question; collapse
        # whitespace and truncate so the gallery card stays compact.
        execu = spec.get("executive") if isinstance(spec.get("executive"), dict) else {}
        desc = str(execu.get("what_is_this") or spec.get("question") or "").strip()
        desc = " ".join(desc.split())
        if len(desc) > 300:
            desc = desc[:297].rstrip() + "…"

        # Same members-vs-studies union as study_figure_count() — this is the
        # gallery's enforcement point of it. Reading `studies` alone made every
        # card on the landing page render "0 studies"; measured on the live
        # gh-pages index.html, so it is pre-existing rather than a regression.
        raw_members = list(spec.get("members") or []) + list(spec.get("studies") or [])
        studies = [s.get("name") if isinstance(s, dict) else s for s in raw_members]
        studies = [str(s) for s in studies if s]
        meta = f"{len(studies)} stud{'y' if len(studies) == 1 else 'ies'}"
        if 0 < len(studies) <= 4:
            meta += " · " + " · ".join(studies)

        blocks.append(
            '<div class="invest">\n'
            f'  <h3><a href="investigations/{_h.escape(slug)}.html">{_h.escape(title)}</a>\n'
            f'      <span class="pill {_h.escape(status_class)}">{_h.escape(status_label)}</span></h3>\n'
            f'  <p>{_h.escape(desc)}</p>\n'
            f'  <p class="meta">{_h.escape(meta)}</p>\n'
            '</div>'
        )
    return "\n\n".join(blocks)


def study_figure_count(ws_root: Path, slug: str) -> int:
    """How many figure HTMLs the investigation's studies reference on disk.

    Used only to decide whether a generated report SHOULD contain embeds, so we
    can flag a silently-stripped report. Counts committed figure files under
    reports/figures/<study>/ for each study in the investigation.
    """
    for base in (ws_root / "workspace" / "investigations", ws_root / "investigations"):
        inv_yaml = base / slug / "investigation.yaml"
        if inv_yaml.is_file():
            break
    else:
        return 0
    spec = yaml.safe_load(inv_yaml.read_text(encoding="utf-8")) or {}
    # Investigations key their members under `members` (schema_version 2 and 4);
    # `studies` is the legacy key and is EMPTY in all 14 current investigations,
    # so reading it alone made this return 0 everywhere and silently disabled the
    # stripped-report guard in render_report(). Union both.
    raw_members = list(spec.get("members") or []) + list(spec.get("studies") or [])
    studies = [s.get("name") if isinstance(s, dict) else s for s in raw_members]
    fig_root = ws_root / "reports" / "figures"
    n = 0
    for st in filter(None, studies):
        d = fig_root / st
        if d.is_dir():
            n += len(list(d.glob("*.html")))
    return n


def render_report(ws_root: Path, slug: str, out_path: Path,
                  expect_figures: int) -> tuple[bool, str]:
    """Render one investigation report to ``out_path``. Returns (ok, message).

    Integrity-checks BEFORE writing: a report that is implausibly small, or
    that claims figures and embeds none, is not written at all, so the publish
    step preserves the last-good page instead of overwriting it with a stub.
    """
    from vivarium_workbench.lib.investigation_report import (
        build_report_data,
        render_html,
    )
    for base in (ws_root / "workspace" / "investigations", ws_root / "investigations"):
        if (base / slug / "investigation.yaml").is_file():
            break
    else:
        return False, "investigation not found"
    data = build_report_data(ws_root, slug)
    html = render_html(data)

    # Count the figures the generator ACTUALLY inlined, off the data dict — never
    # by grepping the rendered HTML. Marker-grepping cannot work against this
    # template: it already carries 2 `<iframe` + 1 `srcdoc` in its own renderer JS
    # (a constant floor of 3), and render_html JSON-escapes `<` to \u003c, so
    # figure markup inside the payload never matches a `<iframe` grep. Under that
    # metric a figure-less report is indistinguishable from a complete one.
    embedded = sum(len(st.get("figures_embedded") or [])
                   for st in (data.get("studies") or []) if isinstance(st, dict))

    size = len(html.encode("utf-8"))
    if size < MIN_REPORT_BYTES:
        return False, f"report too small ({size} B < {MIN_REPORT_BYTES}); not published"
    if expect_figures and embedded == 0:
        return False, (f"{size:,} B but ZERO figures inlined while its members carry "
                       f"{expect_figures} committed figure file(s) — report stripped; "
                       "not published (last-good page preserved)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    # Always surface inlined-vs-committed. The generator sources figures from a
    # study's own viz/ + charts/ (non-recursive) and from `image:`-addressed
    # data-URIs; it does NOT read reports/figures/<study>/, so a report can be
    # legitimately published while still carrying fewer figures than the
    # workspace commits. Printing both makes a partial loss visible in the deploy
    # log instead of hiding behind a bare ✓.
    note = f"{size:,} B, {embedded} figures inlined"
    if expect_figures and embedded < expect_figures:
        note += f" (workspace commits {expect_figures} — PARTIAL)"
    return True, note


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", default=".", help="workspace root (default: .)")
    ap.add_argument("--out", default="reports/published",
                    help="output dir; reports written to <out>/investigations/<slug>.html")
    ap.add_argument("--only", default=None,
                    help="comma-separated investigation slugs to publish (default: all)")
    # Accepted-and-ignored: the generator needs no server. Kept so existing
    # invocations and muscle memory do not hard-fail on an unknown flag.
    ap.add_argument("--url", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--port", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.url is not None or args.port is not None:
        print("note: --url/--port are obsolete (reports render in-process, "
              "no dashboard server involved) — ignoring", file=sys.stderr)

    # Escape hatch, set by the workflow's force_republish dispatch input. A
    # LEGITIMATE shrink (studies removed, figures deliberately trimmed) is
    # otherwise unpublishable, because the stripped-report guard here and the
    # size-ratio gate in the workflow's copy step would both refuse it forever.
    force = bool(os.environ.get("FORCE_REPUBLISH"))
    if force:
        print("FORCE_REPUBLISH set — the stripped-report guard is BYPASSED",
              file=sys.stderr)

    ws_root = Path(args.workspace).resolve()
    out_dir = Path(args.out).resolve()
    slugs = discover_investigations(ws_root)
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        slugs = [s for s in slugs if s in want]
    if not slugs:
        print("no investigations found", file=sys.stderr)
        return 1
    print(f"investigations: {', '.join(slugs)}")

    results: dict[str, tuple[bool, str]] = {}
    for slug in slugs:
        out_path = out_dir / "investigations" / f"{slug}.html"
        expect_figures = 0 if force else study_figure_count(ws_root, slug)
        try:
            ok, msg = render_report(ws_root, slug, out_path, expect_figures)
        except Exception as e:  # noqa: BLE001 — report per-slug, keep going
            ok, msg = False, f"exception: {e}"
        results[slug] = (ok, msg)
        print(f"  {'\u2713' if ok else '\u2717'} {slug}: {msg}")

    # Regenerate the landing-page investigation list from ALL discovered
    # investigations (not just this run's --only subset), so the gh-pages root
    # gallery always lists every investigation with a published report.
    all_slugs = discover_investigations(ws_root)
    fragment = build_index_fragment(ws_root, all_slugs)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_fragment_path = out_dir / "investigations_index.html"
    index_fragment_path.write_text(fragment + "\n", encoding="utf-8")
    print(f"wrote landing-page fragment ({len(all_slugs)} investigations) to "
          f"{index_fragment_path}")

    failed = [s for s, (ok, _) in results.items() if not ok]
    n_ok = len(results) - len(failed)
    print(f"\n{n_ok}/{len(results)} reports published to "
          f"{out_dir / 'investigations'}")
    if failed:
        print(f"FAILED (published reports still shipped): {', '.join(failed)}",
              file=sys.stderr)
    # Gate policy unchanged: a single stubborn investigation must NOT red the
    # whole deploy. Fail only when NOTHING published.
    if n_ok == 0:
        print("no reports published — failing the deploy", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
