# Next steps for biomodels

This workspace IS the model. You build it incrementally through phases.
Most work happens in the **dashboard** — five tabs, each with the
buttons you need. Skills (Claude Code) are the alternative for
code-writing tasks.

```
bash scripts/serve.sh   # opens browser at http://localhost:<port>
```

## 0 — One-time setup

- [ ] Create the venv: `uv venv .venv`
- [ ] Activate it: `source .venv/bin/activate`
- [ ] Install workspace deps: `uv pip install -e ".[dev]"`
- [ ] Lint: `python3 scripts/lint-workspace.py` should print `workspace lint: OK`
- [ ] Commit + (eventually) push: `git init && git add -A && git commit -m "feat: workspace bootstrap"`

## The 5 dashboard tabs

| Tab | What it's for | Key buttons |
|---|---|---|
| **Workspace inputs** | External resources you bring in | + Import · + Dataset · + Reference (PDF) · + Expert doc · **Install** (per import) |
| **Simulation Setup** | What to track + how to run | + Add observable · + Add simulation |
| **Visualizations** | How to render trajectories | + Add visualization (Plotly chart configs) |
| **Registry** | Live `build_core()` introspection | Refresh registry — shows discoverable processes + types |
| **Build Model** | Iterative phases of model development | + Add phase · Start phase · Evaluate gate (per phase) |

A persistent **Branch timeline** below the tabs shows every stage branch
the dashboard has produced. Each row has Copy gh / Copy git merge / Show diff
buttons to land the change.

## 1 — Workspace inputs (any time)

Inputs aren't a sequential stage — load them whenever they're useful.

- **Imports** — external `pbg-*` packages. Register via `+ Import`, then
  click **Install** to `pip install -e <path>` into the workspace venv.
  Once installed, `allocate_core()` auto-discovers their processes via
  [bigraph-schema's discovery convention][discovery]. They appear in the
  Registry tab.
- **Datasets** — experimental data the model validates against. Drag-drop a file.
- **References** — paper PDFs. Drop the file; pypdf extracts metadata; bibtex auto-generates.
- **Expert docs** — lab notes, curated reviews, working drafts. Drag-drop a PDF.

Each `+ Add` lands on a `stage/*` branch you can merge from the Branch timeline.

## 2 — Simulation Setup

- **Observables** — paths in the composite document you'll track
  (e.g. `chromosome.DnaA_count`).
- **Simulations** — run configs: name, t_start, t_end, initial_state,
  parameter_overrides, emitter_config, optionally scoped to specific phases
  and to specific Processes from the Registry.

## 3 — Visualizations

Configs for charts that render observable trajectories. Each visualization
can scope to specific phases and/or a specific simulation.

## 4 — Registry

Live snapshot of `build_core()` — every Process and Type currently
discoverable. After installing an import, click **Refresh registry** to
see new processes appear. Empty registry usually means the venv isn't
set up or the import isn't pip-installed.

## 5 — Build Model (the phases)

Each phase is a small unit of model development:

- An **objective** + a **Phase Gate** (acceptance criteria)
- A cited **source artifact** (expert doc / dataset / reference)
- New mechanisms wired into the composite (processes, wires, store paths)
- **Acceptance tests** that exercise the new mechanism
- A landed **PR** when the gate passes

Click `+ Add phase` to create one. Each phase has its own panel below
the tracker — clicking a pill scrolls to it. Action buttons:

- **Start phase** (planned → in_progress) — branches and scaffolds
- **Evaluate gate** (in_progress → complete or back to in_progress)

If you're using Claude Code, `/pbg-phase <n>` walks the phase iteratively:
opens the phase panel, drives the implementation, runs the gate.

## Two paths to the same workspace

- **Browser dashboard** is the primary UI. Pure-CLI workflows are supported
  via `python3 scripts/lint-workspace.py` + direct YAML edits, but the
  buttons handle branch-creation and PR setup for you.
- **Claude Code + [pbg-superpowers]** adds skills for the work that benefits
  from Claude: `/pbg-phase <n>` (drive a phase), `/pbg-expert <tool>` (wrap
  a new simulator), `/pbg-composer` (compose pbg-* wrappers). The dashboard
  and the skills share the same `workspace.yaml`.

## Config files to know

- `workspace.yaml` — canonical state (observables, simulations, visualizations, phases, imports, datasets, …)
- `references/{papers.bib, claims.yaml}` — bibliography + claim → paper mapping
- `.pbg/schemas/workspace.schema.json` — JSON schema (lint enforces it)
- `phases/phase-<n>.md` — per-phase spec (frontmatter + body)

## Focused dashboard panels

Open just one panel for a targeted interaction:

| URL | What |
|---|---|
| `http://localhost:<port>/?focus=workspace-inputs` | Just datasets / references / expert docs |
| `http://localhost:<port>/?focus=registry` | Just the module catalog + installed modules |
| `http://localhost:<port>/?focus=simulation-setup` | Just observables + simulations |
| `http://localhost:<port>/?focus=visualizations` | Just the viz lifecycle |
| `http://localhost:<port>/?focus=build-model` | Just phases/tasks |

Skills in Claude Code can `open <url>` to surface a focused interaction without dumping users into the full 5-tab UI.

## When stuck

1. `python3 scripts/lint-workspace.py` — most workspace-shape issues surface here.
2. Look at the **Next step** banner at the top of the dashboard — it points at the right tab.
3. File an issue at <https://github.com/vivarium-collective/pbg-superpowers/issues>.

[discovery]: https://github.com/vivarium-collective/pbg-superpowers/blob/main/docs/conventions/discovery.md
[pbg-superpowers]: https://github.com/vivarium-collective/pbg-superpowers

## Re-run follow-ups (from the 2026-10-28 findings)

After the corpus-side fixes in `docs/reviews/2026-11-18-findings-response.md`,
these need the next full corpus run (on the mini) to close out:

- [ ] Regenerate the all-engines corpus so `role` is emitted natively and the
      **tellurium concentration fix** takes effect; re-check cross-engine
      agreement with `role in (parameter, compartment)` excluded.
- [ ] Backfill the `role` column onto the **full-res 200-pt release** T.J.
      analyzes (same `scripts/backfill_variable_roles.py`), on the mini / R2.
- [ ] Verify the **copasi BIOMD0000000465 `auto_ten_seconds`** time grid
      (0–995 → 0–10) — looks model-specific.
- [ ] Make the live dashboard summary (`lazy_viewer._summary_stats`) role-aware
      once the two-tier `series/<bid>.parquet` carries `role`, so our own
      headline excludes constants too.
