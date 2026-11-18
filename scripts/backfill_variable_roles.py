"""Backfill a ``role`` column onto an existing all-engines corpus release.

The corpus release (``corpus_timecourse_all_engines.parquet``;
columns ``biomodel_id, job, engine, variable, time, value``) began carrying
constant **parameter** and **compartment** traces alongside the dynamical
**species**/**flux** ones (report_variables PR #35, "full observable
coverage"). With no label distinguishing them, those flat constant traces
dilute every cross-engine agreement statistic computed over the release
(reviewer findings, 2026-10-28).

This script annotates each row with the variable's SBML ``role`` — one of
``species`` / ``flux`` / ``parameter`` / ``compartment`` — so a downstream
analysis can keep the full coverage yet set the constants aside. It also
collapses every non-finite value (``inf`` / ``-inf`` / NULL) to a single NaN
sentinel, so missing/blown-up points share one encoding instead of a
NaN/NULL/inf mix.

No simulators are run: roles come straight from each model's SBML, read from
the local biomodels pooch cache (``biomodels.common.cache_path``) or the
repo's ``models/<bid>/`` directory.

Usage::

    python -m scripts.backfill_variable_roles \
        --src datasets/corpus_all_engines/corpus_timecourse_all_engines.parquet \
        --out datasets/corpus_all_engines/corpus_timecourse_all_engines.roles.parquet
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import libsbml
import pyarrow as pa
import pyarrow.parquet as pq

from viva_biomodels.report_variables import classify_report_variables
from viva_biomodels.two_tier import UNKNOWN_ROLE


def _default_cache_dir() -> Optional[Path]:
    """The biomodels pooch download cache, if the package is importable."""
    try:
        from biomodels.common import cache_path  # type: ignore
        return Path(cache_path) / "model" / "download"
    except Exception:
        return None


def resolve_sbml(bid: str, cache_dir: Optional[Path], models_dir: Path) -> Optional[Path]:
    """Locate a model's SBML: pooch cache first, then the repo ``models/`` dir."""
    candidates: List[Path] = []
    if cache_dir is not None:
        candidates.append(cache_dir / bid / f"{bid}_url.xml")
        candidates += sorted((cache_dir / bid).glob("*.xml")) if (cache_dir / bid).is_dir() else []
    candidates.append(models_dir / bid / f"{bid}_url.xml")
    candidates += sorted((models_dir / bid).glob("*.xml")) if (models_dir / bid).is_dir() else []
    for c in candidates:
        if c.is_file():
            return c
    return None


def role_map_for_model(sbml_path: Path, variables: List[str]) -> Dict[str, str]:
    """``{variable: role}`` for one model's reported variables (``{}`` on failure)."""
    try:
        sd = libsbml.readSBML(str(sbml_path))
        model = sd.getModel()
        if model is None:
            return {}
        return classify_report_variables(model, variables)
    except Exception:
        return {}


def build_role_lookup(
    bids: List[str],
    vars_by_bid: Dict[str, List[str]],
    cache_dir: Optional[Path],
    models_dir: Path,
) -> tuple[Dict[tuple, str], Dict[str, int]]:
    """Build ``{(bid, variable): role}`` plus a per-role tally. Unresolved
    models / variables get ``UNKNOWN_ROLE`` so every row ends up labelled."""
    lookup: Dict[tuple, str] = {}
    tally: Dict[str, int] = {}
    n_missing_sbml = 0
    for bid in bids:
        variables = vars_by_bid.get(bid, [])
        sbml = resolve_sbml(bid, cache_dir, models_dir)
        roles = role_map_for_model(sbml, variables) if sbml else {}
        if not sbml:
            n_missing_sbml += 1
        for v in variables:
            role = roles.get(v, UNKNOWN_ROLE)
            lookup[(bid, v)] = role
            tally[role] = tally.get(role, 0) + 1
    tally["_models_without_sbml"] = n_missing_sbml
    return lookup, tally


def backfill(src: Path, out: Path, cache_dir: Optional[Path], models_dir: Path) -> Dict:
    table = pq.read_table(src)
    cols = table.column_names
    for required in ("biomodel_id", "variable", "value"):
        if required not in cols:
            raise SystemExit(f"{src} is missing required column {required!r} (has {cols})")

    bid_col = table.column("biomodel_id").to_pylist()
    var_col = table.column("variable").to_pylist()

    # distinct variables per model — the only thing we must classify.
    vars_by_bid: Dict[str, set] = {}
    for b, v in zip(bid_col, var_col):
        vars_by_bid.setdefault(b, set()).add(v)
    vars_by_bid = {b: sorted(vs) for b, vs in vars_by_bid.items()}
    bids = sorted(vars_by_bid)

    lookup, tally = build_role_lookup(bids, vars_by_bid, cache_dir, models_dir)

    # row-aligned role column.
    role_col = [lookup[(b, v)] for b, v in zip(bid_col, var_col)]

    # collapse non-finite values to a single NaN sentinel.
    nan = float("nan")
    raw_vals = table.column("value").to_pylist()  # NULL -> None
    n_inf = sum(1 for x in raw_vals if x is not None and math.isinf(x))
    n_null = sum(1 for x in raw_vals if x is None)
    value_col = [nan if (x is None or not math.isfinite(x)) else float(x) for x in raw_vals]

    # rebuild the table: keep original columns (value normalized), append role.
    arrays, names = [], []
    for name in cols:
        if name == "value":
            arrays.append(pa.array(value_col, type=pa.float32()))
        else:
            arrays.append(table.column(name))
        names.append(name)
    arrays.append(pa.array(role_col).dictionary_encode())
    names.append("role")
    out_table = pa.table(arrays, names=names)

    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, out, compression="zstd")

    return {
        "src": str(src),
        "out": str(out),
        "rows": out_table.num_rows,
        "models": len(bids),
        "role_tally": tally,
        "values_normalized": {"inf": n_inf, "null": n_null},
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="input release parquet")
    ap.add_argument("--out", required=True, type=Path, help="output parquet (with role column)")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="biomodels SBML download cache (default: biomodels pooch cache)")
    ap.add_argument("--models-dir", type=Path, default=Path("models"),
                    help="repo models dir, used as an SBML fallback (default: models/)")
    args = ap.parse_args(argv)

    cache_dir = args.cache_dir or _default_cache_dir()
    summary = backfill(args.src, args.out, cache_dir, args.models_dir)

    import json
    print(json.dumps(summary, indent=2))
    # guardrail: a release that classifies nothing is almost certainly a bad cache.
    tally = summary["role_tally"]
    classified = sum(v for k, v in tally.items() if not k.startswith("_"))
    unknown = tally.get(UNKNOWN_ROLE, 0)
    if classified and unknown == classified:
        print("WARNING: every variable is 'unknown' — no SBML resolved. "
              "Check --cache-dir.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
