"""Two-tier storage for large-scale batch comparisons.

The flat ``results[bid][job][engine]`` store embeds every time series inline,
which is ~99% of the bytes and does not scale (1054 models ≈ 6 GB JSON, a
~700 MB HTML that won't open). This module splits the store into:

* a compact **index.json** — per (model, job): the pairwise comparison matrix,
  bucket, and the pbg↔pbg / pbg↔reference analysis. ~tens of KB/model, so all
  1054 models fit in ~70 MB. Drives the sortable overview + cross-engine tables.
* per-model **series/<bid>.parquet** — the time series in tidy long form
  (job, engine, variable, time, value), downsampled + float32, dictionary-
  compressed. ~30-50 KB/model, lazy-fetched only when a model is opened.

Both the offline converter (existing full stores) and the Ray runner write
through ``write_model`` / ``finalize_index`` so the 6 GB monolith is never
materialized.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

#: Role assigned to a variable whose SBML classification is unavailable (no
#: ``roles`` map was threaded through). The backfill script fills these in.
UNKNOWN_ROLE = "unknown"


def _thin(seq: List[float], max_points: int) -> List[float]:
    n = len(seq)
    if n <= max_points:
        return seq
    stride = n / max_points
    return [seq[int(i * stride)] for i in range(max_points)]


def _finite_or_nan(x: float, nan: float) -> float:
    """Collapse every non-finite value (``inf``/``-inf``/``NaN``) to a single
    NaN sentinel, so missing/blown-up points share one encoding in the value
    column instead of a NaN/NULL/inf mix (findings 2026-10-28)."""
    return x if math.isfinite(x) else nan


def series_table(
    model_results: Dict[str, Any],
    max_points: int = 200,
    roles: Optional[Dict[str, str]] = None,
) -> pa.Table:
    """Flatten one model's ``{job: {engine: {var: [...], time: [...]}}}`` to a
    tidy Arrow table (job, engine, variable, role, time, value), downsampled + f32.

    ``roles`` maps each variable id to its SBML role
    (``species``/``flux``/``parameter``/``compartment``; see
    ``report_variables.classify_report_variables``). A downstream analysis can
    then set the constant ``parameter``/``compartment`` traces aside so they do
    not dilute cross-engine agreement statistics. When ``roles`` is omitted the
    role column is filled with ``"unknown"`` (backfilled separately).
    """
    roles = roles or {}
    jobs: List[str] = []
    engines: List[str] = []
    variables: List[str] = []
    role_col: List[str] = []
    times: List[float] = []
    values: List[float] = []
    nan = float("nan")
    from viva_biomodels import result_leaf

    for job, engine_map in (model_results or {}).items():
        for engine, leaf in (engine_map or {}).items():
            if not leaf:
                continue
            # UTC leaves carry a "time" axis, parameter-scan leaves a "scan"
            # axis; steady-state leaves have neither and are marked with NaN
            # times so the reader can distinguish them on round-trip. Both
            # ordered axes are written into the (generic) "time" parquet column;
            # the per-job index kind relabels a scan axis back on read.
            axis_name, axis_vals = result_leaf.axis_of(leaf)
            is_ss = not axis_name
            # as_scalar tolerates length-1 nested cells ([v]) that some engines /
            # constant-fills emit for params/compartments under current deps.
            t = _thin([result_leaf.as_scalar(x) for x in axis_vals], max_points)
            for var, ys in leaf.items():
                if var in ("time", "scan") or not isinstance(ys, list):
                    continue
                yv = [_finite_or_nan(x, nan)
                      for x in _thin([result_leaf.as_scalar(x) for x in ys], max_points)]
                if is_ss:
                    tv = [nan] * len(yv)
                    m = len(yv)
                else:
                    m = min(len(t), len(yv))
                    tv = t[:m]
                jobs.extend([job] * m)
                engines.extend([engine] * m)
                variables.extend([var] * m)
                role_col.extend([roles.get(var, UNKNOWN_ROLE)] * m)
                times.extend(tv[:m])
                values.extend(yv[:m])
    return pa.table({
        # dictionary-encode the highly-repeated label columns.
        "job": pa.array(jobs).dictionary_encode(),
        "engine": pa.array(engines).dictionary_encode(),
        "variable": pa.array(variables).dictionary_encode(),
        "role": pa.array(role_col).dictionary_encode(),
        "time": pa.array(times, type=pa.float32()),
        "value": pa.array(values, type=pa.float32()),
    })


def write_model(
    bid: str,
    model_results: Dict[str, Any],
    model_comparisons: Dict[str, Any],
    out_dir: Path,
    max_points: int = 200,
    roles: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Write ``series/<bid>.parquet`` and return this model's index entry.

    The index entry carries everything the overview + cross-engine tables need
    (per-job comparison matrix, engines, ok/failed counts) — no time series.
    ``roles`` (``{var_id: role}``) labels each variable's SBML role in the
    series parquet; see ``series_table``.
    """
    series_dir = out_dir / "series"
    series_dir.mkdir(parents=True, exist_ok=True)
    table = series_table(model_results, max_points=max_points, roles=roles)
    pq.write_table(table, series_dir / f"{bid}.parquet", compression="zstd")

    jobs_entry: Dict[str, Any] = {}
    for job, engine_map in (model_results or {}).items():
        leaves = engine_map or {}
        n_ok = sum(1 for v in leaves.values() if v)
        n_failed = sum(1 for v in leaves.values() if not v)
        comp = (model_comparisons or {}).get(job) or {}
        jobs_entry[job] = {
            "engines": comp.get("engines") or sorted(leaves.keys()),
            "matrix": comp.get("matrix") or {},
            "max_nrmse": comp.get("max_nrmse"),
            "bucket": comp.get("bucket"),
            # Second (co-equal) metric: BioSimulations closeness score.
            "max_score": comp.get("max_score"),
            "closeness_bucket": comp.get("closeness_bucket"),
            "matrix_closeness": comp.get("matrix_closeness") or {},
            "n_ok": n_ok,
            "n_failed": n_failed,
            # kind: "utc" if any leaf carries a time vector, else "steady_state".
            "kind": _job_kind(leaves),
        }
    return {"id": bid, "jobs": jobs_entry, "has_series": table.num_rows > 0}


def _job_kind(leaves: Dict[str, Any]) -> str:
    from viva_biomodels import result_leaf
    has_data = False
    for leaf in (leaves or {}).values():
        if leaf and isinstance(leaf, dict):
            has_data = True
            if result_leaf.is_utc(leaf):
                return "utc"
            if result_leaf.is_scan(leaf):
                return "repeated_task"
    return "steady_state" if has_data else "none"


def finalize_index(
    entries: List[Dict[str, Any]],
    out_dir: Path,
    meta: Optional[Dict[str, Any]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write the compact index.json for all models."""
    index = {
        "models": {
            e["id"]: {
                "jobs": e["jobs"],
                "has_series": e.get("has_series"),
                # Preserve per-engine run provenance (status/error/runtime) when
                # present; the viewer's execution drill-down reads it.
                **({"runs": e["runs"]} if e.get("runs") else {}),
            }
            for e in entries if e
        },
        "meta": meta or {},
        "diagnostics": diagnostics or {},
    }
    out = out_dir / "index.json"
    out.write_text(json.dumps(index), encoding="utf-8")
    return out


def convert_store(store_path: str, out_dir: str, max_points: int = 200) -> Dict[str, Any]:
    """Offline: split an existing flat store JSON into the two-tier layout."""
    store = json.loads(Path(store_path).read_text(encoding="utf-8"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = store.get("results") or {}
    comparisons = store.get("comparisons") or {}
    entries = [
        write_model(bid, results.get(bid) or {}, comparisons.get(bid) or {}, out, max_points)
        for bid in results
    ]
    diag = store.get("diagnostics") or {}
    meta = {"n_models": len(entries), "ids": list(results.keys()),
            "crashed_models": (diag.get("meta") or {}).get("crashed_models") or []}
    finalize_index(entries, out, meta=meta, diagnostics=diag)
    return {"n_models": len(entries), "out_dir": str(out)}
