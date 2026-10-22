"""Ray fan-out runner for large-scale BioModels comparison → two-tier storage.

One Ray task per model: runs the batch-compare composite for that single model
(all engines + reference), writes its ``series/<bid>.parquet`` locally, and
returns only the compact index entry — so the multi-GB series never travel
through Ray's object store. Task crashes (the hard segfaulters) surface as Ray
exceptions and are recorded as ``crashed`` instead of killing the run, which
replaces the sequential bisect-on-crash logic.

Usage:
    python -m viva_biomodels.ray_runner --n 100 --out-dir out/compare_all \
        [--simulators copasi,tellurium,simbio,amici,pysces] \
        [--reference-results-dir datasets/biosimulators_sedml_results] \
        [--max-points 200] [--address auto]

AMICI must be precompiled (ninja on PATH) before launching so tasks hit the
cache instead of compiling; the shared on-disk caches (amici_models/,
pysces_models/, models/) live under the working directory.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(os.environ.get("PBG_BIOMODELS_ROOT", os.getcwd()))


def _run_model_store(bid: str, sims: List[str], ref_dir: str,
                     include_steady_state: bool = False,
                     parquet_out_dir: str = "") -> Tuple[dict, dict, dict]:
    """Run the batch-compare composite for one model; return its store slice.

    When ``parquet_out_dir`` is set the composite's default parquet emitter
    writes ``<parquet_out_dir>/<bid>.parquet`` in the corpus schema as it is
    gathered (in addition to the two-tier dataset the caller may build).
    """
    from process_bigraph import Composite, gather_emitter_results
    from viva_biomodels import register_types
    from viva_biomodels.core import build_core
    from viva_superpowers.composite_generator import _REGISTRY, build_generator
    import viva_biomodels.composites.batch_compare_biomodels  # noqa: F401

    core = register_types(build_core())
    entry = next(e for e in _REGISTRY.values() if e.name == "batch-compare-biomodels")
    doc = build_generator(entry, overrides={
        "biomodel_ids": [bid],
        "simulators": list(sims),
        "reference_results_dir": ref_dir or "",
        "reference_simulators": [],
        "include_steady_state": include_steady_state,
        "parquet_out_dir": parquet_out_dir,
    })
    comp = Composite(doc, core=core)
    comp.run(0.0)
    snap = (gather_emitter_results(comp).get(("emitter",)) or [{}])[-1]
    return (snap.get("results") or {}, snap.get("comparisons") or {},
            snap.get("diagnostics") or {})


def _task(bid: str, sims: List[str], ref_dir: str, out_dir: str, max_points: int,
          include_steady_state: bool = False) -> Dict[str, Any]:
    """Ray task body: run one model, write its parquet, return the index entry."""
    os.chdir(str(ROOT))  # caches (amici_models/, pysces_models/, models/) are cwd-relative
    from viva_biomodels.two_tier import write_model
    # The default parquet emitter writes the per-model corpus-schema parquet into
    # <out_dir>/parquet/ as the run is gathered; two_tier still builds the
    # consolidated series/ + index entry below (the canonical dataset).
    res, comp, diag = _run_model_store(
        bid, sims, ref_dir, include_steady_state,
        parquet_out_dir=str(Path(out_dir) / "parquet"))
    entry = write_model(bid, res.get(bid) or {}, comp.get(bid) or {},
                        Path(out_dir), max_points)
    entry["runs"] = (diag.get("runs") or {}).get(bid) or {}
    return entry


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=0,
                    help="run models BIOMD1..N; 0 means use --ids-from-reference")
    ap.add_argument("--ids-from-reference", default="",
                    help="dir of BIOMD* subdirs to enumerate (e.g. the reference dataset)")
    ap.add_argument("--simulators", default="copasi,tellurium,simbio,amici,pysces")
    ap.add_argument("--reference-results-dir", default="datasets/biosimulators_sedml_results")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-points", type=int, default=200)
    ap.add_argument("--address", default="",
                    help="Ray cluster address ('auto' to attach, empty for local)")
    ap.add_argument("--amici-compile", type=str, default="2",
                    help="AMICI_PARALLEL_COMPILE inside tasks")
    ap.add_argument("--steady-state", action="store_true",
                    help="also run a steady-state comparison per model")
    a = ap.parse_args()

    import ray

    sims = [s.strip() for s in a.simulators.split(",") if s.strip()]
    ref_dir = a.reference_results_dir
    out_dir = str((ROOT / a.out_dir) if not os.path.isabs(a.out_dir) else Path(a.out_dir))
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if a.ids_from_reference:
        ids = sorted(p.name for p in Path(a.ids_from_reference).iterdir()
                     if p.is_dir() and p.name.startswith("BIOMD"))
    else:
        ids = [f"BIOMD{n:010d}" for n in range(1, a.n + 1)]

    # ninja (for any uncached AMICI compile) + parallel compile, propagated to workers.
    env_vars = {
        "PATH": f"{ROOT / '.venv/bin'}:{os.environ.get('PATH', '')}",
        "AMICI_PARALLEL_COMPILE": a.amici_compile,
        "PYTHONUTF8": "1",
        "PBG_BIOMODELS_ROOT": str(ROOT),
    }
    ray.init(address=a.address or None, runtime_env={"env_vars": env_vars})
    print(f"Ray up: {ray.cluster_resources()}. Running {len(ids)} models "
          f"under {sims} -> {out_dir}", flush=True)

    # max_retries=0: a crashing model fails ONCE and is recorded, not retried.
    remote = ray.remote(max_retries=0, num_cpus=1)(_task)
    futures = {remote.remote(bid, sims, ref_dir, out_dir, a.max_points,
                             a.steady_state): bid for bid in ids}

    from viva_biomodels.two_tier import finalize_index
    entries: List[Dict[str, Any]] = []
    crashed: List[str] = []
    pending = list(futures.keys())
    done_n = 0
    while pending:
        ready, pending = ray.wait(pending, num_returns=1, timeout=None)
        for fut in ready:
            bid = futures[fut]
            try:
                entries.append(ray.get(fut))
            except Exception as e:  # WorkerCrashedError / RayTaskError / etc.
                crashed.append(bid)
                print(f"  CRASH {bid}: {type(e).__name__}", flush=True)
            done_n += 1
            if done_n % 25 == 0:
                print(f"  {done_n}/{len(ids)} done ({len(crashed)} crashed)", flush=True)

    meta = {"n_models": len(entries), "ids": ids, "crashed_models": crashed,
            "simulators": sims}
    # diagnostics.runs merged from per-model entries for the runtime tab.
    runs = {e["id"]: e.pop("runs", {}) for e in entries}
    finalize_index(entries, Path(out_dir), meta=meta,
                   diagnostics={"runs": runs, "meta": meta})

    ok = {s: 0 for s in sims}
    for bid, jobs in runs.items():
        for job, simmap in (jobs or {}).items():
            for s, rec in (simmap or {}).items():
                if rec.get("status") == "ok":
                    ok[s] = ok.get(s, 0) + 1
    print(f"\nWrote two-tier output to {out_dir}")
    print(f"  models with results: {len(entries)}/{len(ids)}  crashed: {len(crashed)}")
    for s in sims:
        print(f"  {s:10s} ok={ok.get(s, 0)}")
    ray.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
