#!/usr/bin/env python3
"""Build the committed all-engines corpus dataset + a compact per-model index.

Reads the full 5-engine BioModels x simulator timecourse parquet (produced by the
corpus sweep, `out/dataset_all_engines/...`), sorts it by model id so a remote
reader (duckdb-wasm in the corpus explorer) can prune row groups on a
`biomodel_id` predicate, and writes:

  datasets/corpus_all_engines/corpus_timecourse_all_engines.parquet
  datasets/corpus_all_engines/index.json   # compact per-model overview

Run from the workspace root:
  python scripts/build_corpus_all_engines.py --src <path-to-source.parquet>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        default="out/dataset_all_engines/corpus_timecourse_all_engines.parquet",
        help="Source all-engines timecourse parquet.",
    )
    ap.add_argument("--out-dir", default="datasets/corpus_all_engines")
    ap.add_argument("--row-group-size", type=int, default=200_000)
    ap.add_argument(
        "--series-dir", default=None,
        help="If set, also emit one per-model JSON (id.json) here for the "
             "corpus explorer's lazy per-model loading (an R2-only serving "
             "artifact; not committed).",
    )
    args = ap.parse_args()

    src = Path(args.src)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"reading {src} ...")
    table = pq.read_table(src)
    cols = table.column_names
    assert {"biomodel_id", "job", "engine", "variable", "time", "value"} <= set(cols), cols

    # Decode any dictionary-encoded string columns to plain strings so sort_by
    # (which can't order dictionary<...>) and downstream grouping work.
    import pyarrow.types as pat
    new_cols = {}
    for name in cols:
        col = table.column(name).combine_chunks()
        if pat.is_dictionary(col.type):
            col = col.cast(col.type.value_type)
        new_cols[name] = col
    table = pa.table(new_cols)

    # Cluster by model id (then engine/job/variable/time) so a WHERE biomodel_id=…
    # predicate hits a contiguous, prunable set of row groups.
    table = table.sort_by([
        ("biomodel_id", "ascending"),
        ("engine", "ascending"),
        ("job", "ascending"),
        ("variable", "ascending"),
        ("time", "ascending"),
    ])

    dst = out_dir / "corpus_timecourse_all_engines.parquet"
    pq.write_table(
        table, dst, compression="zstd", row_group_size=args.row_group_size,
        use_dictionary=True,
    )
    print(f"wrote {dst} ({dst.stat().st_size/1e6:.1f} MB, {table.num_rows:,} rows)")

    # Compact per-model index for the explorer's instant overview.
    import pyarrow.compute as pc

    bids = table.column("biomodel_id")
    all_engines = sorted(set(table.column("engine").to_pylist()))
    models = []
    series_dir = Path(args.series_dir) if args.series_dir else None
    if series_dir:
        series_dir.mkdir(parents=True, exist_ok=True)
    for bid in sorted(set(bids.to_pylist())):
        mask = pc.equal(bids, pa.scalar(bid))
        sub = table.filter(mask)
        engines = sorted(set(sub.column("engine").to_pylist()))
        variables = sorted(set(sub.column("variable").to_pylist()))
        jobs = sorted(set(sub.column("job").to_pylist()))
        models.append({
            "id": bid,
            "engines": engines,
            "n_engines": len(engines),
            "n_variables": len(variables),
            "n_jobs": len(jobs),
            "n_points": sub.num_rows,
        })
        if series_dir:
            # One JSON per model: a flat list of (engine, variable, job) traces,
            # each with parallel time/value arrays — directly renderable by Plotly.
            se = sub.column("engine").to_pylist()
            sv = sub.column("variable").to_pylist()
            sj = sub.column("job").to_pylist()
            st = sub.column("time").to_pylist()
            svl = sub.column("value").to_pylist()
            traces: dict = {}
            for e, v, j, t, val in zip(se, sv, sj, st, svl):
                key = (e, v, j)
                tr = traces.get(key)
                if tr is None:
                    tr = {"engine": e, "variable": v, "job": j, "t": [], "v": []}
                    traces[key] = tr
                tr["t"].append(t)
                tr["v"].append(val)
            payload = {
                "id": bid,
                "engines": engines,
                "variables": variables,
                "jobs": jobs,
                "traces": list(traces.values()),
            }
            (series_dir / f"{bid}.json").write_text(json.dumps(payload), encoding="utf-8")

    index = {
        "dataset": "biomodels-corpus-all-engines",
        "engines": all_engines,
        "n_models": len(models),
        "n_rows": table.num_rows,
        "columns": cols,
        "models": models,
    }
    idx = out_dir / "index.json"
    idx.write_text(json.dumps(index), encoding="utf-8")
    print(f"wrote {idx} ({idx.stat().st_size/1e3:.1f} KB, {len(models)} models, engines={all_engines})")


if __name__ == "__main__":
    main()
