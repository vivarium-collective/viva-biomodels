# Response to the 2026-10-28 regression findings

Thanks — this was a sharp read. The headline is right: the summary statistics
got diluted, and the cause is upstream, in what the corpus *contains*, not in
how it's scored. Here's what changed on the corpus side and what's left for the
analysis pipeline.

## The root cause

The "full observable coverage" work started filling **constant parameter and
compartment traces** into every engine's output, alongside the dynamical
species and fluxes — with nothing to tell them apart. Those flat traces are the
60–79% of newly-reported constants you measured, and they're why the headline
"identical" rate rose while agreement on time-varying traces fell. In the
current release constant rows actually **outnumber** species rows.

## What we fixed on the corpus side

1. **Every trace now carries a `role`** — `species` / `flux` / `parameter` /
   `compartment`. The release parquet has a new `role` column
   (`datasets/corpus_all_engines/README.md`). This keeps the full coverage *and*
   lets you set the constants aside. **For agreement stats, exclude
   `role in ("parameter","compartment")`** — that removes the dilution at the
   source and makes the both-constant set-aside a one-liner. The existing
   release has been annotated in place (no re-simulation — roles come from each
   model's SBML via `scripts/backfill_variable_roles.py`).

2. **tellurium amounts-vs-concentration** — the compartment-sized rescaling you
   isolated. The direct-RoadRunner path was selecting bare species ids, which
   return *amounts* for substance-units species, while COPASI and PySCeS report
   *concentrations*. tellurium now selects `[S]` (concentration) and keeps the
   bare id as the column label, so it matches the other engines and its own
   earlier releases. This changes values, so it lands with the next full run
   (see below) rather than the annotated release.

3. **Non-finite values** — the 1,600 infinities and the NaN/NULL split are
   collapsed to a single `NaN` sentinel in `value`, so there's one encoding for
   "no finite value." Done in the release now and in the writer for future runs.

4. **The machine-epsilon false divergence** — your `v04_one_engine_flat` case
   (pysces `2.2e-16` vs copasi `0` scoring as divergent because the zero-trace
   tolerance was 0). Our own nRMSE had the identical flaw; it now has a
   `ZERO_TRACE_ATOL = 1e-12` floor so a both-effectively-zero trace is set aside
   like a both-zero one. **The same floor belongs in your pipeline's
   `ZERO_TRACE_ATOL`** — with the `role` column you can also just drop the
   constants first and the issue mostly disappears.

## Things that are yours, not ours

- The `a04_*` / `v04_*` tables and the identical/close/loose/divergent tiers are
  your analysis pipeline — they aren't in viva-biomodels. The two changes that
  live there: the `ZERO_TRACE_ATOL` floor above, and the both-constant set-aside
  (now trivial via `role`). Everything else above we've handled.

## Two clarifications on the findings

- **`T`/`t` is not the time axis leaking in.** Those rows appear *identically
  across all five engines, amici included* — and amici never emits the time
  column. They're legitimate SBML entities a model happens to name `T`/`t`.
  They're now `role`-labelled, so you can see exactly what each one is.
- **copasi BIOMD0000000465 `auto_ten_seconds` grid (0–995 → 0–10)** — flagged
  for a targeted check at the next run; looks model-specific.

## Agreed progress

simbio's jump (34.9% → 92.3%, SED-ML gap closed) and pysces's honest-fail
behavior (no more empty rows) both look right. pysces's 280 absent attempts are
real simulation failures, tracked separately.

## Next full run

The tellurium fix and any value-changing fix regenerate the corpus (on the
mini). That release will fold in the role column natively and let us re-check
the copasi BIOMD465 grid and your consensus numbers with constants excluded.
