# corpus_all_engines — all-simulator timecourse release

`corpus_timecourse_all_engines.parquet` is the full BioModels × 5-engine
timecourse corpus in tidy long form. One row per sampled point of a
`(biomodel_id, job, engine, variable)` trace.

## Columns

| column | type | notes |
|---|---|---|
| `biomodel_id` | string | e.g. `BIOMD0000000001` |
| `job` | string | SED-ML job id (`sim1`, `auto_ten_seconds`, …) |
| `engine` | string | `amici` / `copasi` / `pysces` / `simbio` / `tellurium` |
| `variable` | string | SBML id (bare), the reported observable |
| `role` | string | **`species` / `flux` / `parameter` / `compartment`** |
| `time` | float32 | sample time; **NaN marks a steady-state point** |
| `value` | float32 | the reported value; **NaN marks a missing/non-finite point** |

## The `role` column

Each variable is classified by its SBML role
(`viva_biomodels.report_variables.classify_report_variables`):

- `species`, `flux` — the **dynamical** output of a simulation.
- `parameter`, `compartment` — **constant by construction** over a UTC run
  (`report_variables.CONSTANT_ROLES`). The SED-ML report names them, and the
  wrappers fill them so a live engine has full observable coverage against the
  reference — but a flat constant trace is trivially identical across engines.

**To compute cross-engine agreement statistics, set the constant roles aside**
(`role in ("parameter", "compartment")`). Including them dilutes every headline
rate: a constant trace agrees trivially, exactly like a both-zero comparison,
and the constants outnumber the species rows. Keeping them in the release (with
the label) preserves full coverage while letting each consumer exclude them.

## Value encoding

Every non-finite value — `inf`, `-inf`, or a missing point — is stored as a
single `NaN` sentinel in `value`, so there is one encoding for "no finite
value" rather than a NaN/NULL/inf mix.

## Regenerating / annotating

- Fresh runs emit `role` directly (`two_tier.series_table` threads it from each
  model's SBML).
- To annotate an existing release (add `role`, normalize non-finite values):

      python scripts/backfill_variable_roles.py \
          --src <release>.parquet --out <release>.roles.parquet

  Roles come from each model's SBML in the local biomodels pooch cache — no
  simulators are run.
