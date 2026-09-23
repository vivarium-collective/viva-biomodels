# batch-compare-biomodels — design

**Date:** 2026-05-28
**Status:** approved, ready for plan

## Problem

The current `compare-biomodel` and `compare-simulators` generators fan out one Step per (simulator × biomodel) and per-biomodel sub-trees of flat top-level keys. For N biomodels and M simulators the composite has roughly `N + N*M + N + 1` Step instances, all with flat suffixed keys (`copasi_BIOMD0000000001`, `tellurium_BIOMD0000000001`, …). Two consequences:

1. Documents are large and hard to read — the structure isn't visible in the graph; it's encoded in key suffixes.
2. The comparison step is hardcoded to two engines (or pre-declared N engine names) and reads from sibling flat keys, so adding a simulator means rewriting wiring.

Additionally, `LoadBiomodelStep` only extracts the first `UniformTimeCourse` simulation from the SED-ML, so biomodels with multiple tasks (UTC variants, or steady-state) are silently truncated; the simulator wrappers themselves only know how to run UTC.

## Goal

A new composite `batch-compare-biomodels` that:

- Has **one Step per simulator**, not one per (simulator × biomodel). The runner iterates jobs internally.
- Supports both **uniform time course** and **steady-state** SED-ML tasks. Each simulator gets a UTC step (already exists) and a new SteadyState step (new work in the sibling `pbg-copasi`, `pbg-tellurium`, `viva-simbio` packages).
- Supports **multiple SED-ML tasks per biomodel** (a biomodel can declare more than one UTC or SS simulation, each with its own duration / n_points).
- Stores results in a self-describing **nested** schema: `results[biomodel_id][simulator][sedml_doc] = simulation_result`.
- Has a **comparison step that reads the full results store** and produces all-pairs nRMSE across whatever simulators produced output, per `(biomodel_id, sedml_doc)`. Generalizes to N simulators automatically.

The existing `compare-biomodel` and `compare-simulators` generators stay untouched as legacy.

## Architecture

```
biomodel_ids[]                                      ┌─ results[bid][copasi]
       │                                            │
       ▼ LoadBiomodelStep × N (one per bid)         ├─ results[bid][tellurium]
models[bid] = {                                     │
  sbml_path,                                        └─ results[bid][simbio]
  sedml_jobs: [{name, kind, time?, n_points?}, …]
}            │
             ▼ SimulatorRunnerStep × M (one per requested simulator)
          results: map[bid, map[sim, map[sedml_doc, simulation_result]]]
             │
             ▼ BatchCompareStep
          comparisons[bid][sedml_doc] = compare_n_engines(...)
             │
             ▼ BatchCompareViz
          viz_html
```

- For 5 biomodels and 3 simulators, this is `5 + 3 + 1 + 1 = 10` Step instances instead of `5 + 15 + 5 + 1 = 26`.
- The `models` store is the dispatcher: per-simulator runners read it and iterate. SED-ML is parsed exactly once per biomodel (by `LoadBiomodelStep`), not once per (simulator, biomodel).
- Per-simulator runners execute jobs **sequentially in-process**, with per-job try/except so one crashy model doesn't abort the simulator's whole slice. Failed jobs record `{kind, error: "..."}` in place of a result.

## Bigraph-schema types

Three new named types, registered in `pbg_biomodels/types.py`:

```python
"simulation_result": {
    "kind":        "string",                       # "utc" | "steady_state"
    "time":        "maybe[list[float]]",           # None for steady_state
    "observables": "tree[any]",                    # name -> list[float] (UTC) | float (SS)
}

"biomodel_jobs": {
    "sbml_path":  "string",
    "sedml_jobs": "list[tree]",                    # each: {name, kind, time?, n_points?}
}

"sim_results_per_biomodel": "map[map[simulation_result]]"
# = simulator -> sedml_doc -> simulation_result
```

The full composite-level shape:

```
state.models:       map[biomodel_id, biomodel_jobs]
state.results:      map[biomodel_id, sim_results_per_biomodel]
state.comparisons:  map[biomodel_id, map[sedml_doc, tree]]
```

`observables` uses `tree[any]` rather than a tagged union because bigraph-schema doesn't have first-class sum types; the `kind` tag tells consumers how to interpret leaves. `time` is `maybe[list[float]]` so the `kind="steady_state"` branch has an explicit `None`.

## New / changed code

### Sibling repos (separate PRs, each via `/pbg-expert --lightweight`)

- `pbg_copasi.processes.CopasiSteadyStateStep` — wraps COPASI's `SteadyStateTask`.
- `pbg_tellurium.processes.TelluriumSteadyStateStep` — wraps `roadrunner.steadyState()`.
- `viva_simbio.processes.SimbioSteadyStateStep` — wraps simbio's steady-state solver.

Each takes `model_source` (SBML path) + simulator-specific knobs and emits `{observables: map[name, float]}`. These are real upstream PRs and block the runner's end-to-end tests; pbg-biomodels' own tests use dummy adapters during the period the upstream work is in flight.

### `pbg-biomodels` workspace

**Modified:**

- `pbg_biomodels/steps/load_biomodel.py` — `LoadBiomodelStep`:
  - Input: `biomodel_id`.
  - Output ports: `sbml_path: string`, `sedml_jobs: list[tree]`.
  - Replaces the `time` / `n_points` outputs.
  - Parses every `SedSimulation` in the SED-ML; emits one job per UTC or SteadyState task. Repeated-task wrappers are skipped with a warning (not expanded — out of scope).
  - Each `sedml_jobs` entry: `{"name": str, "kind": "utc"|"steady_state", "time": float|None, "n_points": int|None}`.

- `pbg_biomodels/steps/simulators.py` — add three adapters parallel to the existing UTC adapters:
  - `BiomodelsCopasiSteadyStateStep`
  - `BiomodelsTelluriumSteadyStateStep`
  - `BiomodelsSimbioSteadyStateStep`

  Each takes runtime input `model_source: string`, delegates to its upstream `<Sim>SteadyStateStep`, and emits `{result: simulation_result}` with `kind="steady_state"`, `time=None`, `observables: map[name, float]`.

- `pbg_biomodels/steps/simulator_comparison.py` — add `BatchCompareStep`:
  - Input: `results` (the full nested map).
  - Output: `comparisons` — `map[bid, map[sedml_doc, tree]]`.
  - For each `(bid, sedml_doc)` it gathers the per-simulator `simulation_result`s, groups by `kind`, and routes to `compare_n_engines_utc` or `compare_n_engines_steady_state`. UTC and SS are not compared cross-kind; both kinds appearing under the same `(bid, sedml_doc)` is a SED-ML pathology — record a warning per-pair and skip.

- `pbg_biomodels/comparison.py` — add `compare_n_engines_steady_state(engines: dict[simulator, dict[observable, float]])`. Metric per shared observable: `|a-b| / max(|a|, |b|, eps)`. Mean across shared observables maps to the existing `bucket_for` thresholds so the bucket vocabulary is the same as UTC.

- `pbg_biomodels/simulators.py` — extend the `SimulatorSpec` registry with `steady_state_step` and `steady_state_adapter` fields per simulator, paralleling the existing `utc_step`. `resolve_simulators` and `utc_step_address` keep their current signatures; add `steady_state_adapter_address(name)`.

**New:**

- `pbg_biomodels/types.py` — register `simulation_result`, `biomodel_jobs`, `sim_results_per_biomodel` with the workspace core. Invoked from `pbg_biomodels.register_types`.

- `pbg_biomodels/steps/simulator_runner.py` — `SimulatorRunnerStep`:
  - Config: `{"simulator_name": "copasi"|"tellurium"|"simbio"}` (validated against `ALL_SIMULATORS`).
  - Input: `models` — `map[biomodel_id, biomodel_jobs]`.
  - Output: `results` — `map[bid, map[sedml_doc, simulation_result]]` (the simulator's slice only).
  - `update()`:
    ```
    out = {}
    for bid, model in (state['models'] or {}).items():
        out[bid] = {}
        for job in model['sedml_jobs']:
            try:
                if job['kind'] == 'utc':
                    inner = <utc adapter for simulator_name>
                    payload = inner.update({model_source, time, n_points})
                    out[bid][job['name']] = utc_result_to_simulation_result(payload)
                else:  # 'steady_state'
                    inner = <ss adapter for simulator_name>
                    payload = inner.update({model_source})
                    out[bid][job['name']] = ss_result_to_simulation_result(payload)
            except Exception as exc:
                out[bid][job['name']] = {
                    "kind": job['kind'],
                    "time": None,
                    "observables": {},
                    "error": f"{type(exc).__name__}: {exc}",
                }
    return {"results": out}
    ```
  - The adapter `Step` instances are constructed inline per-job (same pattern the current `BiomodelsCopasiStep` already uses to call `CopasiUTCStep`).

- `pbg_biomodels/composites/batch_compare_biomodels.py` — `@composite_generator(name="batch-compare-biomodels")`:
  - Parameters:
    - `biomodel_ids: list[string]` (one per line in the dashboard form; default `["BIOMD0000000001"]`).
    - `simulators: list[string]` — description lists the allowed values (`copasi`, `tellurium`, `simbio`); default `["copasi", "tellurium", "simbio"]`; validated by `resolve_simulators`.
  - Generates: one `LoadBiomodelStep` per biomodel id writing into `models[bid]`; one `SimulatorRunnerStep` per requested simulator reading `models` and writing into `results.<sim>`; one `BatchCompareStep` reading `results` and writing `comparisons`; one `BatchCompareOverlay` viz step.
  - `with_emitter=True` attaches a `RAMEmitter` emitting `models`, `results`, `comparisons` as `node` so a CLI run can reconstruct the full payload after `run(0.0)`.

- `pbg_biomodels/visualizations/batch_compare_overlay.py` — `BatchCompareOverlay`:
  - Input: `results: map[bid, map[sim, map[sedml_doc, simulation_result]]]`, `comparisons: map[bid, map[sedml_doc, tree]]`.
  - Output: `html`.
  - Renders the existing card+expand grid. Each biomodel card shows the worst-pair bucket aggregated across its sedml docs. Expanding the card reveals one tab per `sedml_doc`; each tab is a per-observable overlay of all simulators that produced output. SS observables render as a small grouped bar chart of final values instead of a line. Reuses Plotly + the toggle-pane pattern from `compare_overlay.py`.

**Untouched (legacy):**

- `pbg_biomodels/composites/compare_biomodel.py`
- `pbg_biomodels/composites/compare_simulators.py`
- `pbg_biomodels/visualizations/compare_overlay.py`
- The existing two-engine `SimulatorComparisonStep` and the existing `MultiSimulatorComparisonStep` (the latter is conceptually superseded but still wired by `compare-simulators`; leave it).

## Tests

Each adds a single file under `tests/`:

- `test_sedml_multi_task_parsing.py` — `LoadBiomodelStep` returns all UTC + SS tasks for a hand-authored SED-ML fixture with two UTC sims and one steady-state task. Plus one real-biomodel sanity test (single UTC) confirming back-compat of the parsing path.
- `test_simulator_runner_step.py` — runner with `simulator_name="copasi"` over a 2-biomodel `models` fixture produces the expected nested shape; runner over a mixed UTC+SS jobs list dispatches to both step kinds; a per-job exception is captured (`{kind, error}`) without aborting the runner.
- `test_batch_compare_step.py` — `BatchCompareStep` over a 3-simulator nested results fixture produces `comparisons[bid][sedml]` with all-pairs nRMSE; UTC pairs and SS pairs both routed correctly; cross-kind appearance under one `(bid, sedml_doc)` records a warning and skips.
- `test_steady_state_adapters.py` — each `Biomodels<Sim>SteadyStateStep` adapter, given a tiny conservation-law SBML fixture, returns the expected steady-state values. Gated on the upstream SS Steps being available; skipped with a clear xfail message otherwise.
- `test_batch_compare_biomodels_composite.py` — end-to-end generator test (1 biomodel, 2 simulators) producing populated `results` + `comparisons` after `Composite(...).run(0.0)`. Mirrors `tests/test_compare_biomodel_generator.py`.

Existing `tests/test_compare_biomodel_generator.py` and friends are untouched.

## Out of scope

- Parallel job execution inside the runner (ThreadPool / subprocess-isolation / async). Sequential is the v1.
- Reference-CSV "engines" — the existing `compare-simulators` generator already supports those; not re-implementing here.
- Migrating `compare-biomodel` or `compare-simulators` to the new schema.
- Auto-discovering new simulators — the registry in `simulators.py` stays explicit.
- Viz interactivity beyond the existing card-expand + per-sedml tab pattern.
- Repeated-task SED-ML expansion (the SED-ML `SedRepeatedTask` element). Skipped with a warning.

## Risk callouts

- **Upstream SteadyState PRs:** three real pieces of work in sibling repos that block the runner's end-to-end test. Mitigation: per-simulator SS adapters in `pbg_biomodels.steps.simulators` can be wired to a dummy upstream class for pbg-biomodels' own unit tests; the real upstream PRs run in parallel via `/pbg-expert`. `test_steady_state_adapters.py` is the only test gated on real upstream availability.
- **SED-ML SteadyState parsing:** `libsedml` exposes `SedSteadyState`. `extract_first_uniform_time_course` becomes `extract_all_simulations` and must distinguish kinds reliably across libsedml versions. Some BioModels SED-MLs contain repeated-task wrappers — skip with a warning rather than expand.
- **`maybe[list[float]]` wires:** bigraph-schema's `maybe[...]` is well-tested in scalar form but less so on `list[float]` leaves. Worst case: collapse to `list[float]` with `[]` meaning "no time vector" — but the `kind` tag makes that interpretation unambiguous.
- **Dashboard parameter form for `list[string]`:** the dashboard renders `list[string]` as a textarea (one entry per line). `simulators` will look the same; the description string is the only place we can list allowed values. Acceptable for v1; a richer enum widget could come later.

## Decisions locked during brainstorm

- Job execution: **sequential in-process** loop inside each runner Step.
- Result shape: **tagged union** per sedml doc (`{kind, time?, observables}`).
- SS Step location: **sibling repos** (pbg-copasi, pbg-tellurium, viva-simbio).
- Dispatcher: **per-biomodel `LoadBiomodelStep` + index store** (`models`).
- Comparison scope: **per `(biomodel, sedml_doc)`, all-pairs across simulators**, no cross-kind.
- Naming + scope: **new `batch-compare-biomodels` composite alongside legacy**, with SS + multi-sedml + nested schema delivered together.
