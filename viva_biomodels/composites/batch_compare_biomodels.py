"""Composite generator: batch-compare-biomodels.

For each biomodel id, fetch the SBML + parse the SED-ML once into a
`biomodel_jobs` record under `models[bid]`. For each requested simulator,
one `SimulatorRunnerStep` iterates `models` and writes its disjoint
`results[bid][job][<sim>]` slice. A single `BatchCompareStep` consumes the
full nested `results` and writes `comparisons`. A single `BatchCompareOverlay`
viz renders the HTML.

Distinct from the legacy `compare-biomodel`:
* one Step per simulator, not per (simulator × biomodel);
* nested `results[biomodel_id][sedml_job_id][simulator]` instead of flat
  suffixed keys (simulator is the *innermost* key, so each per-simulator
  runner writes a disjoint leaf and the `tree` apply deep-merges them);
* each leaf is a `results` type — a flat `map[observable -> timeseries]`;
* supports steady-state tasks (kind="steady_state");
* supports multiple SED-ML simulations per biomodel.

The legacy `compare-biomodel` generator is preserved untouched.
"""
from __future__ import annotations

from typing import Any, Dict, List

from viva_superpowers.composite_generator import composite_generator

from viva_biomodels.simulators import resolve_simulators


def _parquet_emitter_available() -> bool:
    """True when viva-emitters' lightweight parquet emitter is importable."""
    try:
        from viva_emitters import LongFormParquetEmitter  # noqa: F401
        return True
    except Exception:
        return False


def _default_emitter_address() -> str:
    """Default emitter: the corpus-schema parquet emitter when the [parquet-lite]
    extra is installed, else RAMEmitter (so the composite always resolves)."""
    return ("local:LongFormParquetEmitter" if _parquet_emitter_available()
            else "local:RAMEmitter")


LOAD_STEP_ADDRESS    = "local:viva_biomodels.steps.load_biomodel.LoadBiomodelStep"
LOAD_REF_STEP_ADDRESS = "local:viva_biomodels.steps.load_reference_results.LoadReferenceResultsStep"
RUNNER_STEP_ADDRESS  = "local:viva_biomodels.steps.simulator_runner.SimulatorRunnerStep"
COMPARE_STEP_ADDRESS = "local:viva_biomodels.steps.simulator_comparison.BatchCompareStep"
VIZ_STEP_ADDRESS     = "local:viva_biomodels.visualizations.batch_compare_overlay.BatchCompareOverlay"


@composite_generator(
    name="batch-compare-biomodels",
    description=(
        "For each BioModel id, fetch the SBML, parse every SED-ML simulation "
        "(UTC + steady-state), then run each simulator over all jobs via a "
        "single per-simulator runner Step. Comparison reads the full nested "
        "results store and produces all-pairs nRMSE per (biomodel, sedml-doc)."
    ),
    parameters={
        "biomodel_ids": {
            "type": "list[string]",
            "default": ["BIOMD0000000001"],
            "description": (
                "BioModels identifiers, one per line. Each id gets one "
                "LoadBiomodelStep that writes into `models[<id>]`. The special "
                "value `all` expands to the whole corpus (biomodels."
                "get_all_identifiers()); the canonical full run fans this "
                "per-model via viva_biomodels.ray_runner rather than one giant "
                "composite."
            ),
        },
        "simulators": {
            "type": "list[string]",
            "default": ["copasi", "tellurium", "simbio"],
            "description": (
                "Simulators to run, one per line. Allowed values: "
                "copasi, tellurium, simbio. One SimulatorRunnerStep "
                "is created per entry; the runner iterates `models` "
                "and dispatches each job to its UTC or steady-state "
                "adapter."
            ),
        },
        "simbio_rtol": {
            "type": "float",
            "default": 1.0e-6,
            "description": (
                "LSODA relative tolerance for simbio (CVODE-comparable "
                "default). Tighten (e.g. 1e-8) for stiff models where simbio "
                "diverges from COPASI/Tellurium. Ignored by COPASI/Tellurium."
            ),
        },
        "simbio_atol": {
            "type": "float",
            "default": 1.0e-9,
            "description": (
                "LSODA absolute tolerance for simbio. Tighten alongside "
                "simbio_rtol for stiff models."
            ),
        },
        "reference_results_dir": {
            "type": "string",
            "default": "",
            "description": (
                "Path to a BioSimulators SED-ML reference dataset (the dir "
                "containing BIOMD*/<engine>/<version>/.../reports.h5). When set, "
                "each model's reference engines are loaded into the comparison "
                "as `reference:<engine>` and the live engines are sampled on the "
                "reference time grid. Empty (default) disables reference loading."
            ),
        },
        "reference_simulators": {
            "type": "list[string]",
            "default": [],
            "description": (
                "Restrict which reference engines to load, one per line (e.g. "
                "copasi, tellurium). Empty (default) loads every engine present "
                "on disk for each model. Ignored unless reference_results_dir is set."
            ),
        },
        "include_steady_state": {
            "type": "boolean",
            "default": False,
            "description": (
                "Also run a steady-state comparison for every model. Appends a "
                "synthetic steady-state job (when the SED-ML declares none), "
                "dispatched to each engine's SteadyStateStep. BioSimulators "
                "reference data is time-course only, so steady-state jobs compare "
                "live (pbg) engines among themselves."
            ),
        },
        "parquet_out_dir": {
            "type": "string",
            "default": "",
            "description": (
                "When set, the default LongFormParquetEmitter writes the nested "
                "`results` store as a tidy corpus-schema parquet "
                "(biomodel_id, job, engine, variable, time, value) into this "
                "directory as the run is gathered. Empty (default) keeps the "
                "emitter in RAM-only mode (gather_emitter_results still works)."
            ),
        },
    },
    default_n_steps=1,
)
def build_batch_compare_biomodels(
    core: Any = None,
    *,
    biomodel_ids: List[str],
    simulators: List[str],
    simbio_rtol: float = 1.0e-6,
    simbio_atol: float = 1.0e-9,
    reference_results_dir: str = "",
    reference_simulators: List[str] | None = None,
    include_steady_state: bool = False,
    with_emitter: bool = True,
    emitter_address: str = "",
    parquet_out_dir: str = "",
) -> Dict[str, Any]:
    sims = resolve_simulators(simulators)
    # `["all"]` (or "all") means the whole BioModels corpus — mirrors the
    # simulators "all" convention. The full-corpus study declares this so its
    # config honestly reads "all" instead of an implicit single-model default.
    # NOTE: expanding here builds one composite spanning the whole corpus (heavy);
    # the canonical full run instead fans this per-model composite over
    # get_all_identifiers() via viva_biomodels.ray_runner and unions the results.
    if biomodel_ids == "all" or biomodel_ids == ["all"]:
        import biomodels
        biomodel_ids = sorted(biomodels.get_all_identifiers())

    state: Dict[str, Any] = {
        "models":      {bid: {"sbml_path": "", "sedml_jobs": []}
                        for bid in biomodel_ids},
        # biomodel_id > sedml_job_id > simulator > results; the job + simulator
        # levels are inserted by the runners via the `tree` apply.
        "results":     {bid: {} for bid in biomodel_ids},
        # biomodel_id > sedml_job_id > n_points; written by LoadReferenceResultsStep
        # (when reference is on) and read by each runner to align its time grid.
        "ref_grid":    {bid: {} for bid in biomodel_ids},
        "comparisons": {bid: {} for bid in biomodel_ids},
        # host/provenance/per-run timing; runners fill the inner levels.
        "diagnostics": {"meta": {}, "provenance": {},
                        "runs": {bid: {} for bid in biomodel_ids}},
        "viz_html":    "",
    }

    for bid in biomodel_ids:
        state[f"biomodel_id_{bid}"] = bid
        state[f"load_{bid}"] = {
            "_type":   "step",
            "address": LOAD_STEP_ADDRESS,
            "config":  {"auto_steady_state": include_steady_state},
            "inputs":  {"biomodel_id": [f"biomodel_id_{bid}"]},
            "outputs": {
                "sbml_path":  ["models", bid, "sbml_path"],
                "sedml_jobs": ["models", bid, "sedml_jobs"],
            },
        }

    # Reference results are opt-in: only wire the loader (and let it drive the
    # runners' time grid via `ref_grid`) when a dataset dir is given.
    if reference_results_dir:
        state["load_reference"] = {
            "_type":   "step",
            "address": LOAD_REF_STEP_ADDRESS,
            "config":  {
                "reference_results_dir": reference_results_dir,
                "reference_simulators":  list(reference_simulators or []),
            },
            "inputs":  {"models": ["models"]},
            # results[bid][job][reference:<engine>] leaves + ref_grid[bid][job];
            # both deep-merge with the runners' slices via the `tree` apply.
            "outputs": {"results":     ["results"],
                        "ref_grid":    ["ref_grid"],
                        "diagnostics": ["diagnostics"]},
        }

    for sim in sims:
        config: Dict[str, Any] = {"simulator_name": sim}
        # Tolerances only matter for simbio (LSODA); COPASI/Tellurium ignore them.
        if sim == "simbio":
            config["rtol"] = simbio_rtol
            config["atol"] = simbio_atol
        state[f"runner_{sim}"] = {
            "_type":   "step",
            "address": RUNNER_STEP_ADDRESS,
            "config":  config,
            # `ref_grid` lets the runner sample on the reference time grid when
            # reference is on; it's an empty store otherwise (job n_points used).
            "inputs":  {"models": ["models"], "ref_grid": ["ref_grid"]},
            # writes the whole nested tree; the runner only fills its own
            # innermost `results[bid][job][<sim>]` slots, which deep-merge.
            # diagnostics merges the same way.
            "outputs": {"results": ["results"], "diagnostics": ["diagnostics"]},
        }

    state["compare"] = {
        "_type":   "step",
        "address": COMPARE_STEP_ADDRESS,
        "config":  {},
        "inputs":  {"results": ["results"]},
        "outputs": {"comparisons": ["comparisons"]},
    }

    state["batch_overlay_viz"] = {
        "_type":   "step",
        "address": VIZ_STEP_ADDRESS,
        "config":  {
            "title":        "BioModels: batch comparison",
            "biomodel_ids": list(biomodel_ids),
        },
        "inputs":  {"results": ["results"], "comparisons": ["comparisons"],
                    "diagnostics": ["diagnostics"]},
        "outputs": {"html": ["viz_html"]},
    }

    if with_emitter:
        address = emitter_address or _default_emitter_address()
        emitter_config: Dict[str, Any] = {"emit": {
            "models":      "node",
            "results":     "node",
            "ref_grid":    "node",
            "comparisons": "node",
            "diagnostics": "node",
            "viz_html":    "string",
        }}
        # When the default (or requested) emitter is the corpus-schema parquet
        # emitter, configure it to serialize the nested `results` store into the
        # tidy (biomodel_id, job, engine, variable, time, value) layout. It only
        # writes a file when `parquet_out_dir` is set (otherwise it behaves like
        # a RAMEmitter — gather_emitter_results still works either way).
        if address.endswith("LongFormParquetEmitter"):
            emitter_config.update({
                "source_key":  "results",
                "path_levels": ["biomodel_id", "job", "engine"],
                "time_key":    "time",
                "value_dtype": "float32",
                "max_points":  0,
                "write_on_query": bool(parquet_out_dir),
            })
            if parquet_out_dir:
                emitter_config["file_path"] = parquet_out_dir
                emitter_config["parquet_file"] = (
                    f"{biomodel_ids[0]}.parquet" if len(biomodel_ids) == 1
                    else "series.parquet")
        state["emitter"] = {
            "_type":   "step",
            "address": address,
            "config":  emitter_config,
            "inputs":  {
                "models":      ["models"],
                "results":     ["results"],
                "ref_grid":    ["ref_grid"],
                "comparisons": ["comparisons"],
                "diagnostics": ["diagnostics"],
                "viz_html":    ["viz_html"],
            },
        }

    return {"state": state, "run_steps_on_init": True}
