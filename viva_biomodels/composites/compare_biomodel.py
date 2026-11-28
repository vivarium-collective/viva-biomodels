"""Composite generator: fan out across a list of BioModels, compare each
under COPASI and Tellurium, render a summary-card grid of the results.

`compare-biomodel` is a :func:`@composite_generator` — discovered at import
time and served by the dashboard alongside file-based ``*.composite.yaml``
specs. The single user-facing parameter, ``biomodel_ids``, is a
``list[string]``; each id spawns a load → COPASI + Tellurium →
SimulatorComparisonStep branch.

Internal state shape: every per-biomodel store is a *top-level key* suffixed
with the biomodel id (`sbml_path_<bid>`, `copasi_<bid>`, `comparison_<bid>`,
…). All wires are therefore one-segment paths, sidestepping bigraph-schema's
restriction on attaching link paths to string-typed leaves of nested
records. The viz step gets the biomodel_ids list via its `config` and
dynamically declares one named input port per biomodel so the renderer can
reassemble a per-biomodel view at render time.

Internal step addresses are exposed as module constants for Python-level
overriding (tests, alternative simulator backends). They are NOT user-
facing parameters; the dashboard's Configure tab only shows
``biomodel_ids``.
"""
from __future__ import annotations

from typing import Any, Dict, List

from viva_superpowers.composite_generator import composite_generator


LOAD_STEP_ADDRESS = "local:viva_biomodels.steps.load_biomodel.LoadBiomodelStep"
COPASI_STEP_ADDRESS = "local:viva_copasi.processes.CopasiFullReportUTCStep"
TELLURIUM_STEP_ADDRESS = "local:viva_tellurium.processes.TelluriumFullReportUTCStep"
SIMBIO_STEP_ADDRESS = "local:viva_biomodels.steps.simulators.BiomodelsSimbioStep"
COMPARISON_STEP_ADDRESS = "local:viva_biomodels.steps.simulator_comparison.SimulatorComparisonStep"
VISUALIZATION_STEP_ADDRESS = "local:viva_biomodels.visualizations.compare_overlay.CompareOverlay"


def _empty_numeric() -> Dict[str, Any]:
    return {"time": [], "columns": [], "values": []}


@composite_generator(
    name="compare-biomodel",
    description=(
        "For each BioModel id in the list, fetch the SBML, run it under "
        "COPASI and Tellurium, score per-species nRMSE between the two "
        "trajectories, and render a summary-card grid (click a card to "
        "expand that biomodel's species overlay)."
    ),
    parameters={
        "biomodel_ids": {
            "type": "list[string]",
            "default": ["BIOMD0000000001"],
            "description": (
                "BioModels identifiers, one per line. Each id spawns its own "
                "load + COPASI + Tellurium + compare branch."
            ),
        },
    },
    default_n_steps=1,
)
def build_compare_biomodel(
    core: Any = None,
    *,
    biomodel_ids: List[str],
    with_emitter: bool = True,
    load_address: str = LOAD_STEP_ADDRESS,
    copasi_address: str = COPASI_STEP_ADDRESS,
    tellurium_address: str = TELLURIUM_STEP_ADDRESS,
    simbio_address: str = SIMBIO_STEP_ADDRESS,
    comparison_address: str = COMPARISON_STEP_ADDRESS,
    visualization_address: str = VISUALIZATION_STEP_ADDRESS,
    emitter_address: str = "local:RAMEmitter",
) -> Dict[str, Any]:
    """Build a multi-biomodel COPASI-vs-Tellurium comparison composite.

    Args:
        biomodel_ids: BioModels identifiers; one branch is created per id.
        with_emitter: attach a RAMEmitter capturing the per-biomodel
            comparison stores so ``gather_emitter_results`` returns them
            after the run.

    Returns:
        ``{"state", "run_steps_on_init": True}`` ready to feed
        ``process_bigraph.Composite``.
    """
    copasi_key = "copasi"
    tellurium_key = "tellurium"

    state: Dict[str, Any] = {"viz_html": ""}
    viz_inputs: Dict[str, List[str]] = {}
    emit_schema: Dict[str, str] = {}

    for bid in biomodel_ids:
        # Per-biomodel stores — flat top-level keys, suffixed with the id.
        state[f"biomodel_id_{bid}"] = bid
        state[f"sbml_path_{bid}"]   = ""
        state[f"sim_time_{bid}"]    = 0.0
        state[f"n_points_{bid}"]    = 0
        state[f"copasi_{bid}"]      = _empty_numeric()
        state[f"tellurium_{bid}"]   = _empty_numeric()
        state[f"simbio_{bid}"]      = _empty_numeric()
        state[f"comparison_{bid}"]  = {}

        # Per-biomodel step quartet: load → (copasi + tellurium) → compare.
        state[f"load_{bid}"] = {
            "_type":   "step",
            "address": load_address,
            "config":  {},
            "inputs":  {"biomodel_id": [f"biomodel_id_{bid}"]},
            "outputs": {
                "sbml_path": [f"sbml_path_{bid}"],
                "time":      [f"sim_time_{bid}"],
                "n_points":  [f"n_points_{bid}"],
            },
        }

        sim_inputs = {
            "model_source": [f"sbml_path_{bid}"],
            "time":         [f"sim_time_{bid}"],
            "n_points":     [f"n_points_{bid}"],
        }
        state[f"copasi_step_{bid}"] = {
            "_type":   "step",
            "address": copasi_address,
            "config":  {},
            "inputs":  sim_inputs,
            "outputs": {"result": [f"copasi_{bid}"]},
        }
        state[f"tellurium_step_{bid}"] = {
            "_type":   "step",
            "address": tellurium_address,
            "config":  {},
            "inputs":  sim_inputs,
            "outputs": {"result": [f"tellurium_{bid}"]},
        }
        # simbio runs alongside as a third engine; its trajectory feeds the
        # overlay as an extra trace (the nRMSE bucket stays COPASI-vs-Tellurium).
        state[f"simbio_step_{bid}"] = {
            "_type":   "step",
            "address": simbio_address,
            "config":  {},
            "inputs":  sim_inputs,
            "outputs": {"result": [f"simbio_{bid}"]},
        }

        state[f"compare_{bid}"] = {
            "_type":   "step",
            "address": comparison_address,
            "config":  {"engine_a_name": copasi_key, "engine_b_name": tellurium_key},
            "inputs": {
                "engine_a_result": [f"copasi_{bid}"],
                "engine_b_result": [f"tellurium_{bid}"],
            },
            "outputs": {"comparison": [f"comparison_{bid}"]},
        }

        # Per-biomodel viz inputs (the viz declares these dynamically from
        # its config; see CompareOverlay.inputs).
        viz_inputs[f"copasi_{bid}"]     = [f"copasi_{bid}"]
        viz_inputs[f"tellurium_{bid}"]  = [f"tellurium_{bid}"]
        viz_inputs[f"simbio_{bid}"]     = [f"simbio_{bid}"]
        viz_inputs[f"comparison_{bid}"] = [f"comparison_{bid}"]

        # Emitter captures per-biomodel comparison + engine results.
        emit_schema[f"comparison_{bid}"] = "node"
        emit_schema[f"copasi_{bid}"]     = "node"
        emit_schema[f"tellurium_{bid}"]  = "node"
        emit_schema[f"simbio_{bid}"]     = "node"

    # One viz step reads every per-biomodel triplet and renders the grid.
    state["multi_overlay_viz"] = {
        "_type":   "step",
        "address": visualization_address,
        "config":  {
            "title": "BioModels: COPASI vs Tellurium",
            "biomodel_ids": list(biomodel_ids),
        },
        "inputs":  viz_inputs,
        "outputs": {"html": ["viz_html"]},
    }

    if with_emitter:
        state["emitter"] = {
            "_type":   "step",
            "address": emitter_address,
            "config":  {"emit": emit_schema},
            "inputs":  {k: v for k, v in viz_inputs.items()},
        }

    return {"state": state, "run_steps_on_init": True}
