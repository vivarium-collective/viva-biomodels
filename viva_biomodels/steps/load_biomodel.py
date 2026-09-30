"""``LoadBiomodelStep`` — fetch a BioModel by ID and emit SBML path + UTC spec.

Reads ``biomodel_id`` from its single input port, calls
:func:`viva_biomodels.run_biomodels.load_biomodel` (which queries the
``biomodels`` REST API, caches the SBML + SED-ML files locally under
``models/<id>/``, and parses the first UniformTimeCourse simulation out of
the SED-ML), and writes the resolved SBML path, simulation duration, and
output-point count out on three output ports.

Designed to be the entry point of a single-composite pipeline:

    biomodel_id (store)
        ↓ LoadBiomodelStep
    sbml_path / time / n_points (stores)
        ↓ LocalCopasiUTCStep + LocalTelluriumUTCStep (in parallel)
    results[<engine>] (stores)
        ↓ SimulatorComparisonStep / CompareOverlay
    comparison + viz_html
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict

from process_bigraph import Step


class LoadBiomodelStep(Step):
    """Resolve a BioModels identifier to a local SBML file and a list of jobs.

    Inputs:
        biomodel_id: BioModels identifier, e.g. ``"BIOMD0000000001"``.

    Outputs:
        sbml_path: Absolute path to the cached SBML XML file.
        sedml_jobs: List of `{name, kind, time?, n_points?}` records — one
            per UTC + SteadyState simulation declared in the SED-ML.
        time: Back-compat — first UTC job's duration (legacy composites).
        n_points: Back-compat — first UTC job's n_points (legacy composites).

    Side effects:
        Caches the SBML + SED-ML files under ``models/<biomodel_id>/`` in
        the workspace (or current working directory) — same convention
        the bundle's ``run_biomodels`` uses, so the cache is shared.
    """

    config_schema: ClassVar[Dict[str, Any]] = {
        # When true, append a synthetic steady-state job to every model that
        # doesn't already declare one in its SED-ML, so each model gets a
        # steady-state comparison in addition to its time course.
        "auto_steady_state": {"_type": "boolean", "_default": False},
    }

    def inputs(self) -> Dict[str, str]:
        return {"biomodel_id": "string"}

    def outputs(self) -> Dict[str, str]:
        return {
            "sbml_path":  "string",
            "sedml_jobs": "list[tree]",
            "time":       "float",
            "n_points":   "integer",
        }

    def update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Lazy import so loading this module doesn't pull in biomodels'
        # heavy chain when only the type is being inspected.
        import os

        import biomodels

        from viva_biomodels.run_biomodels import (
            extract_all_simulations,
            extract_repeated_tasks,
            load_biomodel,
            read_sedml_doc,
        )

        biomodel_id = state.get("biomodel_id") or ""
        if not biomodel_id:
            raise ValueError(
                "LoadBiomodelStep: input port `biomodel_id` is empty; "
                "set it in the composite state before running."
            )

        meta = biomodels.get_metadata(biomodel_id)
        result = load_biomodel(biomodel_id, meta)

        # Parse every SED-ML simulation for the new sedml_jobs output, plus
        # any 1-D parameter scans (repeatedTask) as `repeated_task` jobs.
        sed_doc = read_sedml_doc(result.sedml_path)
        jobs = extract_all_simulations(sed_doc)
        jobs.extend(extract_repeated_tasks(sed_doc))

        # Optionally guarantee a steady-state job: most BioModels SED-ML declare
        # only a time course, so without this every model is UTC-only. The
        # runner dispatches kind="steady_state" to each engine's SteadyStateStep.
        if self.config.get("auto_steady_state") and not any(
            j.get("kind") == "steady_state" for j in jobs
        ):
            jobs.append({
                "name": "auto_steady_state",
                "kind": "steady_state",
                "time": None,
                "n_points": None,
            })

        # Attach the SED-ML report's constant observables (global parameters +
        # compartments with no rule/initial-assignment) to every UTC job, so the
        # runner can complete each engine's species-only leaf up to the full
        # reference variable set (closes the dominant repeat-match coverage gap).
        try:
            from viva_biomodels.report_variables import constants_for_sbml
            constants = constants_for_sbml(result.sbml_path, sed_doc)
        except Exception:  # never let completion metadata break a load
            constants = {}
        for j in jobs:
            if j.get("kind") == "utc":
                j["constants"] = constants

        # Back-compat outputs — populated from the first UTC job if any.
        first_utc = next((j for j in jobs if j["kind"] == "utc"), None)
        legacy_time = float(first_utc["time"]) if first_utc else float(result.utc.duration)
        legacy_n_points = (
            int(first_utc["n_points"]) if first_utc else int(result.utc.number_of_points)
        )

        return {
            "sbml_path":  os.path.abspath(result.sbml_path),
            "sedml_jobs": jobs,
            "time":       legacy_time,
            "n_points":   legacy_n_points,
        }
