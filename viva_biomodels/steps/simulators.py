"""Thin runtime-input wrappers around pbg-copasi / pbg-tellurium / viva-simbio UTC Steps.

Adapter Steps for the compare-biomodel composite: take ``model_source``,
``time``, ``n_points`` as runtime inputs (so LoadBiomodelStep can feed
them dynamically) and emit the canonical ``numeric_result`` shape on the
``result`` output port. Delegates the actual simulation to the canonical
classes in pbg-tellurium, pbg-copasi, and viva-simbio.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict

from process_bigraph import Step

from pbg_copasi.processes import CopasiUTCStep
from viva_simbio.processes import SimbioUTCStep
from pbg_tellurium.processes import TelluriumUTCStep


_UTC_INPUTS: Dict[str, str] = {
    "model_source": "string",
    "time":         "float",
    "n_points":     "integer",
}


def _validate_n_points(n: Any, where: str) -> int:
    try:
        n = int(n)
    except (TypeError, ValueError):
        raise ValueError(f"{where}: n_points must be an integer >= 2, got {n!r}")
    if n < 2:
        raise ValueError(f"{where}: n_points must be >= 2, got {n}")
    return n


class BiomodelsCopasiStep(Step):
    """Adapter: runtime ``model_source`` → ``pbg_copasi.CopasiUTCStep``.

    CopasiUTCStep.config_schema uses keys: model_source, time, n_points.
    Its update() already returns {'result': {'time', 'columns', 'values'}},
    so no reshape is needed — pass through directly.
    """

    config_schema: ClassVar[Dict[str, Any]] = {}

    def inputs(self) -> Dict[str, str]:
        return dict(_UTC_INPUTS)

    def outputs(self) -> Dict[str, str]:
        return {"result": "numeric_result"}

    def update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        n_points = _validate_n_points(state["n_points"], "BiomodelsCopasiStep")
        inner = CopasiUTCStep(
            config={
                "model_source": state["model_source"],
                "time":         float(state["time"]),
                "n_points":     n_points,
            },
            core=self.core,
        )
        out = inner.update({})
        return {"result": out["result"]}


class BiomodelsTelluriumStep(Step):
    """Adapter: runtime ``model_source`` → ``pbg_tellurium.TelluriumUTCStep``.

    TelluriumUTCStep.config_schema uses keys: model (the SBML/antimony
    source string), model_format, start_time, end_time, n_points.
    Its update() returns {'time_series': list, 'species_trajectories': map[list]};
    this wrapper reshapes to the canonical {'result': {'time', 'columns', 'values'}}.
    """

    config_schema: ClassVar[Dict[str, Any]] = {}

    def inputs(self) -> Dict[str, str]:
        return dict(_UTC_INPUTS)

    def outputs(self) -> Dict[str, str]:
        return {"result": "numeric_result"}

    def update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        n_points = _validate_n_points(state["n_points"], "BiomodelsTelluriumStep")
        inner = TelluriumUTCStep(
            config={
                "model":        state["model_source"],
                "model_format": "sbml",
                "start_time":   0.0,
                "end_time":     float(state["time"]),
                "n_points":     n_points,
            },
            core=self.core,
        )
        out = inner.update({})
        # Reshape from {time_series, species_trajectories} → {result: {time, columns, values}}
        time_list = list(out["time_series"])
        trajectories = out["species_trajectories"]
        columns = list(trajectories.keys())
        n_rows = len(time_list)
        values = [
            [float(trajectories[c][r]) for c in columns]
            for r in range(n_rows)
        ]
        return {
            "result": {
                "time":    time_list,
                "columns": columns,
                "values":  values,
            }
        }


class BiomodelsSimbioStep(Step):
    """Adapter: runtime ``model_source`` → ``viva_simbio.SimbioUTCStep``.

    SimbioUTCStep.config_schema uses keys: model_source, model_format, time,
    n_points. Like the COPASI adapter, its update() already returns
    ``{'result': {'time', 'columns', 'values'}}`` — so this wrapper passes the
    result through directly (no reshape). model_source is an SBML file path,
    which SimbioUTCStep loads via libSBML and rebuilds as a genuine simbio model.
    """

    config_schema: ClassVar[Dict[str, Any]] = {}

    def inputs(self) -> Dict[str, str]:
        return dict(_UTC_INPUTS)

    def outputs(self) -> Dict[str, str]:
        return {"result": "numeric_result"}

    def update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        n_points = _validate_n_points(state["n_points"], "BiomodelsSimbioStep")
        config = {
            "model_source": state["model_source"],
            "model_format": "sbml",
            "time":         float(state["time"]),
            "n_points":     n_points,
        }
        # Optional integration tolerances — tighten for stiff models. When
        # absent, SimbioUTCStep uses its CVODE-comparable defaults (1e-6/1e-9).
        if state.get("rtol") is not None:
            config["rtol"] = float(state["rtol"])
        if state.get("atol") is not None:
            config["atol"] = float(state["atol"])
        inner = SimbioUTCStep(config=config, core=self.core)
        out = inner.update({})
        return {"result": out["result"]}


# ---------------------------------------------------------------------------
# Steady-state adapters
#
# Each adapter lazy-imports the upstream `<Sim>SteadyStateStep` so this
# module stays importable even when the upstream pip package hasn't shipped
# the steady-state class yet. The upstream call is expected to return
# {"observables": {name: float}} (a flat map of final concentrations); the
# adapter wraps that into the `simulation_result` tagged-union shape with
# kind="steady_state" and time=None.
# ---------------------------------------------------------------------------


_STEADY_STATE_INPUTS: Dict[str, str] = {
    "model_source": "string",
}


def _emit_steady_state(observables_map: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "result": {
            "kind":        "steady_state",
            "time":        None,
            "observables": {k: float(v) for k, v in (observables_map or {}).items()},
        }
    }


class BiomodelsCopasiSteadyStateStep(Step):
    """Adapter: SBML path → upstream `CopasiSteadyStateStep`."""

    config_schema: ClassVar[Dict[str, Any]] = {}

    def inputs(self) -> Dict[str, str]:
        return dict(_STEADY_STATE_INPUTS)

    def outputs(self) -> Dict[str, str]:
        return {"result": "simulation_result"}

    def update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        from pbg_copasi.processes import CopasiSteadyStateStep  # lazy upstream import
        inner = CopasiSteadyStateStep(
            config={"model_source": state["model_source"]}, core=self.core,
        )
        out = inner.update({})
        return _emit_steady_state(out.get("observables") or {})


class BiomodelsTelluriumSteadyStateStep(Step):
    """Adapter: SBML path → upstream `TelluriumSteadyStateStep`."""

    config_schema: ClassVar[Dict[str, Any]] = {}

    def inputs(self) -> Dict[str, str]:
        return dict(_STEADY_STATE_INPUTS)

    def outputs(self) -> Dict[str, str]:
        return {"result": "simulation_result"}

    def update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        from pbg_tellurium.processes import TelluriumSteadyStateStep  # lazy upstream import
        inner = TelluriumSteadyStateStep(
            config={"model_source": state["model_source"]}, core=self.core,
        )
        out = inner.update({})
        return _emit_steady_state(out.get("observables") or {})


class BiomodelsSimbioSteadyStateStep(Step):
    """Adapter: SBML path → upstream `SimbioSteadyStateStep`."""

    config_schema: ClassVar[Dict[str, Any]] = {}

    def inputs(self) -> Dict[str, str]:
        return dict(_STEADY_STATE_INPUTS)

    def outputs(self) -> Dict[str, str]:
        return {"result": "simulation_result"}

    def update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        from viva_simbio.processes import SimbioSteadyStateStep  # lazy upstream import
        inner = SimbioSteadyStateStep(
            config={"model_source": state["model_source"]}, core=self.core,
        )
        out = inner.update({})
        return _emit_steady_state(out.get("observables") or {})
