"""Runtime-input adapter + steady-state adapters for the compare-biomodel composite.

The full-report UTC Steps (species concentrations + reaction fluxes) that drive
COPASI / roadrunner / PySCeS / AMICI now live in their own wrapper repos
(``viva_copasi.processes.CopasiFullReportUTCStep`` etc.), conforming to the
``engine_core`` uniform-time-course contract — the registry points straight at
them. What remains here is engine-agnostic glue:

* :class:`BiomodelsSimbioStep` — a thin runtime-input adapter over
  ``viva_simbio.SimbioUTCStep`` (simbio takes its inputs via config, so this
  feeds ``model_source`` / ``time`` / ``n_points`` through).
* the steady-state adapters, which normalize each engine's steady-state return
  (``_extract_steady_observables``) into the canonical tagged shape.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict

from process_bigraph import Step

from viva_simbio.processes import SimbioUTCStep


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
# Each adapter lazy-imports the upstream `<Sim>SteadyStateStep` so this module
# stays importable even when the upstream package hasn't shipped the steady-state
# class yet. The upstream steps do NOT share one return shape — each drifted to
# its own key (copasi/tellurium nest under "results", simbio under
# "steady_state_concentrations", the engine-core contract under
# "result"/"observables"). `_extract_steady_observables` normalizes all of them
# to a flat {id: float} map; the adapter then wraps that into the
# `simulation_result` tagged shape (kind="steady_state", time=None).
# ---------------------------------------------------------------------------


_STEADY_STATE_INPUTS: Dict[str, str] = {
    "model_source": "string",
}


def _scalar(v: Any) -> float:
    """A steady value: the last element of a one-point series, or the value."""
    if isinstance(v, (list, tuple)):
        return float(v[-1]) if v else 0.0
    return float(v)


def _extract_steady_observables(out: Dict[str, Any]) -> Dict[str, float]:
    """Normalize any upstream steady-state return to a flat {id: float} map.

    Handles the drifted upstream shapes so the comparison layer sees one form:
    - engine-core contract: ``{"result": {"observables": {id: float}}}``
    - copasi / tellurium:    ``{"results": {"species_concentrations": {id: [v]},
                                            "fluxes": {id: [v]}}}``
    - simbio:                ``{"steady_state_concentrations": {id: float}}``
    - legacy flat:           ``{"observables": {id: float}}``
    """
    out = out or {}

    result = out.get("result")
    if isinstance(result, dict) and isinstance(result.get("observables"), dict):
        return {k: _scalar(v) for k, v in result["observables"].items()}

    results = out.get("results")
    if isinstance(results, dict):
        obs: Dict[str, float] = {}
        for group in ("species_concentrations", "fluxes"):
            for k, v in (results.get(group) or {}).items():
                obs[k] = _scalar(v)
        if obs:
            return obs

    for flat_key in ("steady_state_concentrations", "observables"):
        flat = out.get(flat_key)
        if isinstance(flat, dict):
            return {k: _scalar(v) for k, v in flat.items()}

    return {}


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
        from viva_copasi.processes import CopasiSteadyStateStep  # lazy upstream import
        inner = CopasiSteadyStateStep(
            config={"model_source": state["model_source"]}, core=self.core,
        )
        out = inner.update({})
        return _emit_steady_state(_extract_steady_observables(out))


class BiomodelsTelluriumSteadyStateStep(Step):
    """Adapter: SBML path → upstream `TelluriumSteadyStateStep`."""

    config_schema: ClassVar[Dict[str, Any]] = {}

    def inputs(self) -> Dict[str, str]:
        return dict(_STEADY_STATE_INPUTS)

    def outputs(self) -> Dict[str, str]:
        return {"result": "simulation_result"}

    def update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        from viva_tellurium.processes import TelluriumSteadyStateStep  # lazy upstream import
        inner = TelluriumSteadyStateStep(
            config={"model_source": state["model_source"]}, core=self.core,
        )
        out = inner.update({})
        return _emit_steady_state(_extract_steady_observables(out))


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
        return _emit_steady_state(_extract_steady_observables(out))
