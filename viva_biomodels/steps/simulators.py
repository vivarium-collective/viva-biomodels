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
        # Drive basico/COPASI directly so the time course reports the SED-ML
        # report's full variable set — floating species AND reaction fluxes —
        # keyed by SBML id (the reference's labels), rather than the species-only
        # output of the upstream CopasiUTCStep. Same engine the BioSimulators
        # COPASI reference used, so the fluxes repeat-match natively. Constant
        # parameters/compartments are completed downstream (report_variables).
        import basico

        basico.load_model(state["model_source"])
        sp = basico.get_species()
        rx = basico.get_reactions()
        sp_name_to_id = dict(zip(sp.index, sp["sbml_id"]))
        rx_name_to_id = dict(zip(rx.index, rx["sbml_id"]))

        selection = ["Time"]
        out_ids = ["time"]
        for name, sid in sp_name_to_id.items():
            selection.append(f"[{name}]")
            out_ids.append(str(sid))
        for name, sid in rx_name_to_id.items():
            selection.append(f"({name}).Flux")
            out_ids.append(str(sid))

        df = basico.run_time_course_with_output(
            output_selection=selection,
            duration=float(state["time"]),
            intervals=n_points - 1,
            use_initial_values=True,
        )
        df = df[selection]  # guarantee column order matches out_ids
        rows = df.values.tolist()
        time_list = [float(r[0]) for r in rows]
        columns = out_ids[1:]
        values = [[float(v) for v in r[1:]] for r in rows]
        return {"result": {"time": time_list, "columns": columns, "values": values}}


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
        # Drive roadrunner directly so we can select the SED-ML report's full
        # variable set — floating species AND reaction rates (fluxes) — rather
        # than the species-only output of the upstream TelluriumUTCStep. This is
        # the same engine (libroadrunner) the BioSimulators tellurium reference
        # used, so the reaction rates repeat-match the reference natively. Global
        # parameters + compartments (constant) are completed downstream from the
        # SBML (see viva_biomodels.report_variables).
        import roadrunner

        rr = roadrunner.RoadRunner(state["model_source"])
        species = list(rr.model.getFloatingSpeciesIds())
        reactions = list(rr.model.getReactionIds())
        rr.selections = ["time"] + species + reactions
        arr = rr.simulate(0.0, float(state["time"]), n_points)
        colnames = list(arr.colnames)
        data = [[float(v) for v in row] for row in arr]
        # First column is time; the rest are the selected observables.
        time_list = [row[0] for row in data]
        columns = colnames[1:]
        values = [row[1:] for row in data]
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


class BiomodelsPyscesStep(Step):
    """Adapter: runtime ``model_source`` → PySCeS, emitting the full report set.

    Drives PySCeS directly so the results carry the SED-ML report's variable
    set — floating species (``getSpecies``) AND reaction fluxes (``getRates``) —
    plus each compartment's PySCeS value. PySCeS works in concentration space, so
    it reports compartments as their normalized value (typically ``1``) and
    fluxes in concentration/time; the BioSimulators PySCeS reference did the same,
    so both repeat-match natively (this is why a uniform SBML-size compartment
    fill mismatched the pysces reference — PySCeS's own value is the faithful one).
    """

    config_schema: ClassVar[Dict[str, Any]] = {}

    def inputs(self) -> Dict[str, str]:
        return dict(_UTC_INPUTS)

    def outputs(self) -> Dict[str, str]:
        return {"result": "numeric_result"}

    def update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        import libsbml
        from viva_pysces.processes import _load_model

        n_points = _validate_n_points(state["n_points"], "BiomodelsPyscesStep")
        model = _load_model(state["model_source"], quiet=True)
        model.doSim(end=float(state["time"]), points=n_points)

        sarr, slabels = model.data_sim.getSpecies(lbls=True)   # Time + species ids
        rarr, rlabels = model.data_sim.getRates(lbls=True)     # Time + reaction ids
        sp_cols = [str(c) for c in slabels[1:]]
        rx_cols = [str(c) for c in rlabels[1:]]

        # Compartments: PySCeS exposes each as a model attribute (its normalized
        # concentration-space value). Emit them so they repeat-match the pysces
        # reference rather than a uniform SBML-size fill.
        sd = libsbml.readSBML(state["model_source"])
        comp_ids = [c.getId() for c in sd.getModel().getListOfCompartments()]
        comp_vals = {cid: float(getattr(model, cid)) for cid in comp_ids
                     if getattr(model, cid, None) is not None}

        times = [float(row[0]) for row in sarr]
        n_rows = len(times)
        columns = sp_cols + rx_cols + list(comp_vals.keys())
        values = []
        for i in range(n_rows):
            row = [float(sarr[i][j + 1]) for j in range(len(sp_cols))]
            row += [float(rarr[i][j + 1]) for j in range(len(rx_cols))]
            row += [comp_vals[cid] for cid in comp_vals]
            values.append(row)
        return {"result": {"time": times, "columns": columns, "values": values}}


class BiomodelsAmiciStep(Step):
    """Adapter: runtime ``model_source`` → AMICI, emitting the full report set.

    Runs AMICI (newer sundials API) and emits floating species (``rdata.x`` /
    state ids) AND reaction fluxes (``rdata.w`` for the model's ``flux_<Rid>``
    expressions, renamed to the bare SBML reaction id). Same engine the
    BioSimulators AMICI reference used, so both repeat-match natively. Constant
    parameters/compartments are completed downstream (report_variables).
    """

    config_schema: ClassVar[Dict[str, Any]] = {}

    def inputs(self) -> Dict[str, str]:
        return dict(_UTC_INPUTS)

    def outputs(self) -> Dict[str, str]:
        return {"result": "numeric_result"}

    def update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np
        from amici.sim.sundials import run_simulation
        from viva_amici.processes import _configure_solver, _load_model

        n_points = _validate_n_points(state["n_points"], "BiomodelsAmiciStep")
        horizon = float(state["time"])
        _mod, model = _load_model(
            {"sbml_file": state["model_source"], "antimony": "", "sbml": ""}
        )
        solver = model.create_solver()
        _configure_solver(solver, self.config)

        state_ids = list(model.get_state_ids())
        model.set_initial_state(list(model.get_initial_state()))
        timepoints = [horizon * i / (n_points - 1) for i in range(n_points)]
        model.set_t0(0.0)
        model.set_timepoints([float(t) for t in timepoints])

        rdata = run_simulation(model, solver)
        if int(rdata.status) != 0:
            raise RuntimeError(
                f"AMICI UTC integration failed: status={int(rdata.status)} "
                f"over horizon={horizon} with n_points={n_points}"
            )

        xs = rdata.x
        # Reaction fluxes live in the expression vector rdata.w as flux_<Rid>.
        expr_ids = list(model.get_expression_ids()) if hasattr(model, "get_expression_ids") else []
        flux = [(i, eid[len("flux_"):]) for i, eid in enumerate(expr_ids)
                if eid.startswith("flux_")]
        w = getattr(rdata, "w", None)
        wnp = np.asarray(w) if w is not None else None

        columns = list(state_ids) + [rid for _, rid in flux]
        values = []
        for r in range(len(timepoints)):
            row = [float(xs[r][c]) for c in range(len(state_ids))]
            if wnp is not None:
                row += [float(wnp[r][i]) for i, _ in flux]
            values.append(row)
        return {"result": {"time": list(timepoints), "columns": columns, "values": values}}


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
