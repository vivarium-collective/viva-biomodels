"""SimulatorRunnerStep — runs every SED-ML job of every biomodel under one simulator.

The runner reads a `map[biomodel_id, biomodel_jobs]` input (the `models`
store, populated upstream by per-biomodel `LoadBiomodelStep` instances)
and writes its slice of the nested results store, shaped
`biomodel_id > sedml_job_id > simulator > results`. The simulator is the
*innermost* key, so each per-simulator runner writes a disjoint
`results[bid][job][<this sim>]` leaf and the `tree` apply deep-merges the
runners' slices. One runner instance per simulator replaces the
(simulator x biomodel) Step explosion of the legacy compare-biomodel composite.

Each leaf is a flat `map[observable -> timeseries]` (`results` type): UTC jobs
carry their sample times under the reserved `time` key; steady-state jobs omit
`time` and store length-1 lists. See `viva_biomodels.result_leaf`.

Each job is dispatched in-process, sequentially. A job exception is caught,
logged via `warnings.warn`, and recorded as an empty leaf `{}` under the job's
simulator slot (the comparison/viz consumers skip empty leaves); the runner's
other jobs continue.
"""
from __future__ import annotations

import os
import time
import warnings
from typing import Any, ClassVar, Dict

from process_bigraph import Step

from viva_biomodels import provenance
from viva_biomodels.simulators import ALL_SIMULATORS


# ---------------------------------------------------------------------------
# Adapter class lookup — kept as module-level functions so tests can
# monkeypatch them without standing up the real upstream simulator stack.
# ---------------------------------------------------------------------------


def _UTC_CLASS_FOR(simulator_name: str):
    """Return the UTC adapter class for the given simulator.

    amici has no local Biomodels adapter — viva-amici's AmiciUTCStep already
    speaks the model_source/time/n_points -> {result} contract, so it's used
    directly. Imported lazily because amici is an optional extra.
    """
    if simulator_name == "amici":
        from viva_amici.processes import AmiciUTCStep
        return AmiciUTCStep
    if simulator_name == "pysces":
        from viva_pysces.processes import PyscesUTCStep
        return PyscesUTCStep
    from viva_biomodels.steps.simulators import (
        BiomodelsCopasiStep,
        BiomodelsSimbioStep,
        BiomodelsTelluriumStep,
    )
    return {
        "copasi":    BiomodelsCopasiStep,
        "tellurium": BiomodelsTelluriumStep,
        "simbio":    BiomodelsSimbioStep,
    }[simulator_name]


def _SS_CLASS_FOR(simulator_name: str):
    """Return the SteadyState adapter class for the given simulator.

    As with the UTC class, amici resolves directly to viva-amici's
    AmiciSteadyStateStep (lazy import — optional extra).
    """
    if simulator_name == "amici":
        from viva_amici.processes import AmiciSteadyStateStep
        return AmiciSteadyStateStep
    if simulator_name == "pysces":
        from viva_pysces.processes import PyscesSteadyStateStep
        return PyscesSteadyStateStep
    from viva_biomodels.steps.simulators import (
        BiomodelsCopasiSteadyStateStep,
        BiomodelsSimbioSteadyStateStep,
        BiomodelsTelluriumSteadyStateStep,
    )
    return {
        "copasi":    BiomodelsCopasiSteadyStateStep,
        "tellurium": BiomodelsTelluriumSteadyStateStep,
        "simbio":    BiomodelsSimbioSteadyStateStep,
    }[simulator_name]


def _utc_to_results(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Reshape `{time, columns, values}` -> a UTC `results` leaf.

    Flat `map[observable -> timeseries]` with the sample times under the
    reserved `time` key.
    """
    result = (payload or {}).get("result") or {}
    cols = result.get("columns") or []
    values = result.get("values") or []
    leaf: Dict[str, list] = {"time": list(result.get("time") or [])}
    for sp in cols:
        leaf[sp] = []
    for row in values:
        for j, sp in enumerate(cols):
            leaf[sp].append(float(row[j]))
    return leaf


def _ss_to_results(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Reshape the adapter's steady-state payload -> a steady-state `results` leaf.

    Each observable's scalar becomes a length-1 list so the leaf type is
    uniformly `map[observable -> list[float]]`. No `time` key (which is what
    marks the leaf as steady-state).
    """
    result = (payload or {}).get("result") or {}
    return {sp: [float(v)] for sp, v in (result.get("observables") or {}).items()}


def _scan_point_scalars(subkind: str, sbml_src: str, subtask: Dict[str, Any],
                        n_points: int, utc_cls, ss_cls, rtol, atol,
                        core) -> Dict[str, float]:
    """Run one scan point and reduce it to ``{observable -> scalar}``.

    A UTC subtask reduces to the **endpoint** (last output row); a
    steady-state subtask reduces to its steady-state observables.
    """
    if subkind == "utc":
        payload = utc_cls(core=core).update({
            "model_source": sbml_src,
            "time":         float((subtask or {}).get("time") or 0.0),
            "n_points":     n_points,
            "rtol":         rtol,
            "atol":         atol,
        })
        result = (payload or {}).get("result") or {}
        cols = result.get("columns") or []
        values = result.get("values") or []
        if not values:
            return {c: float("nan") for c in cols}
        last = values[-1]
        return {c: float(last[j]) for j, c in enumerate(cols)}
    # steady-state subtask
    payload = ss_cls(core=core).update({"model_source": sbml_src})
    result = (payload or {}).get("result") or {}
    return {c: float(v) for c, v in (result.get("observables") or {}).items()}


def effective_n_points(job: Dict[str, Any], ref_n_points: Any) -> int:
    """Sample count for a UTC job.

    When a reference grid count is supplied (``ref_grid[bid][job]``), it wins so
    the live engine samples on the same grid as the BioSimulators reference and
    the index-aligned comparison is meaningful. Otherwise the job's own
    ``n_points`` (or the structural default of 2) is used.
    """
    if ref_n_points:
        return int(ref_n_points)
    return int((job or {}).get("n_points") or 2)


class SimulatorRunnerStep(Step):
    """Run every SED-ML job of every biomodel under one configured simulator.

    Config:
        simulator_name: one of `ALL_SIMULATORS` (`copasi`, `tellurium`,
            `simbio`).

    Inputs:
        models: `map[biomodel_id, biomodel_jobs]` — each entry has
            `sbml_path` + `sedml_jobs` (the LoadBiomodelStep output).

        rtol / atol: optional LSODA tolerances forwarded to the simbio adapter
            (ignored by COPASI/Tellurium, which use CVODE). Leave unset to use
            the adapter's CVODE-comparable defaults; tighten for stiff models.

    Outputs:
        results: this simulator's slice of the nested results store, shaped
            `map[biomodel_id, map[sedml_job_id, map[simulator, results]]]` —
            i.e. `{bid: {job: {<this sim>: <results leaf>}}}`.
        diagnostics: this simulator's slice of the diagnostics tree —
            `{"meta": {host, platform, python},
              "provenance": {<sim>: {versions, git commits, started_utc,
                                     total_runtime_s}},
              "runs": {bid: {job: {<sim>: {runtime_s, status, error,
                                           n_points}}}}}`.
    """

    config_schema: ClassVar[Dict[str, Any]] = {
        "simulator_name": {"_type": "string", "_default": "copasi"},
        "rtol":           {"_type": "maybe[float]", "_default": None},
        "atol":           {"_type": "maybe[float]", "_default": None},
    }

    def inputs(self) -> Dict[str, str]:
        # `ref_grid` is the optional reference sample-count map written by
        # LoadReferenceResultsStep (`{bid: {job: n_points}}`); empty when the
        # reference feature is off, in which case the job's own n_points is used.
        return {"models": "map[biomodel_jobs]", "ref_grid": "tree"}

    def outputs(self) -> Dict[str, str]:
        # `tree` (not `biomodel_results = map[map[map[results]]]`) because the
        # runner emits NEW keys at every nesting level — bid + sedml_job + sim
        # are not known to the upstream schema. Map's apply semantics skip
        # update keys that don't already exist in state (no implicit `_add`);
        # tree's apply inserts them AND deep-merges, so per-simulator runners
        # writing disjoint `results[bid][job][sim]` slices merge cleanly. The
        # diagnostics tree merges the same way.
        return {"results": "tree", "diagnostics": "tree"}

    def update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        name = self.config["simulator_name"]
        if name not in ALL_SIMULATORS:
            raise ValueError(
                f"SimulatorRunnerStep: unknown simulator {name!r}; "
                f"known: {ALL_SIMULATORS}"
            )

        utc_cls = _UTC_CLASS_FOR(name)
        ss_cls = _SS_CLASS_FOR(name)
        rtol = self.config.get("rtol")
        atol = self.config.get("atol")

        host = provenance.host_info()
        prov = provenance.simulator_provenance(name)
        t_total = time.perf_counter()

        ref_grid = state.get("ref_grid") or {}

        out: Dict[str, Dict[str, Dict[str, Any]]] = {}
        runs: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for bid, model in (state.get("models") or {}).items():
            out[bid] = {}
            runs[bid] = {}
            ref_for_bid = ref_grid.get(bid) or {}
            sbml_path = (model or {}).get("sbml_path") or ""
            for job in (model or {}).get("sedml_jobs") or []:
                job_name = job.get("name") or "sim"
                kind = job.get("kind")
                n_points = effective_n_points(job, ref_for_bid.get(job_name))
                status, error = "ok", ""
                t0 = time.perf_counter()
                try:
                    if kind == "utc":
                        inner = utc_cls(core=getattr(self, "core", None))
                        payload = inner.update({
                            "model_source": sbml_path,
                            "time":         float(job.get("time") or 0.0),
                            "n_points":     n_points,
                            "rtol":         rtol,
                            "atol":         atol,
                        })
                        leaf = _utc_to_results(payload)
                    elif kind == "steady_state":
                        inner = ss_cls(core=getattr(self, "core", None))
                        payload = inner.update({"model_source": sbml_path})
                        leaf = _ss_to_results(payload)
                    elif kind == "repeated_task":
                        from viva_biomodels.run_biomodels import mutate_sbml
                        subtask = job.get("subtask") or {}
                        subkind = subtask.get("kind") or "utc"
                        scan_values = [float(v) for v in (job.get("scan_values") or [])]
                        sub_n_points = effective_n_points(
                            subtask, ref_for_bid.get(job_name))
                        # One perturbed model per scan point; reduce to a scalar
                        # per observable. A point that fails contributes NaN.
                        scalars_per_point = []
                        for v in scan_values:
                            try:
                                mutated = mutate_sbml(
                                    sbml_path, job["param_id"],
                                    job["param_attr"], v)
                            except Exception as exc:  # bad target/model
                                warnings.warn(
                                    f"SimulatorRunnerStep[{name}]: {bid}/"
                                    f"{job_name} scan@{v} mutate failed: {exc}",
                                    stacklevel=2,
                                )
                                scalars_per_point.append({})
                                continue
                            try:
                                scalars_per_point.append(_scan_point_scalars(
                                    subkind, mutated, subtask, sub_n_points,
                                    utc_cls, ss_cls, rtol, atol,
                                    getattr(self, "core", None)))
                            except Exception as exc:
                                scalars_per_point.append({})
                                warnings.warn(
                                    f"SimulatorRunnerStep[{name}]: {bid}/"
                                    f"{job_name} scan@{v} failed: {exc}",
                                    stacklevel=2,
                                )
                            finally:
                                try:
                                    os.remove(mutated)
                                except OSError:
                                    pass
                        all_obs = sorted(
                            {c for s in scalars_per_point for c in s})
                        leaf = {"scan": scan_values}
                        for c in all_obs:
                            leaf[c] = [float(s.get(c, float("nan")))
                                       for s in scalars_per_point]
                        if not all_obs:
                            status = "failed"
                            error = "all scan points failed"
                    else:
                        status, error = "unknown_kind", f"unknown job kind {kind!r}"
                        warnings.warn(
                            f"SimulatorRunnerStep[{name}]: {bid}/{job_name} "
                            f"has unknown job kind {kind!r}; emitting empty leaf",
                            stacklevel=2,
                        )
                        leaf = {}
                except Exception as exc:
                    status = "failed"
                    error = f"{type(exc).__name__}: {exc}"
                    warnings.warn(
                        f"SimulatorRunnerStep[{name}]: {bid}/{job_name} failed: "
                        f"{error}; emitting empty leaf",
                        stacklevel=2,
                    )
                    leaf = {}
                runtime_s = time.perf_counter() - t0
                # simulator is the innermost key
                out[bid][job_name] = {name: leaf}
                runs[bid][job_name] = {name: {
                    "runtime_s": runtime_s,
                    "status":    status,
                    "error":     error,
                    "n_points":  n_points,
                }}

        diagnostics = {
            "meta": {
                "host":     host["host"],
                "platform": host["platform"],
                "python":   host["python"],
            },
            "provenance": {name: {
                **prov,
                "started_utc":     host["started_utc"],
                "total_runtime_s": time.perf_counter() - t_total,
            }},
            "runs": runs,
        }
        return {"results": out, "diagnostics": diagnostics}
