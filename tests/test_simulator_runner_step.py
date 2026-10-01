"""`SimulatorRunnerStep` iterates a `models` map and dispatches each
SED-ML job to its simulator's UTC or SteadyState adapter, writing into
the nested `results` store shaped `map[bid, map[sedml_job, map[sim, results]]]`
(simulator innermost). Each leaf is a flat `map[observable -> timeseries]`:
UTC carries the times under the reserved `time` key; steady-state omits `time`
and stores length-1 lists.
"""
from typing import Any, Dict

import pytest
from process_bigraph import allocate_core


def _utc_payload():
    return {
        "result": {
            "time":    [0.0, 0.5, 1.0],
            "columns": ["A", "B"],
            "values":  [[1.0, 0.0], [0.6, 0.4], [0.3, 0.7]],
        }
    }


def _ss_payload():
    return {
        "result": {
            "kind":        "steady_state",
            "time":        None,
            "observables": {"A": 0.25, "B": 0.75},
        }
    }


class _StubUTC:
    def __init__(self, config=None, core=None):
        pass

    def update(self, _state):
        return _utc_payload()


class _StubSS:
    def __init__(self, config=None, core=None):
        pass

    def update(self, _state):
        return _ss_payload()


class _RaisingUTC:
    def __init__(self, config=None, core=None):
        pass

    def update(self, _state):
        raise RuntimeError("simulated COPASI segfault")


@pytest.fixture
def patched_adapters(monkeypatch):
    """Replace the UTC + SS adapter Step classes used by the runner."""
    import viva_biomodels.steps.simulator_runner as mod
    monkeypatch.setattr(mod, "_UTC_CLASS_FOR", lambda name: _StubUTC, raising=True)
    monkeypatch.setattr(mod, "_SS_CLASS_FOR",  lambda name: _StubSS,  raising=True)
    return mod


def test_runner_dispatches_utc_and_steady_state(patched_adapters):
    from viva_biomodels.steps.simulator_runner import SimulatorRunnerStep
    step = SimulatorRunnerStep(
        config={"simulator_name": "copasi"}, core=allocate_core()
    )
    out = step.update({"models": {
        "BIOMD0000000001": {
            "sbml_path": "/tmp/m.xml",
            "sedml_jobs": [
                {"name": "utc1", "kind": "utc",
                 "time": 1.0, "n_points": 3},
                {"name": "ss",   "kind": "steady_state",
                 "time": None, "n_points": None},
            ],
        }
    }})
    results = out["results"]
    bid_results = results["BIOMD0000000001"]
    assert set(bid_results.keys()) == {"utc1", "ss"}
    # simulator is the innermost key; each leaf is a flat observable->timeseries.
    utc_leaf = bid_results["utc1"]["copasi"]
    assert "time" in utc_leaf  # UTC marker
    assert utc_leaf["time"] == [0.0, 0.5, 1.0]
    assert utc_leaf["A"] == [1.0, 0.6, 0.3]
    assert utc_leaf["B"] == [0.0, 0.4, 0.7]
    ss_leaf = bid_results["ss"]["copasi"]
    assert "time" not in ss_leaf  # steady-state marker
    assert ss_leaf == {"A": [0.25], "B": [0.75]}  # length-1 lists


def test_runner_records_per_job_failure(patched_adapters, monkeypatch):
    """A simulator exception leaves an empty results leaf for that job
    (and warns) without aborting the runner's other jobs."""
    import viva_biomodels.steps.simulator_runner as mod
    monkeypatch.setattr(mod, "_UTC_CLASS_FOR", lambda name: _RaisingUTC)
    from viva_biomodels.steps.simulator_runner import SimulatorRunnerStep

    step = SimulatorRunnerStep(
        config={"simulator_name": "copasi"}, core=allocate_core()
    )
    out = step.update({"models": {
        "BIOMD0000000001": {
            "sbml_path": "/tmp/m.xml",
            "sedml_jobs": [
                {"name": "utc1", "kind": "utc", "time": 1.0, "n_points": 3},
                {"name": "ss",   "kind": "steady_state",
                 "time": None, "n_points": None},
            ],
        }
    }})
    bid_results = out["results"]["BIOMD0000000001"]
    # Failed job -> empty leaf under its simulator slot.
    assert bid_results["utc1"]["copasi"] == {}
    # The other job still ran.
    assert bid_results["ss"]["copasi"] == {"A": [0.25], "B": [0.75]}


def test_runner_emits_diagnostics_with_timing_and_provenance(patched_adapters):
    """Alongside results, the runner emits a diagnostics tree with per-run
    timing/status and per-simulator provenance (host, versions, git)."""
    from viva_biomodels.steps.simulator_runner import SimulatorRunnerStep
    step = SimulatorRunnerStep(
        config={"simulator_name": "copasi"}, core=allocate_core()
    )
    out = step.update({"models": {
        "BIOMD0000000001": {
            "sbml_path": "/tmp/m.xml",
            "sedml_jobs": [
                {"name": "utc1", "kind": "utc", "time": 1.0, "n_points": 3},
            ],
        }
    }})
    diag = out["diagnostics"]
    # per-run timing/status, simulator innermost.
    rec = diag["runs"]["BIOMD0000000001"]["utc1"]["copasi"]
    assert rec["status"] == "ok"
    assert isinstance(rec["runtime_s"], float) and rec["runtime_s"] >= 0.0
    # provenance for this simulator + global host meta.
    prov = diag["provenance"]["copasi"]
    assert prov["simulator"] == "copasi"
    assert "started_utc" in prov and "total_runtime_s" in prov
    assert set(diag["meta"]) >= {"host", "platform", "python"}


def test_runner_failure_is_recorded_in_diagnostics(patched_adapters, monkeypatch):
    """A failed job is status='failed' with the error text in diagnostics."""
    import viva_biomodels.steps.simulator_runner as mod
    monkeypatch.setattr(mod, "_UTC_CLASS_FOR", lambda name: _RaisingUTC)
    from viva_biomodels.steps.simulator_runner import SimulatorRunnerStep
    step = SimulatorRunnerStep(
        config={"simulator_name": "copasi"}, core=allocate_core()
    )
    out = step.update({"models": {
        "BIOMD0000000001": {
            "sbml_path": "/tmp/m.xml",
            "sedml_jobs": [{"name": "utc1", "kind": "utc", "time": 1.0, "n_points": 3}],
        }
    }})
    rec = out["diagnostics"]["runs"]["BIOMD0000000001"]["utc1"]["copasi"]
    assert rec["status"] == "failed"
    assert "simulated COPASI segfault" in rec["error"]


def test_runner_writes_one_branch_per_biomodel(patched_adapters):
    """A runner over two biomodels writes both branches into results."""
    from viva_biomodels.steps.simulator_runner import SimulatorRunnerStep
    step = SimulatorRunnerStep(
        config={"simulator_name": "copasi"}, core=allocate_core()
    )
    out = step.update({"models": {
        "BIOMD0000000001": {
            "sbml_path":  "/tmp/m1.xml",
            "sedml_jobs": [{"name": "utc1", "kind": "utc",
                            "time": 1.0, "n_points": 3}],
        },
        "BIOMD0000000002": {
            "sbml_path":  "/tmp/m2.xml",
            "sedml_jobs": [{"name": "utc1", "kind": "utc",
                            "time": 1.0, "n_points": 3}],
        },
    }})
    assert set(out["results"].keys()) == {"BIOMD0000000001", "BIOMD0000000002"}


def test_runner_rejects_unknown_simulator():
    from viva_biomodels.steps.simulator_runner import SimulatorRunnerStep
    with pytest.raises(ValueError, match="unknown simulator"):
        SimulatorRunnerStep(
            config={"simulator_name": "fake-sim"}, core=allocate_core()
        ).update({"models": {}})


def test_effective_n_points_prefers_reference_grid():
    """The reference grid count overrides the job's own n_points so live
    engines sample on the same grid as the reference; absent a reference, the
    job's n_points (or the default) is used unchanged."""
    from viva_biomodels.steps.simulator_runner import effective_n_points

    # No reference override → job's own n_points.
    assert effective_n_points({"n_points": 5}, None) == 5
    # Reference override present → wins.
    assert effective_n_points({"n_points": 5}, 1001) == 1001
    # Neither → structural default.
    assert effective_n_points({}, None) == 2


@pytest.mark.parametrize("engine,module", [("amici", "viva_amici"), ("pysces", "viva_pysces")])
def test_an_engine_that_is_not_installed_is_reported_and_the_run_goes_on(monkeypatch, engine, module):
    """amici and pysces are optional extras. Where one is not installed (an environment built without the extra,
    e.g. a remote container), its runner records every job as `unavailable` with the reason, instead of raising
    and failing the whole composite -- so the other engines' results and comparison still come back."""
    import sys

    from viva_biomodels.steps.simulator_runner import SimulatorRunnerStep

    # A real failed import: Python raises ImportError for a module mapped to None.
    for name in [m for m in sys.modules if m == module or m.startswith(module + ".")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, module, None)
    monkeypatch.delitem(sys.modules, "viva_biomodels.steps.simulators", raising=False)

    step = SimulatorRunnerStep(config={"simulator_name": engine}, core=allocate_core())
    out = step.update({"models": {"BIOMD0000000001": {
        "sbml_path": "/tmp/m.xml",
        "sedml_jobs": [{"name": "utc1", "kind": "utc", "time": 1.0, "n_points": 3},
                       {"name": "ss", "kind": "steady_state", "time": None, "n_points": None}],
    }}})
    runs = out["diagnostics"]["runs"]["BIOMD0000000001"]
    for job in ("utc1", "ss"):
        rec = runs[job][engine]
        assert rec["status"] == "unavailable"
        assert engine in rec["error"] and f"viva-biomodels[{engine}]" in rec["error"]
        assert out["results"]["BIOMD0000000001"][job] == {engine: {}}
