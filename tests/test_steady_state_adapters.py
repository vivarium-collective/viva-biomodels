"""SteadyState adapters delegate to the upstream `<Sim>SteadyStateStep`
classes and reshape the result into the simulation_result tagged-union
shape (kind="steady_state", time=None, observables=map[name, float]).

These tests stub the lazy upstream import so they don't depend on the
real upstream PRs landing first (those are tracked separately).
"""
from typing import Any, Dict

import pytest

from process_bigraph import allocate_core

# Force eager import of the simulators module so its top-level
# `from pbg_copasi.processes import CopasiUTCStep` runs against the real
# upstream module BEFORE `_patch_upstream` ever inserts a stand-in module.
# Without this, the first test would replace `pbg_copasi.processes` in
# sys.modules with an empty stub, then fail at simulator-module load.
import viva_biomodels.steps.simulators  # noqa: F401


def _patch_upstream(monkeypatch, module_path: str, class_name: str, fake_cls):
    """Insert a fake upstream class so the adapter's lazy import succeeds."""
    import sys
    import types

    if module_path in sys.modules:
        mod = sys.modules[module_path]
    else:
        # Create a minimal stand-in package so `from <module_path> import …` works
        # even if the real upstream is missing.
        mod = types.ModuleType(module_path)
        sys.modules[module_path] = mod
    monkeypatch.setattr(mod, class_name, fake_cls, raising=False)


class _FakeSteadyStateClass:
    """Fake upstream class that records its config and returns a canned result."""
    last_config: Dict[str, Any] = {}

    def __init__(self, config=None, core=None):
        type(self).last_config = dict(config or {})

    def update(self, _state):
        return {"observables": {"A": 1.5, "B": 0.5}}


def test_copasi_steady_state_adapter(monkeypatch):
    _patch_upstream(monkeypatch, "pbg_copasi.processes",
                    "CopasiSteadyStateStep", _FakeSteadyStateClass)
    from viva_biomodels.steps.simulators import BiomodelsCopasiSteadyStateStep
    out = BiomodelsCopasiSteadyStateStep(core=allocate_core()).update({"model_source": "/tmp/m.xml"})
    assert out["result"]["kind"] == "steady_state"
    assert out["result"]["time"] is None
    assert out["result"]["observables"] == {"A": 1.5, "B": 0.5}
    assert _FakeSteadyStateClass.last_config["model_source"] == "/tmp/m.xml"


def test_tellurium_steady_state_adapter(monkeypatch):
    _patch_upstream(monkeypatch, "pbg_tellurium.processes",
                    "TelluriumSteadyStateStep", _FakeSteadyStateClass)
    from viva_biomodels.steps.simulators import BiomodelsTelluriumSteadyStateStep
    out = BiomodelsTelluriumSteadyStateStep(core=allocate_core()).update({"model_source": "/tmp/m.xml"})
    assert out["result"]["kind"] == "steady_state"
    assert out["result"]["time"] is None
    assert out["result"]["observables"] == {"A": 1.5, "B": 0.5}


def test_simbio_steady_state_adapter(monkeypatch):
    _patch_upstream(monkeypatch, "viva_simbio.processes",
                    "SimbioSteadyStateStep", _FakeSteadyStateClass)
    from viva_biomodels.steps.simulators import BiomodelsSimbioSteadyStateStep
    out = BiomodelsSimbioSteadyStateStep(core=allocate_core()).update({"model_source": "/tmp/m.xml"})
    assert out["result"]["kind"] == "steady_state"
    assert out["result"]["time"] is None
    assert out["result"]["observables"] == {"A": 1.5, "B": 0.5}


def test_steady_state_adapter_outputs_simulation_result_shape():
    """The output port declaration is `simulation_result`."""
    from viva_biomodels.steps.simulators import BiomodelsCopasiSteadyStateStep
    assert BiomodelsCopasiSteadyStateStep(core=allocate_core()).outputs() == {
        "result": "simulation_result"
    }
