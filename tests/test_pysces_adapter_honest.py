"""BiomodelsPyscesStep routes integration through viva_pysces._simulate_utc, so a
stiff model that only integrates to all-NaN surfaces as an honest failure (the
exception propagates) instead of an all-NaN "ok" leaf. Monkeypatched so it needs
neither a real stiff model nor Assimulo.
"""
import sys
import types

import pytest
from process_bigraph import allocate_core

# viva_pysces is the optional viva-biomodels[pysces] extra — not installed in the
# base CI env. These tests monkeypatch viva_pysces.processes, so skip the module
# entirely when it isn't importable (mirrors how the runner treats pysces as optional).
pytest.importorskip("viva_pysces", reason="requires the viva-biomodels[pysces] extra")

from viva_biomodels.steps.simulators import BiomodelsPyscesStep


@pytest.fixture
def core():
    c = allocate_core()
    c.register_link("BiomodelsPyscesStep", BiomodelsPyscesStep)
    return c


def test_honest_failure_propagates(core, monkeypatch, tmp_path):
    """When _simulate_utc raises (no finite trajectory), update() must raise too."""
    import viva_pysces.processes as pp

    monkeypatch.setattr(pp, "_load_model", lambda src, quiet=True: object())

    def _boom(model, end, points, quiet):
        raise RuntimeError("PySCeS produced no finite trajectory; CVODE unavailable")

    monkeypatch.setattr(pp, "_simulate_utc", _boom)

    sbml = tmp_path / "m.xml"
    sbml.write_text("<sbml/>")  # never parsed — _load_model is stubbed
    with pytest.raises(RuntimeError, match="no finite trajectory"):
        BiomodelsPyscesStep(core=core).update(
            {"model_source": str(sbml), "time": 10.0, "n_points": 10})


def test_uses_simulate_utc_not_dosim(core, monkeypatch, tmp_path):
    """The adapter drives _simulate_utc (the robust path), not model.doSim directly."""
    import viva_pysces.processes as pp
    import numpy as np

    calls = {"simulate_utc": 0, "dosim": 0}

    class _FakeData:
        def getSpecies(self, lbls=True):
            return (np.array([[0.0, 1.0], [1.0, 0.5]]), ["Time", "X"])

        def getRates(self, lbls=True):
            return (np.array([[0.0], [1.0]]), ["Time"])

    class _FakeModel:
        data_sim = _FakeData()

        def doSim(self, end, points):
            calls["dosim"] += 1

    monkeypatch.setattr(pp, "_load_model", lambda src, quiet=True: _FakeModel())

    def _sim(model, end, points, quiet):
        calls["simulate_utc"] += 1
        return model.data_sim.getSpecies(lbls=True) + ("LSODA",)

    monkeypatch.setattr(pp, "_simulate_utc", _sim)
    # libsbml.readSBML is called for compartments; stub it to return no compartments
    import libsbml
    monkeypatch.setattr(libsbml, "readSBML",
                        lambda p: types.SimpleNamespace(
                            getModel=lambda: types.SimpleNamespace(
                                getListOfCompartments=lambda: [])))

    sbml = tmp_path / "m.xml"
    sbml.write_text("<sbml/>")
    out = BiomodelsPyscesStep(core=core).update(
        {"model_source": str(sbml), "time": 10.0, "n_points": 2})
    assert calls["simulate_utc"] == 1
    assert calls["dosim"] == 0  # doSim is reached only through _simulate_utc
    assert "X" in out["result"]["columns"]
