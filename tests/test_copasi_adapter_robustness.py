"""Robustness regression tests for BiomodelsCopasiStep.

Two corpus failure modes that used to abort a COPASI run entirely:

1. Reaction-less models — ``basico.get_reactions()`` returns ``None`` when a
   model has no reactions (dynamics via rateRules), which crashed with
   ``AttributeError: 'NoneType' object has no attribute 'index'``.
2. Species that share a display name across compartments — COPASI disambiguates
   them ("X{c1}"), so selecting the bare "[X]" resolved to nothing and
   ``df[selection]`` raised ``KeyError: [...] not in index``.

Both are built here as tiny synthetic SBML via libsbml (no network), kept apart
from ``test_simulators.py`` so the suite does not require the viva_simbio /
tellurium wrappers to be importable.
"""
import libsbml
import pytest
from process_bigraph import allocate_core

from viva_biomodels.steps.simulators import BiomodelsCopasiStep


def _sbml_string(build) -> str:
    doc = libsbml.SBMLDocument(3, 2)
    model = doc.createModel()
    model.setTimeUnits("second")
    build(model)
    return libsbml.writeSBMLToString(doc)


def _compartment(model, cid):
    c = model.createCompartment()
    c.setId(cid)
    c.setConstant(True)
    c.setSize(1.0)
    c.setSpatialDimensions(3)


def _species(model, sid, comp, *, name=None, amt=1.0, boundary=False):
    s = model.createSpecies()
    s.setId(sid)
    if name is not None:
        s.setName(name)
    s.setCompartment(comp)
    s.setInitialAmount(amt)
    s.setBoundaryCondition(boundary)
    s.setConstant(False)
    s.setHasOnlySubstanceUnits(False)


def _param(model, pid, value):
    p = model.createParameter()
    p.setId(pid)
    p.setValue(value)
    p.setConstant(True)


def _dup_name_model(model):
    """Two compartments, two species that share the display name 'X'."""
    _compartment(model, "c1")
    _compartment(model, "c2")
    _param(model, "k", 0.1)
    _species(model, "X_c1", "c1", name="X", amt=10.0)
    _species(model, "X_c2", "c2", name="X", amt=0.0)
    # X_c1 -> X_c2 with mass-action rate k * X_c1
    r = model.createReaction()
    r.setId("transport")
    r.setReversible(False)
    r.setFast(False)
    rref = r.createReactant()
    rref.setSpecies("X_c1")
    rref.setStoichiometry(1.0)
    rref.setConstant(True)
    pref = r.createProduct()
    pref.setSpecies("X_c2")
    pref.setStoichiometry(1.0)
    pref.setConstant(True)
    kl = r.createKineticLaw()
    kl.setMath(libsbml.parseL3Formula("k * X_c1"))


def _reactionless_model(model):
    """A species driven by a rateRule, with no reactions at all."""
    _compartment(model, "c")
    _param(model, "k", 0.5)
    _species(model, "Y", "c", amt=1.0, boundary=True)
    rule = model.createRateRule()
    rule.setVariable("Y")
    rule.setMath(libsbml.parseL3Formula("-k * Y"))


@pytest.fixture
def core():
    c = allocate_core()
    c.register_link("BiomodelsCopasiStep", BiomodelsCopasiStep)
    return c


def _run(core, sbml_text, tmp_path, name):
    path = tmp_path / f"{name}.xml"
    path.write_text(sbml_text)
    step = BiomodelsCopasiStep(core=core)
    return step.update({"model_source": str(path), "time": 5.0, "n_points": 6})


def test_duplicate_species_names_resolve(core, tmp_path):
    out = _run(core, _sbml_string(_dup_name_model), tmp_path, "dup")
    r = out["result"]
    assert len(r["time"]) == 6
    # Both X species (distinct sbml ids) must survive — the bug dropped them.
    assert set(r["columns"]) >= {"X_c1", "X_c2"}
    assert all(len(row) == len(r["columns"]) for row in r["values"])


def test_reactionless_model_does_not_crash(core, tmp_path):
    # Must not raise AttributeError on get_reactions() -> None.
    out = _run(core, _sbml_string(_reactionless_model), tmp_path, "norxn")
    r = out["result"]
    assert len(r["time"]) == 6
    assert "Y" in r["columns"]
