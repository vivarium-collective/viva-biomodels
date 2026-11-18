"""Unit tests for viva_biomodels.report_variables — SED-ML report-variable
extraction + engine-agnostic constant completion of a UTC results leaf.
"""
from __future__ import annotations

import libsbml

from viva_biomodels.report_variables import (
    classify_report_variables,
    complete_utc_leaf,
    sbml_id_from_target,
)


def test_sbml_id_from_target_species_xpath():
    tgt = "/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='BLL']"
    assert sbml_id_from_target(tgt) == "BLL"


def test_sbml_id_from_target_symbol_is_none():
    assert sbml_id_from_target("urn:sedml:symbol:time") is None
    assert sbml_id_from_target(None) is None


def test_complete_adds_missing_constants_as_flat_series():
    leaf = {"time": [0.0, 1.0, 2.0], "BLL": [1.0, 2.0, 3.0]}
    out = complete_utc_leaf(leaf, {"kf_0": 3000.0, "comp1": 1e-16})
    # species untouched
    assert out["BLL"] == [1.0, 2.0, 3.0]
    # constants filled to the time-grid length
    assert out["kf_0"] == [3000.0, 3000.0, 3000.0]
    assert out["comp1"] == [1e-16, 1e-16, 1e-16]


def test_complete_never_overwrites_existing_key():
    leaf = {"time": [0.0, 1.0], "kf_0": [42.0, 42.0]}
    out = complete_utc_leaf(leaf, {"kf_0": 3000.0})
    assert out["kf_0"] == [42.0, 42.0]


def test_complete_noops_on_empty_leaf_or_no_constants():
    assert complete_utc_leaf({}, {"kf_0": 1.0}) == {}
    leaf = {"time": [], "BLL": []}
    assert complete_utc_leaf(leaf, {"kf_0": 1.0}) == leaf  # no time grid → nothing added


def _toy_doc():
    """A minimal SBML document: compartment `cell`, species `S1`/`S2`, parameter
    `k1`, reaction `R1`. `S2` carries a display name that differs from its id.

    Returns the *document* — the caller must keep it alive, since the model it
    owns becomes invalid once the document is freed (libsbml ownership).
    """
    doc = libsbml.SBMLDocument(3, 2)
    m = doc.createModel()
    c = m.createCompartment()
    c.setId("cell")
    c.setSize(2.0)
    for sid in ("S1", "S2"):
        s = m.createSpecies()
        s.setId(sid)
        s.setCompartment("cell")
        s.setInitialConcentration(1.0)
    m.getSpecies("S2").setName("Substrate")
    p = m.createParameter()
    p.setId("k1")
    p.setValue(3.0)
    r = m.createReaction()
    r.setId("R1")
    return doc


def test_classify_report_variables_covers_all_roles():
    doc = _toy_doc()
    roles = classify_report_variables(doc.getModel(), ["S1", "k1", "cell", "R1"])
    assert roles == {
        "S1": "species",
        "k1": "parameter",
        "cell": "compartment",
        "R1": "flux",  # reaction id → time-varying flux/rate output
    }


def test_classify_matches_species_display_name():
    doc = _toy_doc()
    # COPASI emits a species' display name; it must still classify as species.
    assert classify_report_variables(doc.getModel(), ["Substrate"])["Substrate"] == "species"


def test_classify_unknown_id_falls_back_to_flux():
    doc = _toy_doc()
    assert classify_report_variables(doc.getModel(), ["mystery"])["mystery"] == "flux"
