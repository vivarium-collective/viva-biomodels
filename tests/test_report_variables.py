"""Unit tests for viva_biomodels.report_variables — SED-ML report-variable
extraction + engine-agnostic constant completion of a UTC results leaf.
"""
from __future__ import annotations

from viva_biomodels.report_variables import (
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
