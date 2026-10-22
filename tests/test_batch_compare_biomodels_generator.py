"""The `batch-compare-biomodels` generator builds a composite with one
LoadBiomodelStep per biomodel id, one SimulatorRunnerStep per requested
simulator (not per simulator × biomodel), one BatchCompareStep, and one
BatchCompareOverlay viz step.
"""
import pytest

import viva_biomodels.composites.batch_compare_biomodels  # noqa: F401
from viva_superpowers.composite_generator import _REGISTRY, build_generator


def _entry():
    matches = [e for e in _REGISTRY.values()
               if e.name == "batch-compare-biomodels"]
    assert len(matches) == 1, (
        f"expected 1 generator named 'batch-compare-biomodels', "
        f"got {len(matches)}"
    )
    return matches[0]


def test_generator_is_registered_with_biomodel_and_simulator_params():
    entry = _entry()
    assert "biomodel_ids" in entry.parameters
    assert entry.parameters["biomodel_ids"]["type"] == "list[string]"
    assert "simulators" in entry.parameters
    assert entry.parameters["simulators"]["type"] == "list[string]"


def test_one_load_step_per_biomodel_one_runner_per_simulator():
    """Two biomodels × two simulators → 2 LoadBiomodelStep + 2 SimulatorRunnerStep
    + 1 BatchCompareStep + 1 viz step. Critically NOT one runner per (bid, sim)."""
    doc = build_generator(_entry(), overrides={
        "biomodel_ids": ["BIOMD0000000001", "BIOMD0000000005"],
        "simulators":   ["copasi", "tellurium"],
    })
    state = doc["state"] if "state" in doc else doc

    # One LoadBiomodelStep per biomodel id (keyed `load_<bid>`).
    for bid in ("BIOMD0000000001", "BIOMD0000000005"):
        key = f"load_{bid}"
        assert key in state
        assert state[key]["_type"] == "step"
        # LoadBiomodelStep outputs sbml_path + sedml_jobs into models[bid].
        assert state[key]["outputs"]["sbml_path"] == ["models", bid, "sbml_path"]
        assert state[key]["outputs"]["sedml_jobs"] == ["models", bid, "sedml_jobs"]

    # One SimulatorRunnerStep per simulator (keyed `runner_<sim>`).
    for sim in ("copasi", "tellurium"):
        key = f"runner_{sim}"
        assert key in state
        assert state[key]["_type"] == "step"
        assert state[key]["address"].endswith("SimulatorRunnerStep")
        assert state[key]["config"]["simulator_name"] == sim
        assert state[key]["inputs"]["models"] == ["models"]
        # Runner writes the whole nested tree; it only fills its own innermost
        # results[bid][job][<sim>] slots, which deep-merge across runners.
        assert state[key]["outputs"]["results"] == ["results"]
        # diagnostics (timing/provenance) is wired the same way.
        assert state[key]["outputs"]["diagnostics"] == ["diagnostics"]

    # No (bid, sim) runner keys (the explosion we removed).
    for bid in ("BIOMD0000000001", "BIOMD0000000005"):
        for sim in ("copasi", "tellurium"):
            assert f"runner_{sim}_{bid}" not in state
            assert f"{sim}_step_{bid}" not in state

    # One BatchCompareStep reading the full nested results.
    assert state["compare"]["address"].endswith("BatchCompareStep")
    assert state["compare"]["inputs"]["results"] == ["results"]
    assert state["compare"]["outputs"]["comparisons"] == ["comparisons"]

    # One viz step.
    assert state["batch_overlay_viz"]["address"].endswith("BatchCompareOverlay")
    assert state["batch_overlay_viz"]["inputs"]["results"] == ["results"]
    assert state["batch_overlay_viz"]["inputs"]["comparisons"] == ["comparisons"]


def test_unknown_simulator_rejected_at_build_time():
    with pytest.raises(ValueError, match="Unknown simulator"):
        build_generator(_entry(), overrides={
            "biomodel_ids": ["BIOMD0000000001"],
            "simulators":   ["copasi", "not-a-real-simulator"],
        })


def test_default_simulators_is_all_three():
    doc = build_generator(_entry(), overrides={
        "biomodel_ids": ["BIOMD0000000001"],
    })
    state = doc["state"] if "state" in doc else doc
    for sim in ("copasi", "tellurium", "simbio"):
        assert f"runner_{sim}" in state


def test_simbio_runner_gets_tolerance_config_others_do_not():
    """simbio_rtol/simbio_atol flow only into the simbio runner's config."""
    doc = build_generator(_entry(), overrides={
        "biomodel_ids": ["BIOMD0000000001"],
        "simulators":   ["copasi", "simbio"],
        "simbio_rtol":  1e-8,
        "simbio_atol":  1e-11,
    })
    state = doc["state"] if "state" in doc else doc
    assert state["runner_simbio"]["config"]["rtol"] == 1e-8
    assert state["runner_simbio"]["config"]["atol"] == 1e-11
    # COPASI ignores tolerances — no rtol/atol keys on its runner config.
    assert "rtol" not in state["runner_copasi"]["config"]
    assert "atol" not in state["runner_copasi"]["config"]


def test_legacy_compare_biomodel_generator_still_registered():
    """The new generator does not displace the legacy one."""
    legacy = [e for e in _REGISTRY.values() if e.name == "compare-biomodel"]
    assert len(legacy) == 1


def test_reference_is_off_by_default():
    """No reference_results_dir → no load_reference step; ref_grid seeded empty
    so the runners' ref_grid input has something to read."""
    doc = build_generator(_entry(), overrides={
        "biomodel_ids": ["BIOMD0000000001"],
        "simulators":   ["copasi"],
    })
    state = doc["state"] if "state" in doc else doc
    assert "load_reference" not in state
    assert state["ref_grid"] == {"BIOMD0000000001": {}}


def test_reference_step_wired_when_dir_set():
    """reference_results_dir set → a single load_reference step reading models
    and writing results + ref_grid; every runner reads ref_grid."""
    doc = build_generator(_entry(), overrides={
        "biomodel_ids":         ["BIOMD0000000001"],
        "simulators":           ["copasi", "tellurium"],
        "reference_results_dir": "/data/biosimulators_sedml_results",
    })
    state = doc["state"] if "state" in doc else doc

    assert "load_reference" in state
    ref = state["load_reference"]
    assert ref["config"]["reference_results_dir"] == "/data/biosimulators_sedml_results"
    assert ref["inputs"]["models"] == ["models"]
    assert ref["outputs"]["results"] == ["results"]
    assert ref["outputs"]["ref_grid"] == ["ref_grid"]

    # Each per-simulator runner reads the ref_grid store.
    for sim in ("copasi", "tellurium"):
        assert state[f"runner_{sim}"]["inputs"]["ref_grid"] == ["ref_grid"]


# --- default parquet emitter wiring -----------------------------------------

import viva_biomodels.composites.batch_compare_biomodels as _bc


def test_parquet_out_dir_param_registered():
    assert "parquet_out_dir" in _entry().parameters
    assert _entry().parameters["parquet_out_dir"]["type"] == "string"


def test_default_emitter_is_parquet_when_available(monkeypatch):
    monkeypatch.setattr(_bc, "_parquet_emitter_available", lambda: True)
    assert _bc._default_emitter_address() == "local:LongFormParquetEmitter"


def test_default_emitter_falls_back_to_ram_when_unavailable(monkeypatch):
    monkeypatch.setattr(_bc, "_parquet_emitter_available", lambda: False)
    assert _bc._default_emitter_address() == "local:RAMEmitter"


def _emitter_node(overrides, monkeypatch, available=True):
    monkeypatch.setattr(_bc, "_parquet_emitter_available", lambda: available)
    doc = build_generator(_entry(), overrides=overrides)
    state = doc["state"] if "state" in doc else doc
    return state["emitter"]


def test_parquet_emitter_writes_corpus_schema_when_out_dir_set(monkeypatch, tmp_path):
    em = _emitter_node({"biomodel_ids": ["BIOMD0000000001"],
                        "simulators": ["copasi"],
                        "parquet_out_dir": str(tmp_path)}, monkeypatch)
    assert em["address"] == "local:LongFormParquetEmitter"
    cfg = em["config"]
    assert cfg["source_key"] == "results"
    assert cfg["path_levels"] == ["biomodel_id", "job", "engine"]
    assert cfg["file_path"] == str(tmp_path)
    assert cfg["parquet_file"] == "BIOMD0000000001.parquet"
    assert cfg["write_on_query"] is True


def test_parquet_emitter_ram_only_without_out_dir(monkeypatch):
    em = _emitter_node({"biomodel_ids": ["BIOMD0000000001"],
                        "simulators": ["copasi"]}, monkeypatch)
    assert em["address"] == "local:LongFormParquetEmitter"
    cfg = em["config"]
    assert cfg["write_on_query"] is False
    assert "file_path" not in cfg  # no file written in RAM-only mode


def test_biomodel_ids_all_expands_to_full_corpus(monkeypatch):
    """biomodel_ids=['all'] expands to biomodels.get_all_identifiers()."""
    import biomodels
    monkeypatch.setattr(biomodels, "get_all_identifiers",
                        lambda: ["BIOMD0000000002", "BIOMD0000000001"])
    doc = build_generator(_entry(), overrides={
        "biomodel_ids": ["all"], "simulators": ["copasi"]})
    state = doc["state"] if "state" in doc else doc
    loads = sorted(k for k in state if k.startswith("load_"))
    assert loads == ["load_BIOMD0000000001", "load_BIOMD0000000002"]
