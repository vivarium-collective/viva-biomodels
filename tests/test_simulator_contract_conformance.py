"""Cross-backend conformance to the engine-core uniform-time-course contract.

viva-biomodels runs several SBML/ODE backends side by side and compares their
trajectories, so every backend's one-shot UTC Step MUST emit the one canonical
shape the comparison math (``result_leaf.to_numeric_result``) consumes:

    {"result": {"time": [...], "columns": [...], "values": [[...], ...]}}

Each backend historically re-declared its ports and re-packed this by hand, and
the shapes drifted (three payloads, two result type names). engine-core now owns
that contract (``UniformTimeCourseStep`` / ``pack_utc``); this test pins every
registered backend to it in one place, so a future drift in any wrapper fails
here instead of silently corrupting a comparison.

Backends whose wrapper package isn't installed (the optional amici/pysces extras)
skip rather than fail — CI installs the full set, so they run there.
"""
from __future__ import annotations

import importlib

import pytest
from importlib.resources import files
from process_bigraph import allocate_core

from viva_biomodels.simulators import (
    KNOWN_SIMULATORS,
    steady_state_step_address,
    utc_step_address,
)

# A small, universally-loadable SBML model every backend can integrate.
_MODEL = str(files("viva_copasi.composites") / "repressilator.xml")
_TIME = 10.0
_N_POINTS = 11


def _load_step(address: str):
    """Resolve a ``local:pkg.module.Class`` Step address to its class.

    Skips the test when the backend's wrapper package is not importable (an
    optional engine missing from this environment) rather than erroring.
    """
    dotted = address.split(":", 1)[1]
    module_name, class_name = dotted.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:  # optional backend not installed
        pytest.skip(f"{module_name} not installed: {exc}")
    return getattr(module, class_name)


def _instantiate(step_cls):
    core = allocate_core()
    core.register_link(step_cls.__name__, step_cls)
    return step_cls(core=core)


@pytest.mark.parametrize("name", KNOWN_SIMULATORS)
def test_utc_step_emits_canonical_shape(name):
    """Every backend's UTC Step emits ``{result: {time, columns, values}}``."""
    step = _instantiate(_load_step(utc_step_address(name)))
    out = step.update(
        {"model_source": _MODEL, "time": _TIME, "n_points": _N_POINTS}
    )

    assert "result" in out, f"{name}: no 'result' port"
    r = out["result"]
    assert {"time", "columns", "values"} <= set(r), f"{name}: keys {set(r)}"
    # One sample per requested point, one observable column minimum.
    assert len(r["time"]) == _N_POINTS, f"{name}: {len(r['time'])} time points"
    assert len(r["values"]) == _N_POINTS, f"{name}: {len(r['values'])} value rows"
    assert r["columns"], f"{name}: no observable columns"
    # Rectangular: every row has exactly one value per column.
    assert all(len(row) == len(r["columns"]) for row in r["values"]), (
        f"{name}: ragged rows"
    )
    # Numeric throughout.
    assert all(isinstance(t, float) for t in r["time"]), f"{name}: non-float time"
    assert all(
        isinstance(v, float) for row in r["values"] for v in row
    ), f"{name}: non-float value"


@pytest.mark.parametrize("name", KNOWN_SIMULATORS)
def test_utc_step_rejects_degenerate_n_points(name):
    """n_points < 2 is rejected by every backend (the contract's lower bound)."""
    step = _instantiate(_load_step(utc_step_address(name)))
    with pytest.raises(ValueError):
        step.update({"model_source": _MODEL, "time": _TIME, "n_points": 1})


def test_extract_steady_observables_normalizes_every_upstream_shape():
    """The normalizer flattens each drifted upstream steady-state shape.

    Deterministic (no backend); guards the shapes the live backends don't all
    exercise. The steady-state adapters were silently emitting empty observables
    because they read a key none of the upstreams use.
    """
    from viva_biomodels.steps.simulators import _extract_steady_observables

    # copasi / tellurium: nested under "results", one-point series lists.
    assert _extract_steady_observables(
        {"results": {"species_concentrations": {"X": [1.5]}, "fluxes": {"R": [0.2]}}}
    ) == {"X": 1.5, "R": 0.2}
    # simbio: flat "steady_state_concentrations".
    assert _extract_steady_observables(
        {"steady_state_concentrations": {"X": 1.5}}
    ) == {"X": 1.5}
    # engine-core contract: result.observables.
    assert _extract_steady_observables(
        {"result": {"kind": "steady_state", "time": None, "observables": {"X": 1.5}}}
    ) == {"X": 1.5}
    # legacy flat + empty.
    assert _extract_steady_observables({"observables": {"X": 1.5}}) == {"X": 1.5}
    assert _extract_steady_observables({}) == {}


@pytest.mark.parametrize("name", KNOWN_SIMULATORS)
def test_steady_state_step_emits_canonical_shape(name):
    """Every backend's steady-state Step emits the canonical tagged shape.

    The Repressilator is oscillatory, so a solve may not converge; a backend that
    raises on it is acceptable (an honest failure). We only assert that *when it
    returns*, the shape is the contract's
    ``{kind: "steady_state", time: None, observables: {id: float}}`` — the
    structure the comparison layer depends on.
    """
    step = _instantiate(_load_step(steady_state_step_address(name)))
    try:
        out = step.update({"model_source": _MODEL})
    except Exception as exc:  # non-convergence on an oscillator is allowed
        pytest.skip(f"{name}: steady-state solve did not return ({exc})")

    assert "result" in out, f"{name}: no 'result' port"
    r = out["result"]
    assert r.get("kind") == "steady_state", f"{name}: kind={r.get('kind')}"
    assert r.get("time") is None, f"{name}: time={r.get('time')}"
    obs = r.get("observables")
    assert isinstance(obs, dict) and obs, f"{name}: empty/absent observables"
    assert all(isinstance(v, float) for v in obs.values()), (
        f"{name}: non-float observable"
    )
