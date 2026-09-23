"""Tests for viva_biomodels.steps.simulators wrappers."""
import pytest
from importlib.resources import files

from process_bigraph import allocate_core

from viva_biomodels.steps.simulators import (
    BiomodelsCopasiStep,
    BiomodelsSimbioStep,
    BiomodelsTelluriumStep,
)
from pbg_copasi.processes import CopasiUTCStep
from viva_simbio.processes import SimbioUTCStep
from pbg_tellurium.processes import TelluriumUTCStep


def _model_path() -> str:
    """Path to a small SBML model — viva-copasi's bundled Repressilator."""
    return str(files('viva_copasi.composites') / 'repressilator.xml')


@pytest.fixture
def core():
    c = allocate_core()
    c.register_link('BiomodelsCopasiStep', BiomodelsCopasiStep)
    c.register_link('BiomodelsTelluriumStep', BiomodelsTelluriumStep)
    c.register_link('BiomodelsSimbioStep', BiomodelsSimbioStep)
    c.register_link('CopasiUTCStep', CopasiUTCStep)
    c.register_link('TelluriumUTCStep', TelluriumUTCStep)
    c.register_link('SimbioUTCStep', SimbioUTCStep)
    return c


def test_biomodels_copasi_step_returns_numeric_result(core):
    step = BiomodelsCopasiStep(core=core)
    out = step.update({
        'model_source': _model_path(),
        'time': 10.0,
        'n_points': 11,
    })
    assert 'result' in out
    r = out['result']
    assert set(r.keys()) >= {'time', 'columns', 'values'}
    assert len(r['time']) == 11
    assert len(r['columns']) > 0
    assert len(r['values']) == 11


def test_biomodels_tellurium_step_returns_numeric_result(core):
    step = BiomodelsTelluriumStep(core=core)
    out = step.update({
        'model_source': _model_path(),
        'time': 10.0,
        'n_points': 11,
    })
    assert 'result' in out
    r = out['result']
    assert set(r.keys()) >= {'time', 'columns', 'values'}
    assert len(r['time']) == 11
    assert len(r['columns']) > 0
    assert len(r['values']) == 11
    # Reshape should produce same-shape rows
    assert all(len(row) == len(r['columns']) for row in r['values'])


def test_biomodels_simbio_step_returns_numeric_result(core):
    step = BiomodelsSimbioStep(core=core)
    out = step.update({
        'model_source': _model_path(),
        'time': 10.0,
        'n_points': 11,
    })
    assert 'result' in out
    r = out['result']
    assert set(r.keys()) >= {'time', 'columns', 'values'}
    assert len(r['time']) == 11
    assert len(r['columns']) > 0
    assert len(r['values']) == 11
    assert all(len(row) == len(r['columns']) for row in r['values'])


def test_biomodels_simbio_step_rejects_n_points_below_two(core):
    step = BiomodelsSimbioStep(core=core)
    with pytest.raises(ValueError, match="n_points must be >= 2"):
        step.update({
            'model_source': _model_path(),
            'time': 10.0,
            'n_points': 1,
        })


def test_biomodels_copasi_step_rejects_n_points_below_two(core):
    step = BiomodelsCopasiStep(core=core)
    with pytest.raises(ValueError, match="n_points must be >= 2"):
        step.update({
            'model_source': _model_path(),
            'time': 10.0,
            'n_points': 1,
        })
