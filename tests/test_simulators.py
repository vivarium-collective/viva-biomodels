"""Tests for viva_biomodels.steps.simulators wrappers."""
import textwrap

import pytest
from importlib.resources import files

from process_bigraph import allocate_core

from viva_biomodels.steps.simulators import BiomodelsSimbioStep

# The full-report COPASI/tellurium UTC Steps moved into their wrapper repos; this
# suite now exercises them there (same full-report logic, same contract).
from viva_copasi.processes import CopasiFullReportUTCStep as BiomodelsCopasiStep
from viva_tellurium.processes import (
    TelluriumFullReportUTCStep as BiomodelsTelluriumStep,
)
from viva_copasi.processes import CopasiUTCStep
from viva_simbio.processes import SimbioUTCStep
from viva_tellurium.processes import TelluriumUTCStep


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


# --- tellurium concentration convention (reviewer findings 2026-10-28) -------

_SUBSTANCE_UNITS_SBML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
      <model id="amount_species">
        <listOfCompartments>
          <compartment id="c" spatialDimensions="3" size="2" constant="true"/>
        </listOfCompartments>
        <listOfSpecies>
          <species id="S" compartment="c" initialAmount="10"
                   hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
        </listOfSpecies>
      </model>
    </sbml>
    """)


def test_tellurium_reports_species_concentration_not_amount(core, tmp_path):
    """A substance-units species in a size-2 compartment must be reported as a
    CONCENTRATION (amount / size = 5), matching COPASI/PySCeS — not the bare
    amount (10). Guards the compartment-sized rescaling the reviewer flagged."""
    pytest.importorskip("roadrunner")
    model = tmp_path / "amount_species.xml"
    model.write_text(_SUBSTANCE_UNITS_SBML)

    step = BiomodelsTelluriumStep(core=core)
    out = step.update({"model_source": str(model), "time": 1.0, "n_points": 2})
    r = out["result"]
    # column label is the bare id (so it aligns across engines), value is conc.
    assert "S" in r["columns"]
    i = r["columns"].index("S")
    assert r["values"][0][i] == pytest.approx(5.0)
