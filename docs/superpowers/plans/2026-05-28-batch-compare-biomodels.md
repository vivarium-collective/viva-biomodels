# batch-compare-biomodels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `batch-compare-biomodels` composite generator that runs N biomodels under M simulators using one Step per simulator (not per simulator × biomodel), with a self-describing nested `results[bid][simulator][sedml_doc]` schema covering both uniform time course and steady-state SED-ML tasks.

**Architecture:** New code lives alongside the existing `compare-biomodel` generator (which stays untouched as legacy). Each per-biomodel `LoadBiomodelStep` writes a `biomodel_jobs` record into a shared `models` map; M `SimulatorRunnerStep` instances (one per requested simulator) iterate the map and dispatch each job to a per-simulator UTC or SteadyState adapter; a single `BatchCompareStep` reads the full nested results and produces an all-pairs nRMSE matrix per `(biomodel, sedml_doc)`.

**Tech Stack:** Python 3.12, `process-bigraph`, `bigraph-schema`, `libsedml`, `libsbml`, `pbg-copasi` / `pbg-tellurium` / `viva-simbio` (sibling packages), `plotly` (viz), `pytest`.

**Spec:** `docs/superpowers/specs/2026-05-28-batch-compare-biomodels-design.md`

**Upstream dependency (out of scope for this plan):** SteadyState Step classes `CopasiSteadyStateStep`, `TelluriumSteadyStateStep`, `SimbioSteadyStateStep` in the sibling pip packages are tracked separately (each as its own `/pbg-expert` invocation). The adapters in this plan import them lazily, and the tests in this plan monkeypatch the upstream import — the workspace work is **not** blocked on those upstream PRs.

---

## File layout

**New files in `pbg_biomodels/`:**
- `types.py` — register new bigraph-schema types into the workspace core.
- `steps/simulator_runner.py` — `SimulatorRunnerStep`.
- `composites/batch_compare_biomodels.py` — `@composite_generator` for the new composite.
- `visualizations/batch_compare_overlay.py` — `BatchCompareOverlay` visualization.

**Modified files in `pbg_biomodels/`:**
- `__init__.py` — re-export new types/steps; call new type registration.
- `run_biomodels.py` — add `extract_all_simulations` alongside the existing `extract_first_uniform_time_course`.
- `steps/load_biomodel.py` — replace `time`/`n_points` outputs with `sbml_path` + `sedml_jobs`.
- `steps/simulators.py` — add three SteadyState adapter Steps + helpers, no changes to existing UTC adapters.
- `steps/simulator_comparison.py` — add `BatchCompareStep`.
- `comparison.py` — add `compare_n_engines_steady_state`.
- `simulators.py` — extend `SimulatorSpec` and registry with `steady_state` adapter addresses.
- `steps/__init__.py` — re-export the new step classes.

**New test files in `tests/`:**
- `test_simulation_result_types.py`
- `test_sedml_multi_task_parsing.py`
- `test_load_biomodel_step_sedml_jobs.py`
- `test_steady_state_adapters.py`
- `test_simulator_runner_step.py`
- `test_batch_compare_step.py`
- `test_batch_compare_overlay.py`
- `test_batch_compare_biomodels_generator.py`

Each new file has one clear responsibility; existing files are only touched where they need a small, additive change. The legacy `compare_biomodel.py`, `compare_simulators.py`, `compare_overlay.py`, and `test_compare_biomodel_generator.py` are **not** modified.

---

## Task A1: Register new bigraph-schema types

**Files:**
- Create: `pbg_biomodels/types.py`
- Modify: `pbg_biomodels/__init__.py` (call `register_simulation_types` from `register_types`)
- Test: `tests/test_simulation_result_types.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_simulation_result_types.py`:

```python
"""The batch-compare schema registers `simulation_result`, `biomodel_jobs`,
and `sim_results_per_biomodel` so composites can wire stores by name.
"""
from process_bigraph import allocate_core

from pbg_biomodels import register_types


def _core():
    return register_types(allocate_core())


def test_simulation_result_type_registered():
    core = _core()
    # access_schema returns the resolved schema for a registered type name;
    # raises KeyError-ish if missing.
    schema = core.access("simulation_result")
    assert schema is not None
    assert "kind" in schema or "_type" in schema  # records are dict-shaped


def test_biomodel_jobs_type_registered():
    core = _core()
    schema = core.access("biomodel_jobs")
    assert schema is not None


def test_sim_results_per_biomodel_alias_registered():
    core = _core()
    schema = core.access("sim_results_per_biomodel")
    assert schema is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/eranagmon/code/pbg-biomodels
.venv/bin/python -m pytest tests/test_simulation_result_types.py -v
```

Expected: FAIL — `core.access("simulation_result")` returns None / unresolved.

- [ ] **Step 3: Create `pbg_biomodels/types.py`**

```python
"""Bigraph-schema types for the batch-compare-biomodels composite.

`simulation_result` is the per-(biomodel, simulator, sed-ml-doc) leaf written
by `SimulatorRunnerStep`. The `kind` field tags the union: UTC payloads carry
a `time` vector and per-observable list-of-floats; steady-state payloads carry
`time=None` and per-observable scalars.

`biomodel_jobs` is the per-biomodel record written by `LoadBiomodelStep` —
the SBML path plus the list of simulation tasks parsed out of the SED-ML
(each task is a free-form tree with `name`, `kind`, optional `time`,
optional `n_points`).
"""
from __future__ import annotations


SIMULATION_TYPES = {
    "simulation_result": {
        "kind":        "string",
        "time":        "maybe[list[float]]",
        "observables": "tree[any]",
    },
    "biomodel_jobs": {
        "sbml_path":  "string",
        "sedml_jobs": "list[tree]",
    },
    "sim_results_per_biomodel": "map[map[simulation_result]]",
}


def register_simulation_types(core):
    """Register the batch-compare types into a process-bigraph core."""
    core.register_types(SIMULATION_TYPES)
    return core
```

- [ ] **Step 4: Wire `register_simulation_types` into the workspace `register_types`**

In `pbg_biomodels/__init__.py`, after the existing `TYPES_DICT` block and before the `from pbg_biomodels.steps import …` line, change:

```python
def register_types(core):
    """Register pbg-biomodels bigraph-schema types into a ProcessBigraph core."""
    core.register_types(TYPES_DICT)
    return core
```

to:

```python
from pbg_biomodels.types import register_simulation_types  # noqa: E402


def register_types(core):
    """Register pbg-biomodels bigraph-schema types into a ProcessBigraph core."""
    core.register_types(TYPES_DICT)
    register_simulation_types(core)
    return core
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_simulation_result_types.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add pbg_biomodels/types.py pbg_biomodels/__init__.py tests/test_simulation_result_types.py
git commit -m "$(cat <<'EOF'
types: register simulation_result + biomodel_jobs for batch-compare schema

`simulation_result` is the tagged-union leaf produced by each per-simulator
runner Step (UTC carries a time vector and per-observable trace; steady-state
carries time=None and per-observable scalars). `biomodel_jobs` is the
LoadBiomodelStep output record: SBML path + the SED-ML task list.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task A2: SED-ML multi-task extraction

**Files:**
- Modify: `pbg_biomodels/run_biomodels.py` (add `extract_all_simulations` after the existing `extract_first_uniform_time_course`, no behavior change to the existing function)
- Test: `tests/test_sedml_multi_task_parsing.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sedml_multi_task_parsing.py`:

```python
"""`extract_all_simulations` returns every UTC + SteadyState task in a
SED-ML document, in declaration order, with `kind` tagged so downstream
dispatch can branch.
"""
from pathlib import Path
import textwrap

import libsedml
import pytest

from pbg_biomodels.run_biomodels import extract_all_simulations, read_sedml_doc


def _write(tmp_path, body) -> str:
    p = tmp_path / "sample.sedml"
    p.write_text(textwrap.dedent(body))
    return str(p)


def test_single_utc_task_yields_one_job(tmp_path):
    path = _write(tmp_path, """\
    <?xml version="1.0" encoding="UTF-8"?>
    <sedML xmlns="http://sed-ml.org/sed-ml/level1/version3" level="1" version="3">
      <listOfSimulations>
        <uniformTimeCourse id="sim1" initialTime="0" outputStartTime="0"
                           outputEndTime="10" numberOfPoints="100"/>
      </listOfSimulations>
    </sedML>
    """)
    jobs = extract_all_simulations(read_sedml_doc(path))
    assert len(jobs) == 1
    assert jobs[0]["kind"] == "utc"
    assert jobs[0]["name"] == "sim1"
    assert jobs[0]["time"] == pytest.approx(10.0)
    assert jobs[0]["n_points"] == 100


def test_mixed_utc_and_steady_state(tmp_path):
    path = _write(tmp_path, """\
    <?xml version="1.0" encoding="UTF-8"?>
    <sedML xmlns="http://sed-ml.org/sed-ml/level1/version3" level="1" version="3">
      <listOfSimulations>
        <uniformTimeCourse id="utc_short" initialTime="0" outputStartTime="0"
                           outputEndTime="1" numberOfPoints="10"/>
        <steadyState id="ss"/>
        <uniformTimeCourse id="utc_long" initialTime="0" outputStartTime="0"
                           outputEndTime="100" numberOfPoints="500"/>
      </listOfSimulations>
    </sedML>
    """)
    jobs = extract_all_simulations(read_sedml_doc(path))
    kinds = [j["kind"] for j in jobs]
    names = [j["name"] for j in jobs]
    assert kinds == ["utc", "steady_state", "utc"]
    assert names == ["utc_short", "ss", "utc_long"]
    assert jobs[1]["time"] is None
    assert jobs[1]["n_points"] is None


def test_no_simulations_returns_empty_list(tmp_path):
    path = _write(tmp_path, """\
    <?xml version="1.0" encoding="UTF-8"?>
    <sedML xmlns="http://sed-ml.org/sed-ml/level1/version3" level="1" version="3">
      <listOfSimulations/>
    </sedML>
    """)
    assert extract_all_simulations(read_sedml_doc(path)) == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_sedml_multi_task_parsing.py -v
```

Expected: FAIL — `ImportError: cannot import name 'extract_all_simulations'`.

- [ ] **Step 3: Add `extract_all_simulations` to `pbg_biomodels/run_biomodels.py`**

Find the existing function `extract_first_uniform_time_course` (around line 152). Immediately **after** it, add:

```python
def extract_all_simulations(sed_doc: libsedml.SedDocument) -> list:
    """Yield every UTC + SteadyState simulation in the SED-ML, in order.

    Returns a list of `{name, kind, time?, n_points?}` dicts. `kind` is
    `"utc"` or `"steady_state"`. Steady-state entries have `time=None` and
    `n_points=None`. Anything that isn't UTC or steady-state (repeated-task
    wrappers, custom simulations, …) is skipped with a UserWarning.
    """
    import warnings

    jobs = []
    n_sims = int(sed_doc.getNumSimulations())
    for i in range(n_sims):
        sim = sed_doc.getSimulation(i)
        if sim is None:
            continue
        sim_id = sim.getId() or f"sim_{i}"

        is_utc = False
        is_ss = False
        if hasattr(sim, "isSedUniformTimeCourse"):
            try:
                is_utc = bool(sim.isSedUniformTimeCourse())
            except Exception:
                is_utc = False
        if hasattr(sim, "isSedSteadyState"):
            try:
                is_ss = bool(sim.isSedSteadyState())
            except Exception:
                is_ss = False

        if is_utc:
            jobs.append({
                "name":     sim_id,
                "kind":     "utc",
                "time":     float(sim.getOutputEndTime() - sim.getOutputStartTime()),
                "n_points": int(sim.getNumberOfPoints()),
            })
        elif is_ss:
            jobs.append({
                "name":     sim_id,
                "kind":     "steady_state",
                "time":     None,
                "n_points": None,
            })
        else:
            warnings.warn(
                f"extract_all_simulations: skipping unsupported simulation "
                f"{sim_id!r} (not UTC or steady-state)",
                stacklevel=2,
            )

    return jobs
```

Note: do **not** modify `extract_first_uniform_time_course` — the legacy `LoadBiomodelStep` flow currently uses it and is being replaced in Task A3, but the function itself stays in place for any other caller.

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_sedml_multi_task_parsing.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pbg_biomodels/run_biomodels.py tests/test_sedml_multi_task_parsing.py
git commit -m "$(cat <<'EOF'
sedml: extract_all_simulations — yield every UTC + SteadyState task

Adds a multi-task extractor next to the existing
extract_first_uniform_time_course. Returns {name, kind, time?, n_points?}
records ordered by SED-ML declaration. Repeated-task wrappers and other
non-UTC/non-SS simulations are skipped with a warning rather than expanded
(out of scope for batch-compare v1).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task A3: Upgrade `LoadBiomodelStep` to emit `sbml_path` + `sedml_jobs`

**Files:**
- Modify: `pbg_biomodels/steps/load_biomodel.py`
- Test: `tests/test_load_biomodel_step_sedml_jobs.py`

The previous output ports `time` and `n_points` are **removed**: the legacy `compare-biomodel` generator already populates per-biomodel `sim_time_<bid>` / `n_points_<bid>` stores itself, but it does so by wiring `LoadBiomodelStep`'s `time` / `n_points` outputs directly. We must verify no other caller is wired to these names before removing them.

- [ ] **Step 1: Verify the existing legacy generator's wiring will break (and that this is expected)**

```bash
grep -rn '"time"\|"n_points"' pbg_biomodels/composites/ | grep -v __pycache__
```

Expected: lines in `compare_biomodel.py` and `compare_simulators.py` referencing `time`/`n_points` outputs of the load step. **These legacy composites must be updated to call the old behavior some other way OR we keep the old fields on the step.** To minimise blast radius, we keep `time`/`n_points` on the step as **back-compat outputs** (populated from the first UTC job) AND add the new `sedml_jobs` output. The legacy generators continue to work; the new generator reads `sedml_jobs`.

- [ ] **Step 2: Write the failing test**

Create `tests/test_load_biomodel_step_sedml_jobs.py`:

```python
"""`LoadBiomodelStep` emits a `sedml_jobs` list alongside `sbml_path`,
covering every UTC + SteadyState task in the SED-ML.
"""
import pytest

from pbg_biomodels.steps.load_biomodel import LoadBiomodelStep


def test_outputs_declare_sedml_jobs():
    """The step's output schema includes the new `sedml_jobs` port."""
    step = LoadBiomodelStep()
    outs = step.outputs()
    assert "sbml_path" in outs
    assert "sedml_jobs" in outs
    assert outs["sedml_jobs"] == "list[tree]"
    # Back-compat ports stay (populated from the first UTC job).
    assert "time" in outs
    assert "n_points" in outs


@pytest.mark.network
def test_emits_sedml_jobs_for_real_biomodel(tmp_path, monkeypatch):
    """End-to-end: a real BioModels id parses into at least one UTC job.

    Network-gated: only run when `BIOMODELS_NETWORK_TESTS=1` is set so
    CI without network access doesn't fail.
    """
    import os
    if not os.environ.get("BIOMODELS_NETWORK_TESTS"):
        pytest.skip("set BIOMODELS_NETWORK_TESTS=1 to run network tests")

    monkeypatch.chdir(tmp_path)
    step = LoadBiomodelStep()
    out = step.update({"biomodel_id": "BIOMD0000000001"})
    assert isinstance(out["sbml_path"], str) and out["sbml_path"].endswith(".xml")
    jobs = out["sedml_jobs"]
    assert isinstance(jobs, list) and len(jobs) >= 1
    assert all("kind" in j and "name" in j for j in jobs)
    assert any(j["kind"] == "utc" for j in jobs)
```

- [ ] **Step 3: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_load_biomodel_step_sedml_jobs.py::test_outputs_declare_sedml_jobs -v
```

Expected: FAIL — `'sedml_jobs' in outs` is False.

- [ ] **Step 4: Update `LoadBiomodelStep` to add `sedml_jobs`**

In `pbg_biomodels/steps/load_biomodel.py`, replace the `outputs` and `update` methods. After modification, the full class body is:

```python
class LoadBiomodelStep(Step):
    """Resolve a BioModels identifier to a local SBML file and a list of jobs.

    Inputs:
        biomodel_id: BioModels identifier, e.g. ``"BIOMD0000000001"``.

    Outputs:
        sbml_path: Absolute path to the cached SBML XML file.
        sedml_jobs: List of `{name, kind, time?, n_points?}` records — one
            per UTC + SteadyState simulation declared in the SED-ML.
        time: Back-compat — first UTC job's duration (legacy composites).
        n_points: Back-compat — first UTC job's n_points (legacy composites).

    Side effects:
        Caches the SBML + SED-ML files under ``models/<biomodel_id>/`` in
        the workspace (or current working directory) — same convention
        the bundle's ``run_biomodels`` uses, so the cache is shared.
    """

    config_schema: ClassVar[Dict[str, str]] = {}

    def inputs(self) -> Dict[str, str]:
        return {"biomodel_id": "string"}

    def outputs(self) -> Dict[str, str]:
        return {
            "sbml_path":  "string",
            "sedml_jobs": "list[tree]",
            "time":       "float",
            "n_points":   "integer",
        }

    def update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Lazy import so loading this module doesn't pull in biomodels'
        # heavy chain when only the type is being inspected.
        import os

        import biomodels

        from pbg_biomodels.run_biomodels import (
            extract_all_simulations,
            load_biomodel,
            read_sedml_doc,
        )

        biomodel_id = state.get("biomodel_id") or ""
        if not biomodel_id:
            raise ValueError(
                "LoadBiomodelStep: input port `biomodel_id` is empty; "
                "set it in the composite state before running."
            )

        meta = biomodels.get_metadata(biomodel_id)
        result = load_biomodel(biomodel_id, meta)

        # Parse every SED-ML simulation for the new sedml_jobs output.
        sed_doc = read_sedml_doc(result.sedml_path)
        jobs = extract_all_simulations(sed_doc)

        # Back-compat outputs — populated from the first UTC job if any.
        first_utc = next((j for j in jobs if j["kind"] == "utc"), None)
        legacy_time = float(first_utc["time"]) if first_utc else float(result.utc.duration)
        legacy_n_points = (
            int(first_utc["n_points"]) if first_utc else int(result.utc.number_of_points)
        )

        return {
            "sbml_path":  os.path.abspath(result.sbml_path),
            "sedml_jobs": jobs,
            "time":       legacy_time,
            "n_points":   legacy_n_points,
        }
```

- [ ] **Step 5: Run the new test + the legacy generator test to confirm nothing regresses**

```bash
.venv/bin/python -m pytest tests/test_load_biomodel_step_sedml_jobs.py::test_outputs_declare_sedml_jobs tests/test_compare_biomodel_generator.py -v
```

Expected: all tests PASS (the legacy `compare-biomodel` generator is unaffected because the back-compat outputs still exist).

- [ ] **Step 6: Commit**

```bash
git add pbg_biomodels/steps/load_biomodel.py tests/test_load_biomodel_step_sedml_jobs.py
git commit -m "$(cat <<'EOF'
LoadBiomodelStep: emit sedml_jobs list alongside back-compat time/n_points

New output port `sedml_jobs: list[tree]` carries every UTC + SteadyState
task in the biomodel's SED-ML. The existing `time` and `n_points` ports
are preserved (populated from the first UTC job) so the legacy
compare-biomodel composite keeps working unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task B1: Extend simulator registry with SteadyState addresses

**Files:**
- Modify: `pbg_biomodels/simulators.py` (add `steady_state_step` to `SimulatorSpec`, populate it for all three simulators, add `steady_state_step_address` helper)
- Test: covered by Task B2's tests — no dedicated test for registry data.

- [ ] **Step 1: Modify `SimulatorSpec` and the `_SIMULATORS` registry in `pbg_biomodels/simulators.py`**

Change the `SimulatorSpec` TypedDict:

```python
class SimulatorSpec(TypedDict):
    utc_step: str
    steady_state_step: str   # NEW
    process: str
    process_config: Callable[[str], Dict[str, object]]
    species_out: str
```

Update each entry in `_SIMULATORS` to include the new field:

```python
_SIMULATORS: Dict[str, SimulatorSpec] = {
    "copasi": {
        "utc_step": "local:pbg_biomodels.steps.simulators.BiomodelsCopasiStep",
        "steady_state_step": "local:pbg_biomodels.steps.simulators.BiomodelsCopasiSteadyStateStep",
        "process": "local:CopasiUTCProcess",
        "process_config": lambda sbml: {
            "model_source": sbml,
            "time": 1.0,
            "intervals": 10,
        },
        "species_out": "species_concentrations",
    },
    "tellurium": {
        "utc_step": "local:pbg_biomodels.steps.simulators.BiomodelsTelluriumStep",
        "steady_state_step": "local:pbg_biomodels.steps.simulators.BiomodelsTelluriumSteadyStateStep",
        "process": "local:TelluriumProcess",
        "process_config": lambda sbml: {"model_file": sbml},
        "species_out": "species",
    },
    "simbio": {
        "utc_step": "local:pbg_biomodels.steps.simulators.BiomodelsSimbioStep",
        "steady_state_step": "local:pbg_biomodels.steps.simulators.BiomodelsSimbioSteadyStateStep",
        "process": "local:SimbioUTCProcess",
        "process_config": lambda sbml: {"model_source": sbml, "model_format": "sbml"},
        "species_out": "species_concentrations",
    },
}
```

At the bottom of the file, next to `utc_step_address`, add:

```python
def steady_state_step_address(name: str) -> str:
    return _SIMULATORS[name]["steady_state_step"]
```

- [ ] **Step 2: Sanity-check the imports still work**

```bash
.venv/bin/python -c "from pbg_biomodels.simulators import steady_state_step_address; print(steady_state_step_address('copasi'))"
```

Expected: prints `local:pbg_biomodels.steps.simulators.BiomodelsCopasiSteadyStateStep`.

- [ ] **Step 3: Commit (registry-only; the actual classes land in B2)**

```bash
git add pbg_biomodels/simulators.py
git commit -m "$(cat <<'EOF'
simulators: register steady_state_step addresses + helper

Each simulator's spec now declares a steady_state_step address parallel
to its existing utc_step. SimulatorRunnerStep dispatches on job.kind to
the right one. The address classes land in B2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task B2: Add SteadyState adapter Step classes

**Files:**
- Modify: `pbg_biomodels/steps/simulators.py` (add three SteadyState adapter classes alongside the existing UTC adapters)
- Modify: `pbg_biomodels/steps/__init__.py` (re-export them)
- Test: `tests/test_steady_state_adapters.py`

The adapters lazy-import the upstream `<Sim>SteadyStateStep` so the workspace doesn't break-import when those upstream classes are not yet shipped. Tests patch the lazy import.

- [ ] **Step 1: Write the failing test**

Create `tests/test_steady_state_adapters.py`:

```python
"""SteadyState adapters delegate to the upstream `<Sim>SteadyStateStep`
classes and reshape the result into the simulation_result tagged-union
shape (kind="steady_state", time=None, observables=map[name, float]).

These tests stub the lazy upstream import so they don't depend on the
real upstream PRs landing first (those are tracked separately).
"""
from typing import Any, Dict

import pytest


def _patch_upstream(monkeypatch, module_path: str, class_name: str, fake_cls):
    """Insert a fake upstream class so the adapter's lazy import succeeds."""
    import importlib
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
    from pbg_biomodels.steps.simulators import BiomodelsCopasiSteadyStateStep
    out = BiomodelsCopasiSteadyStateStep().update({"model_source": "/tmp/m.xml"})
    assert out["result"]["kind"] == "steady_state"
    assert out["result"]["time"] is None
    assert out["result"]["observables"] == {"A": 1.5, "B": 0.5}
    assert _FakeSteadyStateClass.last_config["model_source"] == "/tmp/m.xml"


def test_tellurium_steady_state_adapter(monkeypatch):
    _patch_upstream(monkeypatch, "pbg_tellurium.processes",
                    "TelluriumSteadyStateStep", _FakeSteadyStateClass)
    from pbg_biomodels.steps.simulators import BiomodelsTelluriumSteadyStateStep
    out = BiomodelsTelluriumSteadyStateStep().update({"model_source": "/tmp/m.xml"})
    assert out["result"]["kind"] == "steady_state"
    assert out["result"]["time"] is None
    assert out["result"]["observables"] == {"A": 1.5, "B": 0.5}


def test_simbio_steady_state_adapter(monkeypatch):
    _patch_upstream(monkeypatch, "viva_simbio.processes",
                    "SimbioSteadyStateStep", _FakeSteadyStateClass)
    from pbg_biomodels.steps.simulators import BiomodelsSimbioSteadyStateStep
    out = BiomodelsSimbioSteadyStateStep().update({"model_source": "/tmp/m.xml"})
    assert out["result"]["kind"] == "steady_state"
    assert out["result"]["time"] is None
    assert out["result"]["observables"] == {"A": 1.5, "B": 0.5}


def test_steady_state_adapter_outputs_simulation_result_shape():
    """The output port declaration is `simulation_result`."""
    from pbg_biomodels.steps.simulators import BiomodelsCopasiSteadyStateStep
    assert BiomodelsCopasiSteadyStateStep().outputs() == {
        "result": "simulation_result"
    }
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_steady_state_adapters.py -v
```

Expected: FAIL — `ImportError: cannot import name 'BiomodelsCopasiSteadyStateStep'`.

- [ ] **Step 3: Add the SteadyState adapters to `pbg_biomodels/steps/simulators.py`**

At the end of `pbg_biomodels/steps/simulators.py`, append:

```python
# ---------------------------------------------------------------------------
# Steady-state adapters
#
# Each adapter lazy-imports the upstream `<Sim>SteadyStateStep` so this
# module stays importable even when the upstream pip package hasn't shipped
# the steady-state class yet. The upstream call is expected to return
# {"observables": {name: float}} (a flat map of final concentrations); the
# adapter wraps that into the `simulation_result` tagged-union shape with
# kind="steady_state" and time=None.
# ---------------------------------------------------------------------------


_STEADY_STATE_INPUTS: Dict[str, str] = {
    "model_source": "string",
}


def _emit_steady_state(observables_map: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "result": {
            "kind":        "steady_state",
            "time":        None,
            "observables": {k: float(v) for k, v in (observables_map or {}).items()},
        }
    }


class BiomodelsCopasiSteadyStateStep(Step):
    """Adapter: SBML path → upstream `CopasiSteadyStateStep`."""

    config_schema: ClassVar[Dict[str, Any]] = {}

    def inputs(self) -> Dict[str, str]:
        return dict(_STEADY_STATE_INPUTS)

    def outputs(self) -> Dict[str, str]:
        return {"result": "simulation_result"}

    def update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        from pbg_copasi.processes import CopasiSteadyStateStep  # lazy upstream import
        inner = CopasiSteadyStateStep(
            config={"model_source": state["model_source"]}, core=self.core,
        )
        out = inner.update({})
        return _emit_steady_state(out.get("observables") or {})


class BiomodelsTelluriumSteadyStateStep(Step):
    """Adapter: SBML path → upstream `TelluriumSteadyStateStep`."""

    config_schema: ClassVar[Dict[str, Any]] = {}

    def inputs(self) -> Dict[str, str]:
        return dict(_STEADY_STATE_INPUTS)

    def outputs(self) -> Dict[str, str]:
        return {"result": "simulation_result"}

    def update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        from pbg_tellurium.processes import TelluriumSteadyStateStep  # lazy upstream import
        inner = TelluriumSteadyStateStep(
            config={"model_source": state["model_source"]}, core=self.core,
        )
        out = inner.update({})
        return _emit_steady_state(out.get("observables") or {})


class BiomodelsSimbioSteadyStateStep(Step):
    """Adapter: SBML path → upstream `SimbioSteadyStateStep`."""

    config_schema: ClassVar[Dict[str, Any]] = {}

    def inputs(self) -> Dict[str, str]:
        return dict(_STEADY_STATE_INPUTS)

    def outputs(self) -> Dict[str, str]:
        return {"result": "simulation_result"}

    def update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        from viva_simbio.processes import SimbioSteadyStateStep  # lazy upstream import
        inner = SimbioSteadyStateStep(
            config={"model_source": state["model_source"]}, core=self.core,
        )
        out = inner.update({})
        return _emit_steady_state(out.get("observables") or {})
```

- [ ] **Step 4: Re-export from `pbg_biomodels/steps/__init__.py`**

Replace the file's contents with:

```python
"""Process-bigraph Steps contributed by pbg-biomodels."""

from pbg_biomodels.steps.load_biomodel import LoadBiomodelStep
from pbg_biomodels.steps.simulator_comparison import SimulatorComparisonStep
from pbg_biomodels.steps.simulators import (
    BiomodelsCopasiStep,
    BiomodelsCopasiSteadyStateStep,
    BiomodelsSimbioStep,
    BiomodelsSimbioSteadyStateStep,
    BiomodelsTelluriumStep,
    BiomodelsTelluriumSteadyStateStep,
)

__all__ = [
    "LoadBiomodelStep",
    "SimulatorComparisonStep",
    "BiomodelsCopasiStep",
    "BiomodelsCopasiSteadyStateStep",
    "BiomodelsTelluriumStep",
    "BiomodelsTelluriumSteadyStateStep",
    "BiomodelsSimbioStep",
    "BiomodelsSimbioSteadyStateStep",
]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_steady_state_adapters.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add pbg_biomodels/steps/simulators.py pbg_biomodels/steps/__init__.py tests/test_steady_state_adapters.py
git commit -m "$(cat <<'EOF'
simulators: SteadyState adapters for copasi / tellurium / simbio

Adds three Biomodels<Sim>SteadyStateStep adapters that lazy-import the
upstream `<Sim>SteadyStateStep` classes and reshape the result into the
simulation_result tagged-union shape (kind=steady_state, time=None,
observables=map[name, float]). Lazy import lets pbg-biomodels keep
working while the upstream PRs are in flight.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task C1: SimulatorRunnerStep

**Files:**
- Create: `pbg_biomodels/steps/simulator_runner.py`
- Modify: `pbg_biomodels/steps/__init__.py` (re-export `SimulatorRunnerStep`)
- Test: `tests/test_simulator_runner_step.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_simulator_runner_step.py`:

```python
"""`SimulatorRunnerStep` iterates a `models` map and dispatches each
SED-ML job to its simulator's UTC or SteadyState adapter, writing into
a nested `results` map shaped `map[bid, map[sedml_doc, simulation_result]]`.
"""
from typing import Any, Dict

import pytest


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
    import pbg_biomodels.steps.simulator_runner as mod
    monkeypatch.setattr(mod, "_UTC_CLASS_FOR", lambda name: _StubUTC, raising=True)
    monkeypatch.setattr(mod, "_SS_CLASS_FOR",  lambda name: _StubSS,  raising=True)
    return mod


def test_runner_dispatches_utc_and_steady_state(patched_adapters):
    from pbg_biomodels.steps.simulator_runner import SimulatorRunnerStep
    step = SimulatorRunnerStep(config={"simulator_name": "copasi"})
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
    assert bid_results["utc1"]["kind"] == "utc"
    assert bid_results["utc1"]["time"] == [0.0, 0.5, 1.0]
    assert bid_results["utc1"]["observables"]["A"] == [1.0, 0.6, 0.3]
    assert bid_results["utc1"]["observables"]["B"] == [0.0, 0.4, 0.7]
    assert bid_results["ss"]["kind"] == "steady_state"
    assert bid_results["ss"]["time"] is None
    assert bid_results["ss"]["observables"] == {"A": 0.25, "B": 0.75}


def test_runner_records_per_job_failure(patched_adapters, monkeypatch):
    """A simulator exception is recorded as `{kind, error}` for that job
    and does not abort the runner's other jobs."""
    import pbg_biomodels.steps.simulator_runner as mod
    monkeypatch.setattr(mod, "_UTC_CLASS_FOR", lambda name: _RaisingUTC)
    from pbg_biomodels.steps.simulator_runner import SimulatorRunnerStep

    step = SimulatorRunnerStep(config={"simulator_name": "copasi"})
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
    assert bid_results["utc1"]["kind"] == "utc"
    assert "error" in bid_results["utc1"]
    assert "simulated COPASI segfault" in bid_results["utc1"]["error"]
    # The other job still ran.
    assert bid_results["ss"]["kind"] == "steady_state"
    assert bid_results["ss"]["observables"] == {"A": 0.25, "B": 0.75}


def test_runner_writes_one_branch_per_biomodel(patched_adapters):
    """A runner over two biomodels writes both branches into results."""
    from pbg_biomodels.steps.simulator_runner import SimulatorRunnerStep
    step = SimulatorRunnerStep(config={"simulator_name": "copasi"})
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
    from pbg_biomodels.steps.simulator_runner import SimulatorRunnerStep
    with pytest.raises(ValueError, match="unknown simulator"):
        SimulatorRunnerStep(config={"simulator_name": "fake-sim"}).update(
            {"models": {}}
        )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_simulator_runner_step.py -v
```

Expected: FAIL — `ImportError: cannot import name 'SimulatorRunnerStep'`.

- [ ] **Step 3: Create `pbg_biomodels/steps/simulator_runner.py`**

```python
"""SimulatorRunnerStep — runs every SED-ML job of every biomodel under one simulator.

The runner reads a `map[biomodel_id, biomodel_jobs]` input (the `models`
store, populated upstream by per-biomodel `LoadBiomodelStep` instances)
and writes a `map[bid, map[sedml_doc_name, simulation_result]]` slice
keyed by the simulator this runner is configured for. One runner instance
per simulator replaces the (simulator × biomodel) Step explosion of the
legacy compare-biomodel composite.

Each job is dispatched in-process, sequentially. A job exception is caught
and recorded as `{"kind", "time": None, "observables": {}, "error": "..."}`
under the job's name; the runner's other jobs continue.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict

from process_bigraph import Step

from pbg_biomodels.simulators import ALL_SIMULATORS


# ---------------------------------------------------------------------------
# Adapter class lookup — kept as module-level functions so tests can
# monkeypatch them without standing up the real upstream simulator stack.
# ---------------------------------------------------------------------------


def _UTC_CLASS_FOR(simulator_name: str):
    """Return the UTC adapter class for the given simulator."""
    from pbg_biomodels.steps.simulators import (
        BiomodelsCopasiStep,
        BiomodelsSimbioStep,
        BiomodelsTelluriumStep,
    )
    return {
        "copasi":    BiomodelsCopasiStep,
        "tellurium": BiomodelsTelluriumStep,
        "simbio":    BiomodelsSimbioStep,
    }[simulator_name]


def _SS_CLASS_FOR(simulator_name: str):
    """Return the SteadyState adapter class for the given simulator."""
    from pbg_biomodels.steps.simulators import (
        BiomodelsCopasiSteadyStateStep,
        BiomodelsSimbioSteadyStateStep,
        BiomodelsTelluriumSteadyStateStep,
    )
    return {
        "copasi":    BiomodelsCopasiSteadyStateStep,
        "tellurium": BiomodelsTelluriumSteadyStateStep,
        "simbio":    BiomodelsSimbioSteadyStateStep,
    }[simulator_name]


def _utc_to_simulation_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Reshape `{time, columns, values}` → simulation_result tagged-union."""
    result = (payload or {}).get("result") or {}
    cols = result.get("columns") or []
    values = result.get("values") or []
    observables: Dict[str, list] = {sp: [] for sp in cols}
    for row in values:
        for j, sp in enumerate(cols):
            observables[sp].append(float(row[j]))
    return {
        "kind":        "utc",
        "time":        list(result.get("time") or []),
        "observables": observables,
    }


def _ss_to_simulation_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Pass through the adapter's already-shaped simulation_result."""
    result = (payload or {}).get("result") or {}
    return {
        "kind":        "steady_state",
        "time":        None,
        "observables": dict(result.get("observables") or {}),
    }


class SimulatorRunnerStep(Step):
    """Run every SED-ML job of every biomodel under one configured simulator.

    Config:
        simulator_name: one of `ALL_SIMULATORS` (`copasi`, `tellurium`,
            `simbio`).

    Inputs:
        models: `map[biomodel_id, biomodel_jobs]` — each entry has
            `sbml_path` + `sedml_jobs` (the LoadBiomodelStep output).

    Outputs:
        results: `map[biomodel_id, map[sedml_doc_name, simulation_result]]`
            — this simulator's slice of the full results tree.
    """

    config_schema: ClassVar[Dict[str, Any]] = {
        "simulator_name": {"_type": "string", "_default": "copasi"},
    }

    def inputs(self) -> Dict[str, str]:
        return {"models": "map[biomodel_jobs]"}

    def outputs(self) -> Dict[str, str]:
        return {"results": "sim_results_per_biomodel"}

    def update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        name = self.config["simulator_name"]
        if name not in ALL_SIMULATORS:
            raise ValueError(
                f"SimulatorRunnerStep: unknown simulator {name!r}; "
                f"known: {ALL_SIMULATORS}"
            )

        utc_cls = _UTC_CLASS_FOR(name)
        ss_cls = _SS_CLASS_FOR(name)

        out: Dict[str, Dict[str, Any]] = {}
        for bid, model in (state.get("models") or {}).items():
            out[bid] = {}
            sbml_path = (model or {}).get("sbml_path") or ""
            for job in (model or {}).get("sedml_jobs") or []:
                job_name = job.get("name") or "sim"
                kind = job.get("kind")
                try:
                    if kind == "utc":
                        inner = utc_cls(core=getattr(self, "core", None))
                        payload = inner.update({
                            "model_source": sbml_path,
                            "time":         float(job.get("time") or 0.0),
                            "n_points":     int(job.get("n_points") or 2),
                        })
                        out[bid][job_name] = _utc_to_simulation_result(payload)
                    elif kind == "steady_state":
                        inner = ss_cls(core=getattr(self, "core", None))
                        payload = inner.update({"model_source": sbml_path})
                        out[bid][job_name] = _ss_to_simulation_result(payload)
                    else:
                        out[bid][job_name] = {
                            "kind":        str(kind),
                            "time":        None,
                            "observables": {},
                            "error":       f"unknown job kind {kind!r}",
                        }
                except Exception as exc:
                    out[bid][job_name] = {
                        "kind":        kind or "unknown",
                        "time":        None,
                        "observables": {},
                        "error":       f"{type(exc).__name__}: {exc}",
                    }
        return {"results": out}
```

- [ ] **Step 4: Re-export from `pbg_biomodels/steps/__init__.py`**

Add the import + `__all__` entry. After modification, the file's full contents:

```python
"""Process-bigraph Steps contributed by pbg-biomodels."""

from pbg_biomodels.steps.load_biomodel import LoadBiomodelStep
from pbg_biomodels.steps.simulator_comparison import SimulatorComparisonStep
from pbg_biomodels.steps.simulator_runner import SimulatorRunnerStep
from pbg_biomodels.steps.simulators import (
    BiomodelsCopasiStep,
    BiomodelsCopasiSteadyStateStep,
    BiomodelsSimbioStep,
    BiomodelsSimbioSteadyStateStep,
    BiomodelsTelluriumStep,
    BiomodelsTelluriumSteadyStateStep,
)

__all__ = [
    "LoadBiomodelStep",
    "SimulatorComparisonStep",
    "SimulatorRunnerStep",
    "BiomodelsCopasiStep",
    "BiomodelsCopasiSteadyStateStep",
    "BiomodelsTelluriumStep",
    "BiomodelsTelluriumSteadyStateStep",
    "BiomodelsSimbioStep",
    "BiomodelsSimbioSteadyStateStep",
]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_simulator_runner_step.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add pbg_biomodels/steps/simulator_runner.py pbg_biomodels/steps/__init__.py tests/test_simulator_runner_step.py
git commit -m "$(cat <<'EOF'
runner: SimulatorRunnerStep — one Step per simulator, iterates all jobs

Reads a `models` map (LoadBiomodelStep outputs keyed by biomodel id),
dispatches every SED-ML job to its simulator's UTC or SteadyState
adapter, and writes a nested `results[bid][sedml_doc] = simulation_result`
slice keyed by the simulator this runner is configured for. Per-job
try/except so one crashy job doesn't abort the runner.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task D1: `compare_n_engines_steady_state`

**Files:**
- Modify: `pbg_biomodels/comparison.py` (add the SS counterpart to `compare_n_engines`)
- Test: covered by Task D2's tests — but we keep one unit test here for the math.

- [ ] **Step 1: Write the failing test**

Append to a new file `tests/test_comparison_steady_state.py`:

```python
"""All-pairs steady-state comparison: |a-b| / max(|a|,|b|, eps) per observable."""
from pbg_biomodels.comparison import compare_n_engines_steady_state


def test_two_engines_identical_are_in_good_bucket():
    out = compare_n_engines_steady_state({
        "copasi":    {"A": 1.0, "B": 2.0},
        "tellurium": {"A": 1.0, "B": 2.0},
    })
    assert out["engines"] == ["copasi", "tellurium"]
    assert out["max_nrmse"] == 0.0
    assert out["bucket"] == "good"


def test_disjoint_observables_are_not_compared():
    out = compare_n_engines_steady_state({
        "copasi":    {"A": 1.0},
        "tellurium": {"B": 1.0},
    })
    # No shared observables → no nrmse value for that pair.
    pair = out["pairs"]["copasi__tellurium"]
    assert pair["n_shared"] == 0
    assert pair["mean_nrmse"] is None


def test_three_engines_picks_worst_pair():
    out = compare_n_engines_steady_state({
        "copasi":    {"A": 1.0},
        "tellurium": {"A": 1.1},   # 10% off copasi
        "simbio":    {"A": 2.0},   # 50% off copasi (worst pair)
    })
    assert out["max_nrmse"] > 0.4
    assert set(out["worst_pair"]) == {"copasi", "simbio"}
    assert out["bucket"] in {"borderline", "large"}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_comparison_steady_state.py -v
```

Expected: FAIL — `ImportError: cannot import name 'compare_n_engines_steady_state'`.

- [ ] **Step 3: Add `compare_n_engines_steady_state` to `pbg_biomodels/comparison.py`**

Append at the end of `pbg_biomodels/comparison.py`:

```python
def compare_two_engines_steady_state(
    engine_a: Optional[Dict[str, float]],
    engine_b: Optional[Dict[str, float]],
    name_a: str = "a",
    name_b: str = "b",
    eps: float = 1e-30,
) -> Dict[str, Any]:
    """Compare two `{observable_name: float}` maps observable-by-observable.

    Metric: `|a-b| / max(|a|, |b|, eps)` per shared observable; the result's
    `mean_nrmse` is the mean across shared observables. Buckets use the same
    `BUCKET_THRESHOLDS` as the UTC version so the vocabulary is shared.
    """
    a = engine_a or {}
    b = engine_b or {}
    shared = sorted(set(a) & set(b))
    nrmse_by_species: Dict[str, float] = {}
    for sp in shared:
        denom = max(abs(a[sp]), abs(b[sp]), eps)
        nrmse_by_species[sp] = abs(a[sp] - b[sp]) / denom

    if nrmse_by_species:
        mean_nrmse: Optional[float] = (
            sum(nrmse_by_species.values()) / len(nrmse_by_species)
        )
    else:
        mean_nrmse = None

    bucket_id, bucket_label = bucket_for(mean_nrmse)
    return {
        "n_shared":         len(shared),
        "rmse_by_species":  {},  # SS pairs have no time-series RMSE
        "nrmse_by_species": nrmse_by_species,
        "mean_nrmse":       mean_nrmse,
        "bucket":           bucket_id,
        "bucket_label":     bucket_label,
    }


def compare_n_engines_steady_state(
    engines: Dict[str, Optional[Dict[str, float]]],
) -> Dict[str, Any]:
    """All-pairs steady-state comparison across N named engines.

    Same return shape as :func:`compare_n_engines`, but each pair is
    scored with the steady-state metric. Engines with no observables
    (or `None`) are dropped from the comparison.
    """
    present = sorted(name for name, obs in engines.items() if obs)
    pairs: Dict[str, Any] = {}
    matrix: Dict[str, Dict[str, Optional[float]]] = {
        a: {b: None for b in present} for a in present
    }
    max_nrmse: Optional[float] = None
    worst_pair: Optional[List[str]] = None

    for i, a in enumerate(present):
        for b in present[i + 1:]:
            result = compare_two_engines_steady_state(
                engines[a], engines[b], name_a=a, name_b=b,
            )
            pairs[f"{a}__{b}"] = result
            mean = result.get("mean_nrmse")
            matrix[a][b] = mean
            matrix[b][a] = mean
            if mean is not None and (max_nrmse is None or mean > max_nrmse):
                max_nrmse = mean
                worst_pair = [a, b]

    bucket_id, bucket_label = bucket_for(max_nrmse)
    return {
        "engines":      present,
        "pairs":        pairs,
        "matrix":       matrix,
        "max_nrmse":    max_nrmse,
        "worst_pair":   worst_pair,
        "bucket":       bucket_id,
        "bucket_label": bucket_label,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_comparison_steady_state.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pbg_biomodels/comparison.py tests/test_comparison_steady_state.py
git commit -m "$(cat <<'EOF'
comparison: compare_n_engines_steady_state — scalar all-pairs nRMSE

Adds the steady-state counterpart to compare_n_engines. Per shared
observable: |a-b| / max(|a|,|b|, eps); mean across shared observables
maps to the same BUCKET_THRESHOLDS as the UTC version so the bucket
vocabulary is shared.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task D2: BatchCompareStep

**Files:**
- Modify: `pbg_biomodels/steps/simulator_comparison.py` (add `BatchCompareStep`)
- Test: `tests/test_batch_compare_step.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_batch_compare_step.py`:

```python
"""`BatchCompareStep` reads the full nested results store
(map[bid, map[sim, map[sedml_doc, simulation_result]]]) and produces
comparisons[bid][sedml_doc] using compare_n_engines (UTC) or
compare_n_engines_steady_state (SS).
"""
from typing import Any, Dict
import warnings

import pytest

from pbg_biomodels.steps.simulator_comparison import BatchCompareStep


def _utc(time, observables):
    return {"kind": "utc", "time": time, "observables": observables}


def _ss(observables):
    return {"kind": "steady_state", "time": None, "observables": observables}


def test_utc_pairs_use_nrmse_per_species():
    out = BatchCompareStep().update({
        "results": {
            "BIOMD0000000001": {
                "copasi":    {"sim1": _utc([0.0, 1.0],
                                            {"A": [1.0, 0.5], "B": [0.0, 0.5]})},
                "tellurium": {"sim1": _utc([0.0, 1.0],
                                            {"A": [1.0, 0.5], "B": [0.0, 0.5]})},
            }
        }
    })
    cmp = out["comparisons"]["BIOMD0000000001"]["sim1"]
    assert cmp["engines"] == ["copasi", "tellurium"]
    assert cmp["bucket"] == "good"


def test_ss_pairs_use_steady_state_metric():
    out = BatchCompareStep().update({
        "results": {
            "BIOMD0000000001": {
                "copasi":    {"sim_ss": _ss({"A": 1.0, "B": 2.0})},
                "tellurium": {"sim_ss": _ss({"A": 1.0, "B": 2.0})},
                "simbio":    {"sim_ss": _ss({"A": 1.05, "B": 2.0})},
            }
        }
    })
    cmp = out["comparisons"]["BIOMD0000000001"]["sim_ss"]
    assert set(cmp["engines"]) == {"copasi", "tellurium", "simbio"}
    # copasi vs tellurium is identical; copasi vs simbio has a 5% offset on A.
    assert cmp["bucket"] in {"good", "borderline"}


def test_cross_kind_under_one_sedml_doc_is_skipped_with_warning():
    """A SED-ML doc that ends up with both UTC and SS results across
    simulators is a pathological case — record a warning, emit empty."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = BatchCompareStep().update({
            "results": {
                "BIOMD0000000001": {
                    "copasi":    {"sim1": _utc([0.0], {"A": [1.0]})},
                    "tellurium": {"sim1": _ss({"A": 1.0})},
                }
            }
        })
    cmp = out["comparisons"]["BIOMD0000000001"]["sim1"]
    assert cmp["engines"] == []
    assert cmp["bucket"] == "none"
    assert any("mixed kinds" in str(w.message) for w in caught)


def test_simulator_with_no_result_for_sedml_doc_is_dropped():
    """A simulator that doesn't have an entry for a sedml doc is excluded
    from that doc's comparison."""
    out = BatchCompareStep().update({
        "results": {
            "BIOMD0000000001": {
                "copasi":    {"sim1": _utc([0.0], {"A": [1.0]})},
                "tellurium": {},  # simulator ran nothing for this biomodel
            }
        }
    })
    cmp = out["comparisons"]["BIOMD0000000001"]["sim1"]
    assert cmp["engines"] == ["copasi"]  # only the one engine with data
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_batch_compare_step.py -v
```

Expected: FAIL — `ImportError: cannot import name 'BatchCompareStep'`.

- [ ] **Step 3: Add `BatchCompareStep` to `pbg_biomodels/steps/simulator_comparison.py`**

Append at the end of the file:

```python
class BatchCompareStep(Step):
    """All-pairs nRMSE across simulators per (biomodel_id, sedml_doc).

    Inputs:
        results: `map[bid, map[sim, map[sedml_doc, simulation_result]]]`.

    Outputs:
        comparisons: `map[bid, map[sedml_doc, tree]]` — each leaf is the
            `compare_n_engines` (UTC) or `compare_n_engines_steady_state`
            (SS) return shape.

    A `(bid, sedml_doc)` whose simulators produced both UTC and SS results
    is treated as a SED-ML pathology: a warning is emitted and the leaf is
    the `"none"` bucket with empty engines.
    """

    config_schema: ClassVar[Dict[str, Any]] = {}

    def inputs(self) -> Dict[str, str]:
        return {"results": "tree"}

    def outputs(self) -> Dict[str, str]:
        return {"comparisons": "tree"}

    def update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        from pbg_biomodels.comparison import (
            bucket_for,
            compare_n_engines,
            compare_n_engines_steady_state,
        )
        import warnings

        results = state.get("results") or {}
        out: Dict[str, Dict[str, Any]] = {}

        # Index of (bid, sedml_doc) -> {sim: simulation_result}.
        for bid, per_sim in results.items():
            out[bid] = {}
            # Union the sedml-doc names across simulators that have data.
            sedml_docs: list = []
            seen = set()
            for sim_map in (per_sim or {}).values():
                for doc_name in (sim_map or {}).keys():
                    if doc_name not in seen:
                        seen.add(doc_name)
                        sedml_docs.append(doc_name)

            for doc_name in sedml_docs:
                # Gather per-simulator results for this doc.
                utc_engines: Dict[str, Dict[str, Any]] = {}
                ss_engines: Dict[str, Dict[str, float]] = {}
                for sim_name, sim_map in (per_sim or {}).items():
                    sim_result = (sim_map or {}).get(doc_name)
                    if not sim_result:
                        continue
                    if "error" in sim_result and not sim_result.get("observables"):
                        continue  # failed job — no engine slot
                    kind = sim_result.get("kind")
                    obs = sim_result.get("observables") or {}
                    if kind == "utc":
                        time = sim_result.get("time") or []
                        cols = list(obs.keys())
                        n_rows = min((len(obs[c]) for c in cols), default=0)
                        values = [[float(obs[c][r]) for c in cols] for r in range(n_rows)]
                        utc_engines[sim_name] = {
                            "time":    list(time),
                            "columns": cols,
                            "values":  values,
                        }
                    elif kind == "steady_state":
                        ss_engines[sim_name] = {k: float(v) for k, v in obs.items()}

                if utc_engines and ss_engines:
                    warnings.warn(
                        f"BatchCompareStep: {bid!r}/{doc_name!r} has mixed kinds "
                        f"across simulators; skipping",
                        stacklevel=2,
                    )
                    bucket_id, bucket_label = bucket_for(None)
                    out[bid][doc_name] = {
                        "engines":      [],
                        "pairs":        {},
                        "matrix":       {},
                        "max_nrmse":    None,
                        "worst_pair":   None,
                        "bucket":       bucket_id,
                        "bucket_label": bucket_label,
                    }
                elif utc_engines:
                    out[bid][doc_name] = compare_n_engines(utc_engines)
                elif ss_engines:
                    out[bid][doc_name] = compare_n_engines_steady_state(ss_engines)
                else:
                    bucket_id, bucket_label = bucket_for(None)
                    out[bid][doc_name] = {
                        "engines":      [],
                        "pairs":        {},
                        "matrix":       {},
                        "max_nrmse":    None,
                        "worst_pair":   None,
                        "bucket":       bucket_id,
                        "bucket_label": bucket_label,
                    }

        return {"comparisons": out}
```

Also add the missing `ClassVar` import at the top of the file:

```python
from typing import Any, ClassVar, Dict
```

(Replace the existing `from typing import Any, Dict` line.)

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_batch_compare_step.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pbg_biomodels/steps/simulator_comparison.py tests/test_batch_compare_step.py
git commit -m "$(cat <<'EOF'
comparison: BatchCompareStep — per (bid, sedml_doc) all-pairs across simulators

Reads the full nested results tree and groups by (biomodel_id, sedml_doc).
UTC-only docs route to compare_n_engines; SS-only docs route to
compare_n_engines_steady_state. Mixed-kind docs are treated as SED-ML
pathology (warning + 'none' bucket). Replaces the hardcoded two-engine
shape of SimulatorComparisonStep for the new batch composite; the old
two-engine class stays for legacy compare-biomodel.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task E1: BatchCompareOverlay visualization

**Files:**
- Create: `pbg_biomodels/visualizations/batch_compare_overlay.py`
- Test: `tests/test_batch_compare_overlay.py`

The viz takes the full nested `results` + `comparisons` and renders a card grid (one card per biomodel, like the legacy viz) where each card expands into a per-sedml-doc tab strip. Each tab is a per-observable overlay across simulators that have data; SS observables render as a bar chart of final values instead of a line.

- [ ] **Step 1: Write the failing test**

Create `tests/test_batch_compare_overlay.py`:

```python
"""`BatchCompareOverlay` is a Visualization that renders the full nested
results store into an HTML fragment. The smoke test here just checks the
fragment is non-empty and contains the biomodel id + simulator names —
the deep rendering is exercised in the end-to-end generator test.
"""
import pytest

from pbg_biomodels.visualizations.batch_compare_overlay import BatchCompareOverlay


def _utc(time, observables):
    return {"kind": "utc", "time": time, "observables": observables}


def test_overlay_renders_card_per_biomodel():
    viz = BatchCompareOverlay(config={
        "biomodel_ids": ["BIOMD0000000001", "BIOMD0000000002"],
        "title": "batch test",
    })
    out = viz.update({
        "results": {
            "BIOMD0000000001": {
                "copasi":    {"sim1": _utc([0.0, 1.0],
                                            {"A": [1.0, 0.5]})},
                "tellurium": {"sim1": _utc([0.0, 1.0],
                                            {"A": [1.0, 0.5]})},
            },
            "BIOMD0000000002": {
                "copasi":    {"sim1": _utc([0.0, 1.0],
                                            {"A": [2.0, 1.0]})},
            },
        },
        "comparisons": {
            "BIOMD0000000001": {"sim1": {
                "engines": ["copasi", "tellurium"], "pairs": {},
                "matrix": {}, "max_nrmse": 0.0,
                "worst_pair": None, "bucket": "good",
                "bucket_label": "Good (≤1%)",
            }},
            "BIOMD0000000002": {"sim1": {
                "engines": ["copasi"], "pairs": {},
                "matrix": {}, "max_nrmse": None,
                "worst_pair": None, "bucket": "none",
                "bucket_label": "No comparison",
            }},
        },
    })
    html = out["html"]
    assert isinstance(html, str) and html.strip()
    assert "BIOMD0000000001" in html
    assert "BIOMD0000000002" in html
    assert "copasi" in html
    assert "tellurium" in html
    assert "batch test" in html


def test_overlay_with_no_results_returns_placeholder():
    viz = BatchCompareOverlay(config={"biomodel_ids": []})
    out = viz.update({"results": {}, "comparisons": {}})
    assert "No biomodels" in out["html"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_batch_compare_overlay.py -v
```

Expected: FAIL — `ImportError: cannot import name 'BatchCompareOverlay'`.

- [ ] **Step 3: Create `pbg_biomodels/visualizations/batch_compare_overlay.py`**

```python
"""BatchCompareOverlay — N-simulator + multi-sedml-doc summary-card grid.

Each biomodel gets a card colored by its worst-pair bucket (aggregated
across sedml docs); clicking the card reveals a tab strip with one tab
per sedml doc, and each tab renders a per-observable overlay across the
simulators that produced output. Steady-state observables render as a
small grouped bar chart of final values instead of a line.

The HTML is a self-contained fragment that includes Plotly inline.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from plotly.offline import get_plotlyjs

from pbg_superpowers.visualization import Visualization


_BUCKET_COLOR = {
    "good":       "#1b6e3c",
    "borderline": "#b8741a",
    "large":      "#b3261e",
    "none":       "#5d6573",
}

_BUCKET_RANK = {"good": 0, "borderline": 1, "large": 2, "none": -1}


def _aggregate_card_bucket(per_doc_comparisons: Dict[str, Any]) -> Dict[str, Any]:
    """Pick the worst (largest-mean-nrmse) bucket across this biomodel's docs."""
    worst = None
    for cmp_dict in (per_doc_comparisons or {}).values():
        if not cmp_dict:
            continue
        if worst is None:
            worst = cmp_dict
            continue
        wr = _BUCKET_RANK.get(worst.get("bucket"), -1)
        cr = _BUCKET_RANK.get(cmp_dict.get("bucket"), -1)
        if cr > wr:
            worst = cmp_dict
    return worst or {"bucket": "none", "bucket_label": "No comparison",
                     "max_nrmse": None, "engines": []}


def _utc_overlay_figure(
    sim_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Per-observable small-multiples overlay across simulators."""
    species_order: List[str] = []
    seen: set = set()
    for sr in sim_results.values():
        for sp in (sr.get("observables") or {}).keys():
            if sp not in seen:
                seen.add(sp)
                species_order.append(sp)

    if not species_order:
        return {"data": [], "layout": {"title": "No observables"}}

    cols = 3
    rows = (len(species_order) + cols - 1) // cols
    traces: List[Dict[str, Any]] = []
    layout: Dict[str, Any] = {
        "grid":   {"rows": rows, "columns": cols, "pattern": "independent"},
        "height": 220 * rows + 100,
        "legend": {"orientation": "h", "y": 1.04, "x": 0},
        "margin": {"t": 60, "b": 40, "l": 60, "r": 20},
    }
    seen_legend: set = set()
    for i, sp in enumerate(species_order):
        idx = i + 1
        x_key = "xaxis" + ("" if idx == 1 else str(idx))
        y_key = "yaxis" + ("" if idx == 1 else str(idx))
        x_ref = "x" if idx == 1 else f"x{idx}"
        y_ref = "y" if idx == 1 else f"y{idx}"
        layout[y_key] = {"title": {"text": sp}}
        layout[x_key] = {"title": {"text": "time"}}
        for sim_name, sr in sim_results.items():
            obs = (sr.get("observables") or {})
            series = obs.get(sp)
            time = sr.get("time")
            if series is None or time is None:
                continue
            traces.append({
                "x": list(time), "y": list(series),
                "mode": "lines", "name": sim_name,
                "legendgroup": sim_name,
                "showlegend": sim_name not in seen_legend,
                "xaxis": x_ref, "yaxis": y_ref,
            })
            seen_legend.add(sim_name)
    return {"data": traces, "layout": layout}


def _ss_bar_figure(sim_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Grouped bar chart of final steady-state values across simulators."""
    species: List[str] = []
    seen: set = set()
    for sr in sim_results.values():
        for sp in (sr.get("observables") or {}).keys():
            if sp not in seen:
                seen.add(sp)
                species.append(sp)
    if not species:
        return {"data": [], "layout": {"title": "No observables"}}
    traces = [
        {
            "type": "bar",
            "name": sim_name,
            "x":    species,
            "y":    [float((sr.get("observables") or {}).get(sp, 0.0)) for sp in species],
        }
        for sim_name, sr in sim_results.items()
    ]
    return {"data": traces, "layout": {"barmode": "group", "height": 360}}


def _card_html(bid: str, agg: Dict[str, Any]) -> str:
    bucket = agg.get("bucket") or "none"
    label = agg.get("bucket_label") or "No comparison"
    max_n = agg.get("max_nrmse")
    n_eng = len(agg.get("engines") or [])
    max_str = f"{max_n:.4g}" if isinstance(max_n, (int, float)) else "—"
    color = _BUCKET_COLOR.get(bucket, "#5d6573")
    return (
        f'<div class="biomodel-card" data-biomodel="{bid}" '
        f'style="border-left:6px solid {color};padding:10px 14px;'
        f'background:#fafbfc;font-family:-apple-system,sans-serif;'
        f'cursor:pointer;border-radius:4px;'
        f'display:flex;align-items:center;gap:12px;">'
        f'<span class="chevron" style="font-size:11px;color:#888;width:10px;'
        f'transition:transform 0.15s;display:inline-block;">▶</span>'
        f'<div style="flex:1;"><div style="font-weight:600;font-size:13px;">{bid}</div>'
        f'<div style="font-size:12px;color:#444;margin-top:2px;">{label} · '
        f'worst nRMSE {max_str} · {n_eng} engines</div></div></div>'
    )


def _detail_html(bid: str, doc_figs: Dict[str, Dict[str, Any]]) -> str:
    """Render the tab strip + plot containers for one biomodel's sedml docs."""
    if not doc_figs:
        body = '<div style="color:#888;padding:8px;">No data for this biomodel.</div>'
    else:
        tabs = "".join(
            f'<button class="batch-tab" data-bid="{bid}" data-doc="{doc}" '
            f'style="margin-right:4px;padding:4px 10px;font-size:12px;">{doc}</button>'
            for doc in doc_figs.keys()
        )
        panes = "".join(
            f'<div class="batch-pane" id="pane-{bid}-{doc}" '
            f'style="display:none;"><div id="plot-{bid}-{doc}"></div></div>'
            for doc in doc_figs.keys()
        )
        body = (
            f'<div class="batch-tab-strip" style="margin-bottom:8px;">{tabs}</div>'
            f'{panes}'
        )
        fig_blob = {doc: fig for doc, fig in doc_figs.items()}
        body += (
            "<script>"
            "window.__batchFigures = window.__batchFigures || {};"
            f'window.__batchFigures[{json.dumps(bid)}] = {json.dumps(fig_blob)};'
            "</script>"
        )
    return (
        f'<div id="detail-{bid}" class="biomodel-detail" '
        f'style="display:none;padding:8px 14px 14px 28px;'
        f'background:#fcfcfd;border-left:1px solid #e5e7eb;'
        f'border-radius:0 0 4px 4px;margin-top:-2px;">{body}</div>'
    )


_TOGGLE_JS = """
window.__batchFigures = window.__batchFigures || {};
window.__batchFigsPlotted = window.__batchFigsPlotted || {};
function _ensurePlot(bid, doc) {
  var key = bid + "::" + doc;
  if (window.__batchFigsPlotted[key]) return;
  var fig = (window.__batchFigures[bid] || {})[doc];
  if (!fig || !window.Plotly) return;
  window.Plotly.newPlot("plot-" + bid + "-" + doc, fig.data, fig.layout,
                        {responsive: true, displaylogo: false});
  window.__batchFigsPlotted[key] = true;
}
document.querySelectorAll('.biomodel-card').forEach(function(card) {
  card.addEventListener('click', function() {
    var bid = card.getAttribute('data-biomodel');
    var pane = document.getElementById('detail-' + bid);
    if (!pane) return;
    var isOpen = pane.style.display !== 'none';
    pane.style.display = isOpen ? 'none' : 'block';
    var chev = card.querySelector('.chevron');
    if (chev) chev.style.transform = isOpen ? 'rotate(0deg)' : 'rotate(90deg)';
    if (!isOpen) {
      var firstTab = pane.querySelector('.batch-tab');
      if (firstTab) firstTab.click();
    }
  });
});
document.querySelectorAll('.batch-tab').forEach(function(tab) {
  tab.addEventListener('click', function() {
    var bid = tab.getAttribute('data-bid');
    var doc = tab.getAttribute('data-doc');
    var detail = document.getElementById('detail-' + bid);
    if (!detail) return;
    detail.querySelectorAll('.batch-pane').forEach(function(p) {
      p.style.display = 'none';
    });
    var pane = document.getElementById('pane-' + bid + '-' + doc);
    if (pane) pane.style.display = 'block';
    _ensurePlot(bid, doc);
  });
});
"""


class BatchCompareOverlay(Visualization):
    """N-simulator + multi-sedml-doc card-grid overlay.

    Inputs:
        results: `map[bid, map[sim, map[sedml_doc, simulation_result]]]`.
        comparisons: `map[bid, map[sedml_doc, tree]]`.

    Output: a single `html` fragment.
    """

    config_schema = {
        "title":        {"_type": "string", "_default": ""},
        "biomodel_ids": {"_type": "list[string]", "_default": []},
    }

    def inputs(self) -> Dict[str, Any]:
        return {"results": "tree", "comparisons": "tree"}

    def update(self, state: Dict[str, Any]) -> Dict[str, str]:
        results = state.get("results") or {}
        comparisons = state.get("comparisons") or {}
        ids = list((self.config or {}).get("biomodel_ids") or []) or list(results.keys())

        if not ids or not results:
            return {"html":
                '<div style="padding:20px;color:#888;'
                'font-family:-apple-system,sans-serif;">'
                'No biomodels to compare.</div>'}

        rows: List[str] = []
        for bid in ids:
            per_sim = results.get(bid) or {}

            # Index sedml docs in this biomodel's results (union across sims).
            sedml_docs: List[str] = []
            seen: set = set()
            for sim_map in per_sim.values():
                for doc in (sim_map or {}).keys():
                    if doc not in seen:
                        seen.add(doc)
                        sedml_docs.append(doc)

            doc_figs: Dict[str, Dict[str, Any]] = {}
            for doc in sedml_docs:
                sim_results = {
                    sim_name: (sim_map or {}).get(doc) or {}
                    for sim_name, sim_map in per_sim.items()
                    if (sim_map or {}).get(doc)
                }
                if not sim_results:
                    continue
                first_kind = next(iter(sim_results.values())).get("kind")
                if first_kind == "steady_state":
                    doc_figs[doc] = _ss_bar_figure(sim_results)
                else:
                    doc_figs[doc] = _utc_overlay_figure(sim_results)

            agg = _aggregate_card_bucket(comparisons.get(bid) or {})
            rows.append(_card_html(bid, agg))
            rows.append(_detail_html(bid, doc_figs))

        title = (self.config or {}).get("title", "")
        title_html = (
            f'<h3 style="margin:0 0 12px 0;font-family:-apple-system,sans-serif;">'
            f'{title}</h3>'
        ) if title else ''
        return {"html": (
            f'<div>{title_html}'
            f'<div class="biomodel-list" style="'
            f'display:flex;flex-direction:column;gap:4px;">'
            + "".join(rows) +
            '</div>'
            f'<script>{get_plotlyjs()}</script>'
            f'<script>{_TOGGLE_JS}</script>'
            '</div>'
        )}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_batch_compare_overlay.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pbg_biomodels/visualizations/batch_compare_overlay.py tests/test_batch_compare_overlay.py
git commit -m "$(cat <<'EOF'
viz: BatchCompareOverlay — multi-sim + multi-sedml-doc card grid

Each biomodel card aggregates the worst bucket across its sedml docs;
clicking the card reveals a tab strip with one tab per sedml doc, each
plotting a per-observable overlay across simulators. SS docs render as a
grouped bar chart of final values instead of a line. Self-contained
Plotly fragment; click-to-expand is lazy-rendered.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task E2: batch-compare-biomodels composite generator

**Files:**
- Create: `pbg_biomodels/composites/batch_compare_biomodels.py`
- Test: `tests/test_batch_compare_biomodels_generator.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_batch_compare_biomodels_generator.py`:

```python
"""The `batch-compare-biomodels` generator builds a composite with one
LoadBiomodelStep per biomodel id, one SimulatorRunnerStep per requested
simulator (not per simulator × biomodel), one BatchCompareStep, and one
BatchCompareOverlay viz step.
"""
import pytest

import pbg_biomodels.composites.batch_compare_biomodels  # noqa: F401
from pbg_superpowers.composite_generator import _REGISTRY, build_generator


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
        assert state[key]["outputs"]["results"] == ["results", sim]

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


def test_legacy_compare_biomodel_generator_still_registered():
    """The new generator does not displace the legacy one."""
    legacy = [e for e in _REGISTRY.values() if e.name == "compare-biomodel"]
    assert len(legacy) == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_batch_compare_biomodels_generator.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'pbg_biomodels.composites.batch_compare_biomodels'`.

- [ ] **Step 3: Create `pbg_biomodels/composites/batch_compare_biomodels.py`**

```python
"""Composite generator: batch-compare-biomodels.

For each biomodel id, fetch the SBML + parse the SED-ML once into a
`biomodel_jobs` record under `models[bid]`. For each requested simulator,
one `SimulatorRunnerStep` iterates `models` and writes its slice of
`results[sim]`. A single `BatchCompareStep` consumes the full nested
`results` and writes `comparisons`. A single `BatchCompareOverlay` viz
renders the HTML.

Distinct from the legacy `compare-biomodel`:
* one Step per simulator, not per (simulator × biomodel);
* nested `results[bid][sim][sedml_doc]` instead of flat suffixed keys;
* supports steady-state tasks (kind="steady_state");
* supports multiple SED-ML simulations per biomodel.

The legacy `compare-biomodel` generator is preserved untouched.
"""
from __future__ import annotations

from typing import Any, Dict, List

from pbg_superpowers.composite_generator import composite_generator

from pbg_biomodels.simulators import resolve_simulators


LOAD_STEP_ADDRESS    = "local:pbg_biomodels.steps.load_biomodel.LoadBiomodelStep"
RUNNER_STEP_ADDRESS  = "local:pbg_biomodels.steps.simulator_runner.SimulatorRunnerStep"
COMPARE_STEP_ADDRESS = "local:pbg_biomodels.steps.simulator_comparison.BatchCompareStep"
VIZ_STEP_ADDRESS     = "local:pbg_biomodels.visualizations.batch_compare_overlay.BatchCompareOverlay"


@composite_generator(
    name="batch-compare-biomodels",
    description=(
        "For each BioModel id, fetch the SBML, parse every SED-ML simulation "
        "(UTC + steady-state), then run each simulator over all jobs via a "
        "single per-simulator runner Step. Comparison reads the full nested "
        "results store and produces all-pairs nRMSE per (biomodel, sedml-doc)."
    ),
    parameters={
        "biomodel_ids": {
            "type": "list[string]",
            "default": ["BIOMD0000000001"],
            "description": (
                "BioModels identifiers, one per line. Each id gets one "
                "LoadBiomodelStep that writes into `models[<id>]`."
            ),
        },
        "simulators": {
            "type": "list[string]",
            "default": ["copasi", "tellurium", "simbio"],
            "description": (
                "Simulators to run, one per line. Allowed values: "
                "copasi, tellurium, simbio. One SimulatorRunnerStep "
                "is created per entry; the runner iterates `models` "
                "and dispatches each job to its UTC or steady-state "
                "adapter."
            ),
        },
    },
    default_n_steps=1,
)
def build_batch_compare_biomodels(
    core: Any = None,
    *,
    biomodel_ids: List[str],
    simulators: List[str],
    with_emitter: bool = True,
    emitter_address: str = "local:RAMEmitter",
) -> Dict[str, Any]:
    sims = resolve_simulators(simulators)

    state: Dict[str, Any] = {
        "models":      {bid: {"sbml_path": "", "sedml_jobs": []}
                        for bid in biomodel_ids},
        "results":     {bid: {sim: {} for sim in sims}
                        for bid in biomodel_ids},
        "comparisons": {bid: {} for bid in biomodel_ids},
        "viz_html":    "",
    }

    for bid in biomodel_ids:
        state[f"biomodel_id_{bid}"] = bid
        state[f"load_{bid}"] = {
            "_type":   "step",
            "address": LOAD_STEP_ADDRESS,
            "config":  {},
            "inputs":  {"biomodel_id": [f"biomodel_id_{bid}"]},
            "outputs": {
                "sbml_path":  ["models", bid, "sbml_path"],
                "sedml_jobs": ["models", bid, "sedml_jobs"],
            },
        }

    for sim in sims:
        state[f"runner_{sim}"] = {
            "_type":   "step",
            "address": RUNNER_STEP_ADDRESS,
            "config":  {"simulator_name": sim},
            "inputs":  {"models": ["models"]},
            "outputs": {"results": ["results", sim]},
        }

    state["compare"] = {
        "_type":   "step",
        "address": COMPARE_STEP_ADDRESS,
        "config":  {},
        "inputs":  {"results": ["results"]},
        "outputs": {"comparisons": ["comparisons"]},
    }

    state["batch_overlay_viz"] = {
        "_type":   "step",
        "address": VIZ_STEP_ADDRESS,
        "config":  {
            "title":        "BioModels: batch comparison",
            "biomodel_ids": list(biomodel_ids),
        },
        "inputs":  {"results": ["results"], "comparisons": ["comparisons"]},
        "outputs": {"html": ["viz_html"]},
    }

    if with_emitter:
        state["emitter"] = {
            "_type":   "step",
            "address": emitter_address,
            "config":  {"emit": {
                "models":      "node",
                "results":     "node",
                "comparisons": "node",
            }},
            "inputs":  {
                "models":      ["models"],
                "results":     ["results"],
                "comparisons": ["comparisons"],
            },
        }

    return {"state": state, "run_steps_on_init": True}
```

The runner addresses are computed per-simulator from the registry but all use the same `SimulatorRunnerStep` class — the simulator-specific behavior lives in the runner's `config.simulator_name`. The `resolve_simulators` call validates the simulator names against the registry up front, so an unknown name raises `ValueError` before the composite is built.

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_batch_compare_biomodels_generator.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Run the full test suite to confirm nothing regresses**

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: every test PASSes (network-gated test in `test_load_biomodel_step_sedml_jobs.py` is skipped).

- [ ] **Step 6: Commit**

```bash
git add pbg_biomodels/composites/batch_compare_biomodels.py tests/test_batch_compare_biomodels_generator.py
git commit -m "$(cat <<'EOF'
composites: batch-compare-biomodels generator (alongside legacy)

One LoadBiomodelStep per biomodel writing into models[<bid>], one
SimulatorRunnerStep per requested simulator (NOT per simulator × biomodel),
one BatchCompareStep reading the full nested results store, one
BatchCompareOverlay viz. For N biomodels × M simulators the composite has
N + M + 2 step instances instead of the legacy N + N*M + N + 1. The
legacy `compare-biomodel` generator stays registered and untouched.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task E3: End-to-end smoke test

**Files:**
- Test: `tests/test_batch_compare_biomodels_end_to_end.py`

A small integration test that constructs the composite, replaces the per-simulator adapter classes with stubs (so we don't depend on the upstream SS PRs landing or on biomodels network access), runs it, and reads back results + comparisons + viz_html through the RAM emitter.

- [ ] **Step 1: Write the failing test**

Create `tests/test_batch_compare_biomodels_end_to_end.py`:

```python
"""End-to-end: build the batch composite with stubbed adapters + a stubbed
LoadBiomodelStep, run it for 0 seconds (steps fire on init), and assert
that results + comparisons + viz_html populate as expected.
"""
import os

import pytest


_FAKE_UTC = {
    "result": {
        "time":    [0.0, 0.5, 1.0],
        "columns": ["A", "B"],
        "values":  [[1.0, 0.0], [0.6, 0.4], [0.3, 0.7]],
    }
}


_FAKE_SS = {
    "result": {
        "kind":        "steady_state",
        "time":        None,
        "observables": {"A": 0.5, "B": 0.5},
    }
}


class _StubLoad:
    """Replaces LoadBiomodelStep — no network, returns canned jobs."""
    def __init__(self, config=None, core=None):
        pass
    def inputs(self): return {"biomodel_id": "string"}
    def outputs(self):
        return {
            "sbml_path":  "string",
            "sedml_jobs": "list[tree]",
            "time":       "float",
            "n_points":   "integer",
        }
    def update(self, state):
        return {
            "sbml_path":  f"/tmp/{state['biomodel_id']}.xml",
            "sedml_jobs": [
                {"name": "sim1", "kind": "utc", "time": 1.0, "n_points": 3},
                {"name": "ss",   "kind": "steady_state",
                 "time": None, "n_points": None},
            ],
            "time":       1.0,
            "n_points":   3,
        }


class _StubUTC:
    def __init__(self, config=None, core=None):
        pass
    def update(self, _state):
        return _FAKE_UTC


class _StubSS:
    def __init__(self, config=None, core=None):
        pass
    def update(self, _state):
        return _FAKE_SS


@pytest.fixture
def stubbed(monkeypatch):
    """Patch LoadBiomodelStep and the runner's adapter lookups."""
    import pbg_biomodels.steps.load_biomodel as load_mod
    monkeypatch.setattr(load_mod, "LoadBiomodelStep", _StubLoad)

    import pbg_biomodels.steps.simulator_runner as runner_mod
    monkeypatch.setattr(runner_mod, "_UTC_CLASS_FOR", lambda name: _StubUTC)
    monkeypatch.setattr(runner_mod, "_SS_CLASS_FOR",  lambda name: _StubSS)

    # Re-bind the local-address dispatch the runner relies on: when a
    # Composite resolves "local:…LoadBiomodelStep" it imports the module and
    # reads the class by name. The patch above replaces the attribute at
    # that import path, so the next address resolution picks up _StubLoad.
    return monkeypatch


def test_end_to_end_populates_results_and_comparisons(stubbed):
    from process_bigraph import Composite, gather_emitter_results

    from pbg_biomodels import register_types
    from pbg_biomodels.core import build_core
    from pbg_superpowers.composite_generator import _REGISTRY, build_generator
    import pbg_biomodels.composites.batch_compare_biomodels  # noqa: F401

    entry = next(e for e in _REGISTRY.values()
                 if e.name == "batch-compare-biomodels")
    doc = build_generator(entry, overrides={
        "biomodel_ids": ["BIOMD0000000001"],
        "simulators":   ["copasi", "tellurium"],
    })
    composite = Composite(doc, core=register_types(build_core()))
    composite.run(0.0)

    snap = (gather_emitter_results(composite).get(("emitter",)) or [{}])[-1]
    results = snap["results"]
    comparisons = snap["comparisons"]

    # Both simulators produced a `sim1` UTC result and a `ss` SS result.
    for sim in ("copasi", "tellurium"):
        bid_result = results["BIOMD0000000001"][sim]
        assert "sim1" in bid_result
        assert bid_result["sim1"]["kind"] == "utc"
        assert "ss" in bid_result
        assert bid_result["ss"]["kind"] == "steady_state"

    # One comparison per sedml doc.
    bid_cmp = comparisons["BIOMD0000000001"]
    assert "sim1" in bid_cmp and "ss" in bid_cmp
    assert set(bid_cmp["sim1"]["engines"]) == {"copasi", "tellurium"}
    assert bid_cmp["sim1"]["bucket"] == "good"
    assert set(bid_cmp["ss"]["engines"]) == {"copasi", "tellurium"}
    assert bid_cmp["ss"]["bucket"] == "good"

    # Viz produced non-empty HTML.
    assert "viz_html" in snap and "BIOMD0000000001" in snap["viz_html"]
```

- [ ] **Step 2: Run the end-to-end test**

```bash
.venv/bin/python -m pytest tests/test_batch_compare_biomodels_end_to_end.py -v
```

Expected: 1 test PASS. If it fails on `local:` address resolution for the stubbed `LoadBiomodelStep`, that means the composite's address resolver re-imports the class by name from the module path. The fix is to ensure the test patches *that* module attribute (it does — `pbg_biomodels.steps.load_biomodel.LoadBiomodelStep`), but **also** to make sure the address string in the composite generator matches: `local:pbg_biomodels.steps.load_biomodel.LoadBiomodelStep`. This is already the case.

- [ ] **Step 3: Run the full suite one last time**

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: all tests PASS, network-gated test skipped.

- [ ] **Step 4: Commit**

```bash
git add tests/test_batch_compare_biomodels_end_to_end.py
git commit -m "$(cat <<'EOF'
batch-compare-biomodels: end-to-end smoke test via RAM emitter

Builds the composite with stubbed LoadBiomodelStep + stubbed runner
adapter classes (no network, no real upstream simulators), runs it
through process_bigraph.Composite, and asserts results + comparisons +
viz_html populate as expected for both UTC and steady-state branches.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review

**Spec coverage:**

| Spec section | Task(s) |
|---|---|
| Types: `simulation_result`, `biomodel_jobs`, `sim_results_per_biomodel` | A1 |
| LoadBiomodelStep: emit all SED-ML tasks (UTC + SS) | A2, A3 |
| Steady-state Steps in sibling repos | Out of plan (separate `/pbg-expert` PRs); adapters in B2 lazy-import them |
| SimulatorSpec registry: steady-state field | B1 |
| SteadyState adapter Steps | B2 |
| SimulatorRunnerStep (one per simulator, iterates jobs) | C1 |
| compare_n_engines_steady_state | D1 |
| BatchCompareStep (full nested store; per-(bid, sedml) all-pairs) | D2 |
| BatchCompareOverlay viz | E1 |
| batch-compare-biomodels generator | E2 |
| Legacy compare-biomodel preserved | A3 (back-compat outputs), E2 (legacy still registered test) |
| End-to-end test (1 biomodel × 2 simulators) | E3 |

**Placeholder scan:** no TBDs, no "TODO", no "fill in details", no "similar to Task N". Every code step has actual code; every test step has actual test code.

**Type consistency:**
- `simulation_result` shape `{kind, time, observables}` used identically in A1, B2, C1, D2, E1, E3.
- `SimulatorRunnerStep.config.simulator_name` used in C1 (definition), C1 tests, E2 (generator wires it), E3 (end-to-end).
- Address strings: every reference to `BiomodelsCopasiSteadyStateStep`, `SimulatorRunnerStep`, `BatchCompareStep`, `BatchCompareOverlay` uses the same dotted path across the generator (`E2`), the adapter registry (`B1`), and the lazy-import lookup in the runner (`C1`).
- `models[bid]` shape: `{sbml_path: str, sedml_jobs: list[tree]}` matches the `biomodel_jobs` type from A1, the `LoadBiomodelStep` outputs from A3, the runner's input from C1, and the generator's pre-allocated state from E2.

No drift found.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-28-batch-compare-biomodels.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
