"""Unit tests for `viva_biomodels.reference_results` — reading BioSimulators
SED-ML reference results (`reports.h5`) into the composite's `results` leaf
shape and resolving the on-disk `<root>/<bid>/<engine>/<version>/.../reports.h5`
layout.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest


def _write_reports_h5(
    path: Path,
    *,
    labels=("Time", "A", "B"),
    n_time=5,
    report_id="autogen_report_for_task1",
    doc="BIOMD0000000001_url.sedml",
    report_type="SedReport",
):
    """Write a minimal BioSimulators-style reports.h5 (one report dataset)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # rows = one per dataset label; columns = time points. Row i is a ramp i..i+n.
    data = np.array(
        [[float(r) + float(c) for c in range(n_time)] for r in range(len(labels))],
        dtype="float64",
    )
    with h5py.File(path, "w") as f:
        grp = f.create_group(doc)
        ds = grp.create_dataset(report_id, data=data)
        ds.attrs["_type"] = report_type
        ds.attrs["sedmlDataSetLabels"] = list(labels)
        ds.attrs["sedmlId"] = report_id
    return data


def test_read_reference_leaf_maps_time_and_observables(tmp_path):
    from viva_biomodels.reference_results import read_reference_leaf

    data = _write_reports_h5(tmp_path / "reports.h5", labels=("Time", "A", "B"), n_time=5)

    leaf = read_reference_leaf(tmp_path / "reports.h5")

    # 'Time' row → reserved 'time' key; other labels are observables.
    assert "time" in leaf
    assert leaf["time"] == data[0].tolist()
    assert leaf["A"] == data[1].tolist()
    assert leaf["B"] == data[2].tolist()
    # No leftover 'Time' label.
    assert "Time" not in leaf


def _zip_reports_h5(zip_path: Path, **kw):
    """Write a results.zip holding `outputs/reports.h5` (the as-shipped layout)."""
    import io
    import zipfile

    h5_buf = Path(str(zip_path) + ".tmp.h5")
    data = _write_reports_h5(h5_buf, **kw)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(h5_buf, arcname="outputs/reports.h5")
    h5_buf.unlink()
    return data


def test_reads_reports_h5_from_results_zip(tmp_path):
    """Most models ship reports.h5 zipped inside `<ver>/results.zip`; the loader
    must read it without an extracted copy on disk."""
    from viva_biomodels.reference_results import (
        discover_reference_engines,
        read_reference_leaf,
        resolve_engine_h5,
    )

    root = tmp_path / "dataset"
    bid = "BIOMD0000000002"
    data = _zip_reports_h5(
        root / bid / "copasi" / "4.45.296" / "results.zip",
        labels=("Time", "A", "B"),
        n_time=4,
    )

    # discovery + resolution see the zipped engine.
    assert discover_reference_engines(root, bid) == ["copasi"]
    h5 = resolve_engine_h5(root, bid, "copasi")
    assert h5 is not None

    # read_reference_leaf transparently reads the zip member.
    leaf = read_reference_leaf(h5)
    assert leaf["time"] == data[0].tolist()
    assert leaf["A"] == data[1].tolist()


def test_discover_and_resolve_engine_h5_with_version_dirs(tmp_path):
    from viva_biomodels.reference_results import (
        discover_reference_engines,
        resolve_engine_h5,
    )

    root = tmp_path / "dataset"
    bid = "BIOMD0000000001"
    # copasi/amici use a nested `results/outputs/`; tellurium uses `outputs/`.
    _write_reports_h5(root / bid / "copasi" / "4.45.296" / "results" / "outputs" / "reports.h5")
    _write_reports_h5(root / bid / "copasi" / "4.46.300" / "results" / "outputs" / "reports.h5")
    _write_reports_h5(root / bid / "tellurium" / "2.2.10" / "outputs" / "reports.h5")

    # Engines present (sorted), skipping any without a reports.h5.
    assert discover_reference_engines(root, bid) == ["copasi", "tellurium"]

    # Latest version dir wins for an engine with several.
    copasi = resolve_engine_h5(root, bid, "copasi")
    assert copasi is not None and "4.46.300" in str(copasi)

    # The `outputs/` (non-nested) variant resolves too.
    assert resolve_engine_h5(root, bid, "tellurium") is not None

    # Missing engine → None (not an error).
    assert resolve_engine_h5(root, bid, "amici") is None
    # Missing model → no engines.
    assert discover_reference_engines(root, "BIOMD9999999999") == []


def test_read_reference_leaf_tolerates_unreadable_file(tmp_path):
    """An unreadable reference archive is treated as 'no reference' (empty leaf),
    not a crash — a model must not be lost over a corrupt reference file
    (environmental h5py 'bad flag combination' errors)."""
    from viva_biomodels.reference_results import read_reference_leaf

    bogus = tmp_path / "reports.h5"
    bogus.write_bytes(b"not an hdf5 file at all")
    with pytest.warns(UserWarning, match="unreadable"):
        assert read_reference_leaf(bogus) == {}
