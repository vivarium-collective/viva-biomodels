"""Read BioSimulators SED-ML reference results into the composite leaf shape.

The BioSimulators test-suite stores one ``reports.h5`` per
``(biomodel, simulator, version)`` under a dataset directory laid out as::

    <root>/<BIOMD…>/<engine>/<version>/.../outputs/reports.h5

Inside each HDF5, SED-ML reports live under a ``<doc>.sedml`` group as 2-D
datasets shaped ``[n_dataset, n_timepoint]``. The row labels are carried in the
``sedmlDataSetLabels`` attribute (e.g. ``['Time', 'A', 'B', ...]``).

:func:`read_reference_leaf` converts the UTC report into the same flat
``map[observable -> timeseries]`` leaf the live runners emit (the ``Time`` row
becomes the reserved ``time`` key — see :mod:`viva_biomodels.result_leaf`).
"""
from __future__ import annotations

import warnings
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from viva_biomodels.result_leaf import SCAN_KEY, TIME_KEY

# Case-insensitive label that marks the sample-time row inside a SED-ML report.
_TIME_LABEL = "time"

# Member path of the report inside a BioSimulators ``results.zip``.
_ZIP_REPORT_MEMBER = "outputs/reports.h5"


def _decode(value: Any) -> str:
    """HDF5 attrs come back as ``str`` or ``bytes`` depending on the writer."""
    return value.decode() if isinstance(value, bytes) else str(value)


def read_reference_leaf(h5_path: str | Path) -> Dict[str, List[float]]:
    """Read the UTC SED-ML report from a ``reports.h5`` into a results leaf.

    Accepts either an extracted ``reports.h5`` or a BioSimulators
    ``results.zip`` (the as-shipped layout) — for a zip, the ``outputs/reports.h5``
    member is read in memory. Returns ``{time: [...], <observable>: [...], ...}``
    with the ``Time`` row remapped to the reserved ``time`` key; picks the first
    ``SedReport`` dataset that carries a time row.
    """
    import h5py

    try:
        if str(h5_path).lower().endswith(".zip"):
            import io

            with zipfile.ZipFile(h5_path) as z:
                raw = z.read(_ZIP_REPORT_MEMBER)
            with h5py.File(io.BytesIO(raw), "r") as f:
                return _leaf_from_h5(f)

        with h5py.File(h5_path, "r") as f:
            return _leaf_from_h5(f)
    except (OSError, KeyError, ValueError, RuntimeError, zipfile.BadZipFile) as e:
        # An unreadable reference archive means "no reference for this model",
        # not a failed run — the live engines still produce their timecourses.
        warnings.warn(f"read_reference_leaf: {h5_path} unreadable ({type(e).__name__}: {e}); "
                      "treating as no reference", stacklevel=2)
        return {}


def _leaf_from_h5(f) -> Dict[str, List[float]]:
    report = _find_utc_report(f)
    if report is None:
        return {}
    data = report[()]
    labels = [_decode(v) for v in report.attrs["sedmlDataSetLabels"]]
    leaf: Dict[str, List[float]] = {}
    for i, label in enumerate(labels):
        key = TIME_KEY if label.lower() == _TIME_LABEL else label
        leaf[key] = data[i].tolist()
    return leaf


def discover_reference_engines(root: str | Path, bid: str) -> List[str]:
    """Sorted engine names under ``<root>/<bid>/`` that have a ``reports.h5``."""
    model_dir = Path(root) / bid
    if not model_dir.is_dir():
        return []
    engines = [
        d.name
        for d in model_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".") and resolve_engine_h5(root, bid, d.name)
    ]
    return sorted(engines)


def resolve_engine_h5(root: str | Path, bid: str, engine: str) -> Optional[Path]:
    """Locate one engine's ``reports.h5``, preferring the latest version dir.

    Prefers an extracted ``reports.h5`` (both on-disk variants ``<ver>/outputs/``
    and ``<ver>/results/outputs/``); falls back to the as-shipped
    ``<ver>/results.zip`` (which holds ``outputs/reports.h5``). Returns ``None``
    when the engine, file, and zip are all absent.
    """
    engine_dir = Path(root) / bid / engine
    if not engine_dir.is_dir():
        return None
    # Version dirs sorted descending so the newest wins.
    versions = sorted(
        (d for d in engine_dir.iterdir() if d.is_dir() and not d.name.startswith(".")),
        reverse=True,
    )
    for vdir in versions:
        hits = sorted(vdir.rglob("reports.h5"))
        if hits:
            return hits[0]
        zips = sorted(vdir.glob("results.zip"))
        if zips:
            return zips[0]
    return None


def _find_utc_report(group) -> Optional[Any]:
    """Depth-first search for the first SedReport dataset with a time row.

    A corrupt / unreadable HDF5 object is skipped rather than aborting the
    search. Some BioSimulators reference archives raise
    ``KeyError: ... bad flag combination for message`` on open under newer HDF5
    builds; a model must not be dropped just because one object in its reference
    file is unreadable — the rest of the tree may still hold the UTC report.
    """
    import h5py

    for key in group:
        try:
            item = group[key]
            if isinstance(item, h5py.Group):
                found = _find_utc_report(item)
                if found is not None:
                    return found
                continue
            if _decode(item.attrs.get("_type", "")) != "SedReport":
                continue
            labels = [_decode(v).lower() for v in item.attrs.get("sedmlDataSetLabels", [])]
            if _TIME_LABEL in labels:
                return item
        except (KeyError, OSError, ValueError, RuntimeError):
            continue
    return None


def _find_scan_report(group) -> Optional[Any]:
    """First SedReport dataset of rank >= 3 (a repeatedTask / scan report).

    BioSimulators writes a repeatedTask report as
    ``[n_dataset, *scan_dims, n_timepoint]`` — rank >= 3 (a plain UTC report is
    rank 2). Depth-first, mirroring :func:`_find_utc_report`.
    """
    import h5py

    for key in group:
        item = group[key]
        if isinstance(item, h5py.Group):
            found = _find_scan_report(item)
            if found is not None:
                return found
            continue
        if _decode(item.attrs.get("_type", "")) != "SedReport":
            continue
        if getattr(item, "ndim", 0) >= 3 and "sedmlDataSetLabels" in item.attrs:
            return item
    return None


def read_reference_scan_leaf(h5_path: str | Path,
                             scan_values: Optional[List[float]] = None
                             ) -> Dict[str, List[float]]:
    """Reduce a BioSimulators repeatedTask report to a response-curve leaf.

    The report is shaped ``[n_dataset, *scan_dims, n_timepoint]``. Each
    observable is reduced to its **endpoint** (last timepoint) at every scan
    point — mirroring the live-engine reduction — and the scan dims are
    flattened to a single ordered axis. Returns a scan leaf
    ``{"scan": [...], "<observable>": [...]}`` where the ``scan`` axis is
    ``scan_values`` (aligned 1:1 by index; both sides come from the same SED-ML
    range) when supplied, else ``[0, 1, ..., n-1]``. Returns ``{}`` when no
    scan report is present or the shape is unusable.
    """
    import h5py

    if str(h5_path).lower().endswith(".zip"):
        import io
        import zipfile

        with zipfile.ZipFile(h5_path) as z:
            raw = z.read(_ZIP_REPORT_MEMBER)
        with h5py.File(io.BytesIO(raw), "r") as f:
            return _scan_leaf_from_h5(f, scan_values)

    with h5py.File(h5_path, "r") as f:
        return _scan_leaf_from_h5(f, scan_values)


def _scan_leaf_from_h5(f, scan_values: Optional[List[float]]) -> Dict[str, List[float]]:
    report = _find_scan_report(f)
    if report is None:
        return {}
    data = report[()]
    if getattr(data, "ndim", 0) < 3:
        return {}
    labels = [_decode(v) for v in report.attrs["sedmlDataSetLabels"]]
    # Drop the time axis (last) by taking the endpoint, then flatten scan dims:
    # [n_label, *scan_dims, n_time] -> [n_label, n_scan].
    endpoint = data[..., -1]
    n_label = endpoint.shape[0]
    flat = endpoint.reshape(n_label, -1)
    n_scan = flat.shape[1]

    axis = [float(x) for x in (list(scan_values or [])[:n_scan])]
    if len(axis) < n_scan:  # pad with indices when scan_values is short/absent
        axis += [float(i) for i in range(len(axis), n_scan)]

    leaf: Dict[str, List[float]] = {SCAN_KEY: axis}
    for i, label in enumerate(labels):
        if i >= n_label or label.lower() == _TIME_LABEL:
            continue
        leaf[label] = [float(x) for x in flat[i].tolist()]
    return leaf
