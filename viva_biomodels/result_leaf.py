"""Shared helpers for the ``results`` leaf — a flat ``map[observable -> timeseries]``.

The batch-compare composite stores one leaf per
``(biomodel_id, sedml_job_id, simulator)``. Each leaf is a plain mapping from
observable name to its timeseries (``list[float]``):

* **UTC** jobs carry the sample times under the reserved key ``"time"``; every
  other key is an observable's trajectory (same length as ``time``).
* **repeated-task** (parameter-scan) jobs carry the swept parameter values under
  the reserved key ``"scan"``; every other key is an observable's *response
  curve* — its (reduced) value at each scan point, same length as ``"scan"``.
* **steady-state** jobs omit both axes and store each observable as a length-1
  list — the single steady-state value.

Classification is therefore purely structural: a leaf is UTC iff it has a
``"time"`` key, a parameter scan iff it has a ``"scan"`` key, else steady-state.
Keeping these accessors in one module stops the runner (which writes leaves),
the comparison Step, and the overlay viz (which both read them) from drifting
apart on the leaf format.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Tuple

# Reserved observable name for the sample-time vector inside a UTC leaf.
TIME_KEY = "time"
# Reserved observable name for the swept-parameter vector inside a scan leaf.
SCAN_KEY = "scan"
# Both reserved axis keys — excluded from a leaf's observables.
_AXIS_KEYS = (TIME_KEY, SCAN_KEY)


def is_utc(leaf: Dict[str, Any]) -> bool:
    """True for a UTC leaf (has a time vector); False otherwise."""
    return TIME_KEY in (leaf or {})


def is_scan(leaf: Dict[str, Any]) -> bool:
    """True for a repeated-task / parameter-scan leaf (has a ``scan`` vector)."""
    return SCAN_KEY in (leaf or {})


def kind_of(leaf: Dict[str, Any]) -> str:
    """``"utc"`` / ``"repeated_task"`` / ``"steady_state"`` (structural).

    A leaf should never carry both axes (the scan reduction strips the time
    axis). If it somehow does, ``utc`` wins and a warning is emitted.
    """
    leaf = leaf or {}
    if is_utc(leaf):
        if is_scan(leaf):
            warnings.warn(
                "result_leaf.kind_of: leaf carries both 'time' and 'scan' "
                "axes; treating as UTC",
                stacklevel=2,
            )
        return "utc"
    return "repeated_task" if is_scan(leaf) else "steady_state"


def axis_of(leaf: Dict[str, Any]) -> Tuple[str, List[float]]:
    """The leaf's reserved axis as ``(name, values)``.

    ``("time", [...])`` for UTC, ``("scan", [...])`` for a parameter scan, and
    ``("", [])`` for steady-state (no axis).
    """
    leaf = leaf or {}
    if TIME_KEY in leaf:
        return TIME_KEY, list(leaf.get(TIME_KEY) or [])
    if SCAN_KEY in leaf:
        return SCAN_KEY, list(leaf.get(SCAN_KEY) or [])
    return "", []


def time_of(leaf: Dict[str, Any]) -> List[float]:
    """The reserved time vector (empty list when absent)."""
    return list((leaf or {}).get(TIME_KEY) or [])


def scan_of(leaf: Dict[str, Any]) -> List[float]:
    """The reserved scan-parameter vector (empty list when absent)."""
    return list((leaf or {}).get(SCAN_KEY) or [])


def observables_of(leaf: Dict[str, Any]) -> Dict[str, Any]:
    """Every series except the reserved axis vectors (``time`` / ``scan``)."""
    return {k: v for k, v in (leaf or {}).items() if k not in _AXIS_KEYS}


def to_numeric_result(leaf: Dict[str, Any]) -> Dict[str, Any]:
    """UTC/scan leaf -> ``{time, columns, values}`` (the comparison math shape).

    The reserved axis (``time`` or ``scan``) is carried under the ``time`` key
    of the numeric result. The comparison math is axis-agnostic (it compares
    ``columns``/``values`` row-by-row), so a scan response curve is scored
    exactly like a time course — over the scan axis instead of time.
    """
    _, axis = axis_of(leaf)
    obs = observables_of(leaf)
    cols = list(obs.keys())
    n_rows = min((len(obs[c]) for c in cols), default=0)
    values = [[as_scalar(obs[c][r]) for c in cols] for r in range(n_rows)]
    return {"time": axis, "columns": cols, "values": values}


def as_scalar(x: Any) -> float:
    """Coerce one observable cell to a float.

    A cell is normally a bare scalar, but a constant-filled observable
    (parameter / compartment) can arrive length-1 nested — ``[v]`` instead of
    ``v`` — depending on the engine / SED-ML job shape, which used to crash
    ``float()`` and drop the whole model. Unwrap any length-1 nesting; a
    genuinely multi-valued cell collapses to its first element (a defensive last
    resort — such a cell is malformed to begin with)."""
    while isinstance(x, (list, tuple)):
        x = x[0] if x else 0.0
    return float(x)


def steady_state_scalars(leaf: Dict[str, Any]) -> Dict[str, float]:
    """Steady-state leaf -> ``{observable: scalar}`` (unwraps length-1 lists)."""
    out: Dict[str, float] = {}
    for k, v in observables_of(leaf).items():
        if isinstance(v, (list, tuple)):
            out[k] = float(v[0]) if v else 0.0
        else:
            out[k] = float(v)
    return out
