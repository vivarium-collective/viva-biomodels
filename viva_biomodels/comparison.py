"""Shared simulator-comparison math.

Used by both the post-hoc HTML report builder (``analysis.py``) and the
runtime :class:`SimulatorComparisonStep` (``steps/simulator_comparison.py``).

Inputs are per-engine ``numeric_result`` payloads — dicts of shape
``{time: list[float], columns: list[str], values: list[list[float]]}``,
emitted by the COPASI / Tellurium UTC steps.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional


# Normalized-RMSE thresholds (mean across shared species).
BUCKET_THRESHOLDS = [
    ("good",       0.01,     "Good (≤1%)"),
    ("borderline", 0.10,     "Borderline (1–10%)"),
    ("large",      math.inf, "Large diff (>10%)"),
]
NO_COMPARISON_BUCKET = ("none", "No comparison")

# A trace whose largest magnitude across both engines is at or below this floor
# is treated as effectively zero and set aside from the normalized-RMSE mean —
# exactly like an exactly-both-zero trace. Without it, one engine reporting a
# rounding-noise constant (e.g. 2.2e-16) against another's exact 0 normalizes to
# nRMSE 1.0 and lands in the "large diff" bucket, a false divergence that is
# only machine epsilon (reviewer findings, 2026-10-28). 1e-12 sits far above
# float rounding noise yet below any physically meaningful trajectory.
ZERO_TRACE_ATOL = 1e-12


def union_species(engines: Dict[str, Any]) -> List[str]:
    """Return the ordered union of species columns across all engines."""
    species: List[str] = []
    seen: set = set()
    for payload in engines.values():
        if not payload:
            continue
        for c in payload.get("columns", []):
            if c not in seen:
                seen.add(c)
                species.append(c)
    return species


def compare_two_engines(
    engine_a: Optional[Dict[str, Any]],
    engine_b: Optional[Dict[str, Any]],
    name_a: str = "a",
    name_b: str = "b",
) -> Dict[str, Any]:
    """Compare two numeric_result payloads species-by-species.

    Computes RMSE and normalized RMSE (RMSE / max|y|) for every species that
    appears in BOTH engines' columns, then a mean nRMSE across shared species
    and a coarse bucket label.

    Returns the shape consumed by the HTML report's metadata card and the
    dashboard Step. Returns the ``none`` bucket if either engine is missing
    or no species are shared.
    """
    engines = {name_a: engine_a, name_b: engine_b}
    species = union_species(engines)

    rmse_by_species: Dict[str, float] = {}
    nrmse_by_species: Dict[str, float] = {}
    n_shared = 0

    for sp in species:
        ys_per_engine: Dict[str, List[float]] = {}
        for eng_name, payload in engines.items():
            if not payload:
                continue
            cols = payload.get("columns", [])
            if sp not in cols:
                continue
            j = cols.index(sp)
            ys_per_engine[eng_name] = [row[j] for row in payload.get("values", [])]
        if len(ys_per_engine) < 2:
            continue
        n_shared += 1
        keys = sorted(ys_per_engine.keys())
        y1, y2 = ys_per_engine[keys[0]], ys_per_engine[keys[1]]
        n = min(len(y1), len(y2))
        if n == 0:
            continue
        rmse = math.sqrt(sum((y1[k] - y2[k]) ** 2 for k in range(n)) / n)
        rmse_by_species[sp] = rmse
        denom = max((abs(v) for v in y1[:n]), default=0.0)
        denom = max(denom, max((abs(v) for v in y2[:n]), default=0.0))
        # Set aside traces that are both effectively zero (within machine
        # epsilon) — normalizing their RMSE would amplify rounding noise into a
        # spurious divergence. See ZERO_TRACE_ATOL.
        if denom > ZERO_TRACE_ATOL:
            nrmse_by_species[sp] = rmse / denom

    if nrmse_by_species:
        mean_nrmse: Optional[float] = sum(nrmse_by_species.values()) / len(nrmse_by_species)
    else:
        mean_nrmse = None

    # Closeness (BioSimulations allclose-style), max over shared species; the
    # pair is `close` only if every shared species is close.
    pair_close, pair_score, have_pair = True, 0.0, False
    for sp in species:
        ys: Dict[str, List[float]] = {}
        for eng_name, payload in engines.items():
            if not payload:
                continue
            cols = payload.get("columns", [])
            if sp in cols:
                j = cols.index(sp)
                ys[eng_name] = [row[j] for row in payload.get("values", [])]
        if len(ys) < 2:
            continue
        have_pair = True
        keys = sorted(ys)
        c, s = closeness_score(ys[keys[0]], ys[keys[1]])
        pair_close = pair_close and c
        pair_score = max(pair_score, s)

    # NB: delegate to bucket_for, which has the final fallback. The inline loop
    # used to omit it, so a NaN mean_nrmse (from series that diverge to inf/NaN,
    # e.g. a non-convergent steady state) matched no threshold — `nan <= inf` is
    # False — leaving bucket_id unbound and crashing the whole task.
    bucket_id, bucket_label = bucket_for(mean_nrmse)

    return {
        "n_shared": n_shared,
        "rmse_by_species": rmse_by_species,
        "nrmse_by_species": nrmse_by_species,
        "mean_nrmse": mean_nrmse,
        "bucket": bucket_id,
        "bucket_label": bucket_label,
        "closeness_score": pair_score if have_pair else None,
        "closeness_close": bool(pair_close) if have_pair else False,
    }


def bucket_for(mean_nrmse: Optional[float]) -> tuple:
    """Map a mean-nRMSE value to a ``(bucket_id, bucket_label)`` pair."""
    if mean_nrmse is None:
        return NO_COMPARISON_BUCKET
    for bid, threshold, label in BUCKET_THRESHOLDS:
        if mean_nrmse <= threshold:
            return bid, label
    return BUCKET_THRESHOLDS[-1][0], BUCKET_THRESHOLDS[-1][2]


def closeness_score(y1, y2):
    """BioSimulations allclose-style score for two aligned value series.

    Faithful to biosimulations_runutils ``hdf5_compare.compare_arrays``:
    ``atol = max(1e-3, 1e-5*max|a|, 1e-5*max|b|)``, ``rtol = 1e-4``,
    ``score = max(|a-b| / (atol + rtol*|b|))``; ``close = score <= 1``.
    Returns ``(close: bool, score: float)``. NaN in either series ->
    ``(False, 1e10)``; arithmetic blow-up -> ``(False, 1e12)``.
    """
    a = [float(x) for x in y1]
    b = [float(x) for x in y2]
    n = min(len(a), len(b))
    if n == 0:
        return True, 0.0
    a, b = a[:n], b[:n]
    if any(math.isnan(x) for x in a) or any(math.isnan(x) for x in b):
        return False, 1e10
    atol = max(1e-3, max(abs(x) for x in a) * 1e-5, max(abs(x) for x in b) * 1e-5)
    rtol = 1e-4
    try:
        score = max(abs(a[i] - b[i]) / (atol + rtol * abs(b[i])) for i in range(n))
    except (ZeroDivisionError, OverflowError, ValueError):
        return False, 1e12
    return score <= 1.0, score


CLOSENESS_ERROR_FLOOR = 1e10


def closeness_bucket_for(score: Optional[float]) -> tuple:
    """Map a closeness score to a ``(bucket_id, label)`` pair."""
    if score is None:
        return NO_COMPARISON_BUCKET
    if score >= CLOSENESS_ERROR_FLOOR:
        return "error", "Error"
    if score <= 1.0:
        return "close", "Close (≤1)"
    return "not_close", "Not close (>1)"


def _closeness_rollup(pairs: Dict[str, Any], present: List[str]) -> Dict[str, Any]:
    """Build the closeness matrix + worst-pair summary from per-pair results.

    Shared by the UTC and steady-state ``compare_n_engines`` variants (each
    pair result carries ``closeness_score`` from ``compare_two_engines*``).
    """
    matrix = {a: {b: None for b in present} for a in present}
    max_score: Optional[float] = None
    worst: Optional[List[str]] = None
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            cs = (pairs.get(f"{a}__{b}") or {}).get("closeness_score")
            matrix[a][b] = cs
            matrix[b][a] = cs
            if cs is not None and (max_score is None or cs > max_score):
                max_score, worst = cs, [a, b]
    bucket_id, bucket_label = closeness_bucket_for(max_score)
    return {
        "matrix_closeness":      matrix,
        "max_score":             max_score,
        "worst_pair_closeness":  worst,
        "closeness_bucket":      bucket_id,
        "closeness_bucket_label": bucket_label,
    }


def compare_n_engines(engines: Dict[str, Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """All-pairs comparison across N named engines (simulators + references).

    Runs :func:`compare_two_engines` on every unordered pair of engines that
    carry data, and summarizes the worst (largest mean-nRMSE) pair.

    Returns::

        {
          "engines":   ["copasi", "tellurium", ...],   # engines with data
          "pairs": {
            "copasi__tellurium": {<compare_two_engines result>}, ...
          },
          "matrix":    {a: {b: mean_nrmse_or_None}},     # symmetric, diag None
          "max_nrmse": float | None,                     # worst pair's mean nRMSE
          "worst_pair": [a, b] | None,
          "bucket":     str,                             # bucket of the worst pair
          "bucket_label": str,
        }
    """
    present = sorted(
        name for name, payload in engines.items()
        if payload and payload.get("columns")
    )

    pairs: Dict[str, Any] = {}
    matrix: Dict[str, Dict[str, Optional[float]]] = {
        a: {b: None for b in present} for a in present
    }
    max_nrmse: Optional[float] = None
    worst_pair: Optional[List[str]] = None

    for i, a in enumerate(present):
        for b in present[i + 1:]:
            result = compare_two_engines(engines[a], engines[b], name_a=a, name_b=b)
            pairs[f"{a}__{b}"] = result
            mean = result.get("mean_nrmse")
            matrix[a][b] = mean
            matrix[b][a] = mean
            if mean is not None and (max_nrmse is None or mean > max_nrmse):
                max_nrmse = mean
                worst_pair = [a, b]

    bucket_id, bucket_label = bucket_for(max_nrmse)
    return {
        "engines": present,
        "pairs": pairs,
        "matrix": matrix,
        "max_nrmse": max_nrmse,
        "worst_pair": worst_pair,
        "bucket": bucket_id,
        "bucket_label": bucket_label,
        **_closeness_rollup(pairs, present),
    }


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

    # Closeness over shared scalars (each a length-1 series); max over shared.
    pair_close, pair_score = True, 0.0
    for sp in shared:
        c, s = closeness_score([a[sp]], [b[sp]])
        pair_close = pair_close and c
        pair_score = max(pair_score, s)

    bucket_id, bucket_label = bucket_for(mean_nrmse)
    return {
        "n_shared":         len(shared),
        "rmse_by_species":  {},  # SS pairs have no time-series RMSE
        "nrmse_by_species": nrmse_by_species,
        "mean_nrmse":       mean_nrmse,
        "bucket":           bucket_id,
        "bucket_label":     bucket_label,
        "closeness_score":  pair_score if shared else None,
        "closeness_close":  bool(pair_close) if shared else False,
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
        **_closeness_rollup(pairs, present),
    }
