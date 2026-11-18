"""Round-trip tests for the two-tier storage layer (index + parquet series).

Covers the subtle steady-state case: SS leaves carry no time vector, so they
must round-trip through parquet (NaN-time marker) and come back as steady-state
(is_utc False) rather than masquerading as a time course.
"""
import pytest

pytest.importorskip("pyarrow")

import math

import pyarrow.parquet as pq

from viva_biomodels import result_leaf
from viva_biomodels.two_tier import series_table, write_model
from viva_biomodels.lazy_viewer import _parquet_leaves_aligned


def _model_results():
    return {
        "auto_ten_seconds": {  # UTC: time + observables
            "copasi": {"time": [0.0, 1.0, 2.0], "A": [10.0, 5.0, 2.5], "B": [0.0, 5.0, 7.5]},
            "tellurium": {"time": [0.0, 1.0, 2.0], "A": [10.0, 5.1, 2.4], "B": [0.0, 4.9, 7.6]},
        },
        "auto_steady_state": {  # steady-state: length-1 values, NO time
            "amici": {"A": [2.5], "B": [7.5]},
            "pysces": {"A": [2.51], "B": [7.49]},
        },
    }


def test_two_tier_roundtrip_utc_and_steady_state(tmp_path):
    bid = "BIOMD0000000001"
    entry = write_model(bid, _model_results(), {}, tmp_path, max_points=200)

    # index entry records both jobs with the right kind.
    kinds = {job: j["kind"] for job, j in entry["jobs"].items()}
    assert kinds["auto_ten_seconds"] == "utc"
    assert kinds["auto_steady_state"] == "steady_state"

    leaves = _parquet_leaves_aligned(tmp_path, bid)

    # UTC leaf comes back WITH a time vector.
    utc = leaves["auto_ten_seconds"]["copasi"]
    assert result_leaf.is_utc(utc) is True
    assert utc["time"] == pytest.approx([0.0, 1.0, 2.0])
    assert utc["A"] == pytest.approx([10.0, 5.0, 2.5])

    # Steady-state leaf comes back WITHOUT a time key (is_utc False), values intact.
    ss = leaves["auto_steady_state"]["amici"]
    assert result_leaf.is_utc(ss) is False
    assert "time" not in ss
    assert ss["A"] == pytest.approx([2.5])
    assert ss["B"] == pytest.approx([7.5])


def test_series_table_labels_variable_roles():
    results = {
        "auto_ten_seconds": {
            "copasi": {"time": [0.0, 1.0], "A": [1.0, 2.0], "k1": [3.0, 3.0], "cell": [2.0, 2.0]},
        }
    }
    roles = {"A": "species", "k1": "parameter", "cell": "compartment"}
    tbl = series_table(results, roles=roles).to_pydict()
    by_var = dict(zip(tbl["variable"], tbl["role"]))
    assert by_var == {"A": "species", "k1": "parameter", "cell": "compartment"}


def test_series_table_role_defaults_to_unknown_without_map():
    results = {"j": {"copasi": {"time": [0.0, 1.0], "A": [1.0, 2.0]}}}
    tbl = series_table(results).to_pydict()
    assert set(tbl["role"]) == {"unknown"}


def test_series_table_collapses_inf_and_nan_to_single_sentinel():
    results = {
        "j": {
            "copasi": {
                "time": [0.0, 1.0, 2.0, 3.0],
                "A": [1.0, float("inf"), float("-inf"), float("nan")],
            }
        }
    }
    vals = series_table(results, roles={"A": "species"}).to_pydict()["value"]
    assert vals[0] == pytest.approx(1.0)
    # every non-finite point becomes NaN — no inf survives, one encoding.
    assert all(math.isnan(v) for v in vals[1:])
    assert not any(math.isinf(v) for v in vals)
