"""Unit tests for viva_biomodels.result_leaf — leaf-shape accessors."""
from __future__ import annotations

from viva_biomodels.result_leaf import to_numeric_result


def test_to_numeric_result_scalar_cells():
    leaf = {"time": [0.0, 1.0, 2.0], "A": [1.0, 2.0, 3.0]}
    out = to_numeric_result(leaf)
    assert out["columns"] == ["A"]
    assert out["values"] == [[1.0], [2.0], [3.0]]
    assert out["time"] == [0.0, 1.0, 2.0]


def test_to_numeric_result_unwraps_length1_nested_cells():
    """A constant-filled observable can arrive length-1 nested ([v]) rather than
    bare; it must coerce to a scalar instead of crashing float() (reviewer
    re-run: constant parameter/compartment traces on some job shapes)."""
    leaf = {
        "time": [0.0, 1.0, 2.0],
        "A": [1.0, 2.0, 3.0],          # normal scalar cells
        "k": [[9.0], [9.0], [9.0]],     # constant param, length-1 nested
    }
    out = to_numeric_result(leaf)
    ki = out["columns"].index("k")
    ai = out["columns"].index("A")
    assert [row[ki] for row in out["values"]] == [9.0, 9.0, 9.0]
    assert [row[ai] for row in out["values"]] == [1.0, 2.0, 3.0]
