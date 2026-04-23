"""
Tests for summarize() in stats.py.

RESEARCH CONCEPT: These statistics are used to compare Case 1 and Case 2 matrices:
  - density: fraction of track-pairs that are "connected" (observed interactions)
  - symmetry: validity check (both matrices must be symmetric by construction)
  - row-sum distribution: degree distribution — how evenly tracks are connected
    Case 1 → bimodal (dominant tags = many neighbors, rare tags = isolated)
    Case 2 → unimodal (every track has ~comparable number of VA neighbors)
"""

import pytest
import numpy as np
import pandas as pd

from preprocess.stats import summarize

pytestmark = pytest.mark.unit

EXPECTED_KEYS = {
    "name", "shape", "density", "symmetric",
    "min", "max", "mean",
    "row_sum_min", "row_sum_max", "row_sum_mean", "row_sum_std",
}


def _make_df(array):
    return pd.DataFrame(array.astype(float))


# ── return contract ───────────────────────────────────────────────────────────

def test_summarize_returns_dict():
    mat = _make_df(np.eye(2))
    assert isinstance(summarize(mat, "test"), dict)


def test_summarize_required_keys():
    mat = _make_df(np.eye(3))
    result = summarize(mat, "test")
    assert set(result.keys()) == EXPECTED_KEYS


def test_summarize_name_is_preserved():
    mat = _make_df(np.eye(2))
    assert summarize(mat, "My Matrix")["name"] == "My Matrix"


# ── shape ─────────────────────────────────────────────────────────────────────

def test_summarize_shape_format_3x3():
    mat = _make_df(np.zeros((3, 3)))
    assert summarize(mat, "x")["shape"] == "3 x 3"


@pytest.mark.parametrize("size", [2, 5, 10])
def test_summarize_shape_reflects_matrix_size(size):
    mat = _make_df(np.zeros((size, size)))
    assert summarize(mat, "x")["shape"] == f"{size} x {size}"


# ── density ───────────────────────────────────────────────────────────────────

def test_summarize_density_identity_matrix():
    # 3×3 identity: 3 ones / 9 total = 0.3333
    mat = _make_df(np.eye(3))
    assert summarize(mat, "x")["density"] == pytest.approx(1 / 3, abs=0.001)


def test_summarize_density_all_ones():
    mat = _make_df(np.ones((4, 4)))
    assert summarize(mat, "x")["density"] == pytest.approx(1.0)


def test_summarize_density_all_zeros():
    mat = _make_df(np.zeros((4, 4)))
    assert summarize(mat, "x")["density"] == pytest.approx(0.0)


def test_summarize_density_known_value():
    # 2×2 matrix with 1 one out of 4 entries = 0.25
    mat = _make_df(np.array([[1, 0], [0, 0]]))
    assert summarize(mat, "x")["density"] == pytest.approx(0.25)


# ── symmetry ──────────────────────────────────────────────────────────────────

def test_summarize_symmetric_true_for_identity():
    mat = _make_df(np.eye(3))
    assert summarize(mat, "x")["symmetric"] is True


def test_summarize_symmetric_true_for_block_diagonal():
    arr = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=float)
    mat = _make_df(arr)
    assert summarize(mat, "x")["symmetric"] is True


def test_summarize_symmetric_false_for_asymmetric():
    # Upper-triangular matrix is NOT symmetric (off-diagonal differs)
    arr = np.array([[0, 1], [0, 0]], dtype=float)
    mat = _make_df(arr)
    assert summarize(mat, "x")["symmetric"] is False


# ── row-sum statistics ────────────────────────────────────────────────────────

def test_summarize_row_sum_known_values():
    # Row sums: 1, 2, 3 → min=1, max=3, mean=2
    arr = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
    ], dtype=float)
    mat = _make_df(arr)
    stats = summarize(mat, "x")
    assert stats["row_sum_min"] == pytest.approx(1.0)
    assert stats["row_sum_max"] == pytest.approx(3.0)
    assert stats["row_sum_mean"] == pytest.approx(2.0)


def test_summarize_row_sum_std_all_equal_rows():
    # All rows identical → std = 0
    mat = _make_df(np.ones((4, 4)))
    assert summarize(mat, "x")["row_sum_std"] == pytest.approx(0.0, abs=1e-5)


def test_summarize_min_max_mean_values():
    arr = np.array([[0.0, 1.0], [1.0, 0.0]])
    mat = _make_df(arr)
    stats = summarize(mat, "x")
    assert stats["min"] == pytest.approx(0.0)
    assert stats["max"] == pytest.approx(1.0)
    assert stats["mean"] == pytest.approx(0.5)
