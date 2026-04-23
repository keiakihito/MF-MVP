"""
Tests for build_va_distance_matrix and its private helpers.

RESEARCH CONCEPT (Case 2): The VA distance matrix is the "continuous" signal.
Its defining property is smooth/gradient structure: tracks close in
Valence-Arousal space get R=1, creating a graded neighborhood rather than
discrete blocks. The threshold controls density vs. connectivity.

Key formula:  similarity(i,j) = 1 − euclidean_dist(VA_i, VA_j) / √2
              R[i,j] = 1  ↔  similarity(i,j) ≥ threshold

√2 normalizes by the maximum possible distance in the unit [0,1]² space,
so similarity is always in [0, 1].
"""

import pytest
import numpy as np
import pandas as pd

from preprocess.matrices import (
    build_va_distance_matrix,
    _extract_va_vectors,
    _pairwise_va_similarity,
    _threshold_to_binary,
)

pytestmark = pytest.mark.unit


# ── _extract_va_vectors ───────────────────────────────────────────────────────

def test_extract_va_vectors_shape(tiny_df_va_spread):
    va = _extract_va_vectors(tiny_df_va_spread)
    assert va.shape == (4, 2)


def test_extract_va_vectors_dtype(tiny_df_va_spread):
    va = _extract_va_vectors(tiny_df_va_spread)
    assert va.dtype == np.float32


def test_extract_va_vectors_column_order(tiny_df_va_spread):
    va = _extract_va_vectors(tiny_df_va_spread)
    # First track: valence=0.50, arousal=0.50
    assert va[0, 0] == pytest.approx(0.50, abs=1e-5)
    assert va[0, 1] == pytest.approx(0.50, abs=1e-5)
    # Last (isolated) track: valence=0.90, arousal=0.50
    assert va[3, 0] == pytest.approx(0.90, abs=1e-5)
    assert va[3, 1] == pytest.approx(0.50, abs=1e-5)


# ── _pairwise_va_similarity ───────────────────────────────────────────────────

def test_pairwise_va_similarity_self_similarity_is_one():
    va = np.array([[0.3, 0.7], [0.6, 0.2], [0.9, 0.9]], dtype=np.float32)
    sim = _pairwise_va_similarity(va)
    # dist(x, x) = 0 → sim = 1 - 0/√2 = 1.0
    assert np.allclose(np.diag(sim), 1.0)


def test_pairwise_va_similarity_max_distance_gives_zero():
    # (0,0) and (1,1) are diagonally opposite corners of [0,1]²
    # dist = √2, so sim = 1 - √2/√2 = 0.0
    va = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    sim = _pairwise_va_similarity(va)
    assert sim[0, 1] == pytest.approx(0.0, abs=1e-5)


def test_pairwise_va_similarity_range_zero_to_one():
    rng = np.random.default_rng(42)
    va = rng.random((20, 2)).astype(np.float32)
    sim = _pairwise_va_similarity(va)
    assert sim.min() >= 0.0 - 1e-6
    assert sim.max() <= 1.0 + 1e-6


def test_pairwise_va_similarity_symmetric():
    rng = np.random.default_rng(7)
    va = rng.random((5, 2)).astype(np.float32)
    sim = _pairwise_va_similarity(va)
    assert np.allclose(sim, sim.T)


def test_pairwise_va_similarity_known_value():
    # dist((0,0),(0.1,0)) = 0.1 → sim = 1 - 0.1/√2 ≈ 0.9293
    va = np.array([[0.0, 0.0], [0.1, 0.0]], dtype=np.float32)
    sim = _pairwise_va_similarity(va)
    expected = 1.0 - 0.1 / np.sqrt(2)
    assert sim[0, 1] == pytest.approx(expected, abs=1e-5)


# ── _threshold_to_binary ──────────────────────────────────────────────────────

def test_threshold_to_binary_above_threshold_is_one():
    sim = np.array([[1.0, 0.96], [0.96, 1.0]], dtype=np.float32)
    R = _threshold_to_binary(sim, threshold=0.95)
    assert R[0, 1] == 1.0


def test_threshold_to_binary_below_threshold_is_zero():
    sim = np.array([[1.0, 0.80], [0.80, 1.0]], dtype=np.float32)
    R = _threshold_to_binary(sim, threshold=0.95)
    assert R[0, 1] == 0.0


def test_threshold_to_binary_exactly_at_threshold_is_one():
    # The condition is >=, so equality → R = 1
    sim = np.array([[1.0, 0.95], [0.95, 1.0]], dtype=np.float32)
    R = _threshold_to_binary(sim, threshold=0.95)
    assert R[0, 1] == 1.0


def test_threshold_to_binary_output_is_only_zeros_and_ones():
    rng = np.random.default_rng(0)
    sim = rng.random((5, 5)).astype(np.float32)
    R = _threshold_to_binary(sim, threshold=0.5)
    assert set(R.flatten().tolist()) <= {0.0, 1.0}


# ── build_va_distance_matrix — return type and indexing ──────────────────────

def test_build_va_matrix_returns_dataframe(tiny_df_va_spread):
    mat = build_va_distance_matrix(tiny_df_va_spread, threshold=0.95)
    assert isinstance(mat, pd.DataFrame)


def test_build_va_matrix_index_name(tiny_df_va_spread):
    mat = build_va_distance_matrix(tiny_df_va_spread, threshold=0.95)
    assert mat.index.name == "track_id"
    assert mat.columns.name == "track_id"


def test_build_va_matrix_shape(tiny_df_va_spread):
    mat = build_va_distance_matrix(tiny_df_va_spread, threshold=0.95)
    assert mat.shape == (4, 4)


# ── build_va_distance_matrix — mathematical properties ───────────────────────

def test_va_matrix_binary_values_only(tiny_df_va_spread):
    mat = build_va_distance_matrix(tiny_df_va_spread, threshold=0.95)
    assert set(mat.values.flatten().tolist()) <= {0.0, 1.0}


def test_va_matrix_symmetric(tiny_df_va_spread):
    mat = build_va_distance_matrix(tiny_df_va_spread, threshold=0.95)
    assert np.allclose(mat.values, mat.values.T)


def test_va_matrix_diagonal_is_all_ones(tiny_df_va_spread):
    # Unlike the tag matrix, VA diagonal is always 1 because sim(i,i) = 1.0 ≥ any threshold ≤ 1
    mat = build_va_distance_matrix(tiny_df_va_spread, threshold=0.95)
    assert np.all(np.diag(mat.values) == 1.0)


# ── RESEARCH CONCEPT: smooth/gradient structure ───────────────────────────────

def test_va_matrix_isolated_track_has_no_off_diagonal_connections(tiny_df_va_spread):
    # Track 4 (index 3) is at (0.9, 0.5) — far from the (0.5, 0.5) cluster
    mat = build_va_distance_matrix(tiny_df_va_spread, threshold=0.95)
    off_diag_row = mat.values[3].copy()
    off_diag_row[3] = 0.0  # ignore self-connection
    assert off_diag_row.sum() == 0.0


def test_va_matrix_clustered_tracks_all_connected(tiny_df_va_spread):
    # Tracks 1, 2, 3 (indices 0,1,2) are near (0.5,0.5) and should all connect
    mat = build_va_distance_matrix(tiny_df_va_spread, threshold=0.95)
    assert mat.values[0, 1] == 1.0
    assert mat.values[0, 2] == 1.0
    assert mat.values[1, 2] == 1.0


@pytest.mark.parametrize("threshold", [0.5, 0.7, 0.9, 0.95, 1.0])
def test_va_matrix_density_decreases_with_threshold(tiny_df_va_spread, threshold):
    # Higher threshold → stricter → fewer connections → lower density.
    # This parametrize documents how the research threshold of 0.95 was chosen.
    mats = {
        t: build_va_distance_matrix(tiny_df_va_spread, threshold=t)
        for t in [0.5, 0.7, 0.9, 0.95, 1.0]
    }
    thresholds = sorted(mats.keys())
    densities = [mats[t].values.mean() for t in thresholds]
    for i in range(len(densities) - 1):
        assert densities[i] >= densities[i + 1], (
            f"Density should be non-increasing: "
            f"t={thresholds[i]} density={densities[i]:.3f} "
            f"but t={thresholds[i+1]} density={densities[i+1]:.3f}"
        )
