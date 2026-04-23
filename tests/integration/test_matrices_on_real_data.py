"""
Integration tests: run the matrix builders on the real 203-track dataset.

RESEARCH CONCEPT VALIDATION:
  - Case 1 (tag overlap) density ≈ 0.283 with block-diagonal structure
  - Case 2 (VA distance, t=0.95) density ≈ 0.248 with smooth structure
  - Case 2 density < Case 1 density (research finding for this dataset)
  - Both matrices are symmetric
  - Diagonal differences: Case 1 has zeros for untagged tracks; Case 2 is all-ones
  - The saved CSVs match what the builders produce (reproducibility check)
"""

import pytest
import numpy as np

from preprocess.matrices import build_tag_overlap_matrix, build_va_distance_matrix
from preprocess.stats import summarize

pytestmark = pytest.mark.integration


# ── Case 1: tag overlap matrix on real data ───────────────────────────────────

def test_real_tag_matrix_shape(real_labels):
    mat = build_tag_overlap_matrix(real_labels)
    assert mat.shape == (203, 203)


def test_real_tag_matrix_symmetric(real_labels):
    mat = build_tag_overlap_matrix(real_labels)
    assert np.allclose(mat.values, mat.values.T)


def test_real_tag_matrix_binary_only(real_labels):
    mat = build_tag_overlap_matrix(real_labels)
    unique = set(np.unique(mat.values.flatten()))
    assert unique <= {0.0, 1.0}


def test_real_tag_matrix_density_approx(real_labels):
    # Expected density ≈ 0.283; allow a window for robustness
    mat = build_tag_overlap_matrix(real_labels)
    stats = summarize(mat, "tag")
    assert 0.25 <= stats["density"] <= 0.32, (
        f"Tag matrix density {stats['density']:.4f} outside expected range [0.25, 0.32]"
    )


def test_real_tag_matrix_diagonal_reflects_tagged_tracks(real_labels):
    # Tracks with no tags have a zero-vector → self dot = 0 → diagonal = 0
    # Tracks with ≥1 tag → diagonal = 1
    mat = build_tag_overlap_matrix(real_labels)
    diag = np.diag(mat.values)
    tag_cols = ["energetic", "tense", "calm", "lyrical"]
    has_tag = (real_labels[tag_cols].sum(axis=1) > 0).values
    # Diagonal entry should be 1 iff track has a tag
    assert np.all(diag[has_tag] == 1.0)
    assert np.all(diag[~has_tag] == 0.0)


# ── Case 2: VA distance matrix on real data ───────────────────────────────────

def test_real_va_matrix_shape(real_labels):
    mat = build_va_distance_matrix(real_labels, threshold=0.95)
    assert mat.shape == (203, 203)


def test_real_va_matrix_symmetric(real_labels):
    mat = build_va_distance_matrix(real_labels, threshold=0.95)
    assert np.allclose(mat.values, mat.values.T)


def test_real_va_matrix_binary_only(real_labels):
    mat = build_va_distance_matrix(real_labels, threshold=0.95)
    unique = set(np.unique(mat.values.flatten()))
    assert unique <= {0.0, 1.0}


def test_real_va_matrix_density_approx(real_labels):
    # Expected density ≈ 0.248 at threshold=0.95; allow a window
    mat = build_va_distance_matrix(real_labels, threshold=0.95)
    stats = summarize(mat, "va")
    assert 0.20 <= stats["density"] <= 0.30, (
        f"VA matrix density {stats['density']:.4f} outside expected range [0.20, 0.30]"
    )


def test_real_va_matrix_diagonal_all_ones(real_labels):
    # Every track has similarity 1.0 with itself → diagonal = 1 for all 203 tracks
    mat = build_va_distance_matrix(real_labels, threshold=0.95)
    assert np.all(np.diag(mat.values) == 1.0)


# ── RESEARCH FINDING: VA density < tag density ───────────────────────────────

def test_real_va_matrix_density_lower_than_tag(real_labels):
    # This is a key research finding: the VA distance matrix is sparser than the
    # tag overlap matrix for this music dataset.
    tag_mat = build_tag_overlap_matrix(real_labels)
    va_mat  = build_va_distance_matrix(real_labels, threshold=0.95)
    tag_density = tag_mat.values.mean()
    va_density  = va_mat.values.mean()
    assert va_density < tag_density, (
        f"Expected VA density ({va_density:.4f}) < tag density ({tag_density:.4f})"
    )


# ── Reproducibility: built matrices match saved CSVs ─────────────────────────

def test_real_tag_matrix_matches_saved_csv(real_labels, real_R_tag):
    # The saved CSV must exactly match what the builder produces from labels.
    # Validates that the CSV was not modified after generation.
    built = build_tag_overlap_matrix(real_labels)
    assert built.shape == real_R_tag.shape
    assert np.allclose(built.values, real_R_tag.values), (
        "build_tag_overlap_matrix output does not match matrix_case1_tag_overlap.csv"
    )


def test_real_va_matrix_matches_saved_csv(real_labels, real_R_va):
    # The saved CSV was generated with threshold=0.95 (see generate_matrices.py).
    built = build_va_distance_matrix(real_labels, threshold=0.95)
    assert built.shape == real_R_va.shape
    assert np.allclose(built.values, real_R_va.values), (
        "build_va_distance_matrix(t=0.95) output does not match matrix_case2_va_distance_t095.csv"
    )
