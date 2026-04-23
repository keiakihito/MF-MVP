"""
Tests for build_tag_overlap_matrix and its private helpers.

RESEARCH CONCEPT (Case 1): The tag overlap matrix is the "discrete" signal.
Its defining property is block-diagonal structure: tracks sharing the same
emotional tag cluster together, producing disjoint groups. MF learns tight
latent factor clusters from this structure (one cluster per emotion category).

Key invariant:  R[i,j] = 1  ↔  tracks i and j share ≥1 character tag
                             ↔  (tag_vec_i · tag_vec_j) > 0
"""

import pytest
import numpy as np
import pandas as pd

from preprocess.matrices import (
    build_tag_overlap_matrix,
    _extract_tag_vectors,
    _tag_overlap_to_binary,
    CHARACTER_TAGS,
)

pytestmark = pytest.mark.unit


# ── _extract_tag_vectors ──────────────────────────────────────────────────────

def test_extract_tag_vectors_shape(tiny_df_disjoint_tags):
    tag_vecs = _extract_tag_vectors(tiny_df_disjoint_tags)
    assert tag_vecs.shape == (4, 4)


def test_extract_tag_vectors_dtype(tiny_df_disjoint_tags):
    tag_vecs = _extract_tag_vectors(tiny_df_disjoint_tags)
    assert tag_vecs.dtype == np.float32


def test_extract_tag_vectors_column_order(tiny_df_disjoint_tags):
    # energetic=1, tense=0, calm=0, lyrical=0 for first track
    tag_vecs = _extract_tag_vectors(tiny_df_disjoint_tags)
    assert list(tag_vecs[0]) == [1.0, 0.0, 0.0, 0.0]
    # energetic=0, tense=0, calm=1, lyrical=0 for third track
    assert list(tag_vecs[2]) == [0.0, 0.0, 1.0, 0.0]


def test_extract_tag_vectors_character_tags_constant():
    assert CHARACTER_TAGS == ["energetic", "tense", "calm", "lyrical"]


# ── _tag_overlap_to_binary ────────────────────────────────────────────────────

def test_tag_overlap_binary_same_tag_gives_one():
    # Two tracks both energetic: dot = 1 > 0 → R = 1
    tag_matrix = np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32)
    R = _tag_overlap_to_binary(tag_matrix)
    assert R[0, 1] == 1.0


def test_tag_overlap_binary_different_tags_gives_zero():
    # Track 0 = energetic, track 1 = calm: dot = 0 → R = 0
    tag_matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    R = _tag_overlap_to_binary(tag_matrix)
    assert R[0, 1] == 0.0


def test_tag_overlap_binary_partial_overlap_gives_one():
    # Track 0 = energetic+tense, track 1 = tense only: share "tense" → R = 1
    tag_matrix = np.array([[1, 1, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    R = _tag_overlap_to_binary(tag_matrix)
    assert R[0, 1] == 1.0


def test_tag_overlap_binary_output_is_only_zeros_and_ones():
    tag_matrix = np.eye(4, dtype=np.float32)
    R = _tag_overlap_to_binary(tag_matrix)
    assert set(R.flatten().tolist()) <= {0.0, 1.0}


# ── build_tag_overlap_matrix — return type and indexing ──────────────────────

def test_build_tag_overlap_returns_dataframe(tiny_df_disjoint_tags):
    mat = build_tag_overlap_matrix(tiny_df_disjoint_tags)
    assert isinstance(mat, pd.DataFrame)


def test_build_tag_overlap_index_name(tiny_df_disjoint_tags):
    mat = build_tag_overlap_matrix(tiny_df_disjoint_tags)
    assert mat.index.name == "track_id"
    assert mat.columns.name == "track_id"


def test_build_tag_overlap_index_values(tiny_df_disjoint_tags):
    mat = build_tag_overlap_matrix(tiny_df_disjoint_tags)
    assert list(mat.index) == [1, 2, 3, 4]
    assert list(mat.columns) == [1, 2, 3, 4]


def test_build_tag_overlap_shape(tiny_df_disjoint_tags):
    mat = build_tag_overlap_matrix(tiny_df_disjoint_tags)
    assert mat.shape == (4, 4)


# ── build_tag_overlap_matrix — mathematical properties ───────────────────────

def test_tag_overlap_binary_values_only(tiny_df_disjoint_tags):
    mat = build_tag_overlap_matrix(tiny_df_disjoint_tags)
    unique_vals = set(mat.values.flatten().tolist())
    assert unique_vals <= {0.0, 1.0}


def test_tag_overlap_symmetric(tiny_df_disjoint_tags):
    mat = build_tag_overlap_matrix(tiny_df_disjoint_tags)
    assert np.allclose(mat.values, mat.values.T)


def test_tag_overlap_diagonal_is_one_for_tagged_tracks(tiny_df_disjoint_tags):
    mat = build_tag_overlap_matrix(tiny_df_disjoint_tags)
    # All 4 tracks have ≥1 tag → self-dot-product > 0 → diagonal = 1
    for tid in [1, 2, 3, 4]:
        assert mat.loc[tid, tid] == 1.0


def test_tag_overlap_diagonal_is_zero_for_untagged_tracks(tiny_df_no_tags):
    # Zero-vector tracks: dot(0,0) = 0 → diagonal = 0
    mat = build_tag_overlap_matrix(tiny_df_no_tags)
    assert np.all(np.diag(mat.values) == 0.0)


# ── RESEARCH CONCEPT: block-diagonal structure ────────────────────────────────

def test_tag_overlap_block_diagonal_within_group_a(tiny_df_disjoint_tags):
    mat = build_tag_overlap_matrix(tiny_df_disjoint_tags)
    # Tracks 1 and 2 are both energetic → should be connected
    assert mat.loc[1, 2] == 1.0
    assert mat.loc[2, 1] == 1.0


def test_tag_overlap_block_diagonal_within_group_b(tiny_df_disjoint_tags):
    mat = build_tag_overlap_matrix(tiny_df_disjoint_tags)
    # Tracks 3 and 4 are both calm → should be connected
    assert mat.loc[3, 4] == 1.0
    assert mat.loc[4, 3] == 1.0


def test_tag_overlap_block_diagonal_cross_group_is_zero(tiny_df_disjoint_tags):
    mat = build_tag_overlap_matrix(tiny_df_disjoint_tags)
    # energetic ↔ calm pairs should have NO connection
    for a in [1, 2]:
        for b in [3, 4]:
            assert mat.loc[a, b] == 0.0, f"Expected R[{a},{b}]=0 (cross-group)"
            assert mat.loc[b, a] == 0.0


def test_tag_overlap_bridge_track_connects_all_pairs(tiny_df_overlapping_tags):
    # Track 2 has both energetic+calm → bridges the two groups.
    # All 3 pairs (1-2, 2-3, 1-3 via transitivity through 2) should have R=1.
    mat = build_tag_overlap_matrix(tiny_df_overlapping_tags)
    assert mat.loc[1, 2] == 1.0  # share energetic
    assert mat.loc[2, 3] == 1.0  # share calm
    # Direct connection 1-3: track 1 is energetic only, track 3 is calm only → no shared tag
    assert mat.loc[1, 3] == 0.0


@pytest.mark.parametrize("n_tracks", [1, 2, 10])
def test_tag_overlap_single_tag_group_is_all_ones(n_tracks):
    # All N tracks share the same single tag → fully dense matrix (all ones)
    df = pd.DataFrame({
        "track_id": list(range(n_tracks)),
        "energetic": [1] * n_tracks,
        "tense":     [0] * n_tracks,
        "calm":      [0] * n_tracks,
        "lyrical":   [0] * n_tracks,
        "valence":   [0.5] * n_tracks,
        "arousal":   [0.5] * n_tracks,
    })
    mat = build_tag_overlap_matrix(df)
    assert np.all(mat.values == 1.0)
