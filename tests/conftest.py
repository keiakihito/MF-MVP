"""
conftest.py — Shared fixtures for the MF research test suite.

Fixture taxonomy:
  tiny_*       : 4–6 track DataFrames/arrays, entirely in-memory, no file I/O
  block_*      : fixtures designed to produce perfect block-diagonal structure
  real_*       : session-scoped fixtures that read from data/ (integration only)
"""

import os
import pytest
import numpy as np
import pandas as pd
import torch

from preprocess.matrices import build_tag_overlap_matrix, build_va_distance_matrix
from mf.mf_experiment import MFModel

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ── Tiny label DataFrames ─────────────────────────────────────────────────────

@pytest.fixture
def tiny_df_disjoint_tags():
    """
    4 tracks split into two tag-disjoint groups:
      Group A (energetic): tracks 1, 2
      Group B (calm):      tracks 3, 4
    Expected tag overlap matrix: perfect 2×2 block diagonal (within-group=1, cross=0).
    """
    return pd.DataFrame({
        "track_id": [1, 2, 3, 4],
        "energetic": [1, 1, 0, 0],
        "tense":     [0, 0, 0, 0],
        "calm":      [0, 0, 1, 1],
        "lyrical":   [0, 0, 0, 0],
        "valence":   [0.3, 0.32, 0.7, 0.72],
        "arousal":   [0.4, 0.41, 0.6, 0.61],
    })


@pytest.fixture
def tiny_df_overlapping_tags():
    """
    3 tracks where track 2 bridges two groups via multi-tag membership:
      track 1: energetic only
      track 2: energetic + calm  (bridges both groups)
      track 3: calm only
    All pairs should have R=1 (every pair shares ≥1 tag via track 2's membership).
    """
    return pd.DataFrame({
        "track_id": [1, 2, 3],
        "energetic": [1, 1, 0],
        "tense":     [0, 0, 0],
        "calm":      [0, 1, 1],
        "lyrical":   [0, 0, 0],
        "valence":   [0.3, 0.5, 0.7],
        "arousal":   [0.4, 0.5, 0.6],
    })


@pytest.fixture
def tiny_df_no_tags():
    """
    2 tracks with all-zero tag vectors.
    Expected: R = [[0, 0], [0, 0]] — no similarity signal at all.
    Tests the boundary case where untagged tracks are isolated from everything.
    """
    return pd.DataFrame({
        "track_id": [1, 2],
        "energetic": [0, 0],
        "tense":     [0, 0],
        "calm":      [0, 0],
        "lyrical":   [0, 0],
        "valence":   [0.5, 0.5],
        "arousal":   [0.5, 0.5],
    })


@pytest.fixture
def tiny_df_va_spread():
    """
    4 tracks: tracks 1–3 clustered near (0.5, 0.5); track 4 isolated at (0.9, 0.5).
    At threshold=0.95, tracks 1–3 connect to each other but NOT to track 4.
    Tests the continuous/smooth structure of the VA distance matrix.
    """
    return pd.DataFrame({
        "track_id": [1, 2, 3, 4],
        "energetic": [0, 0, 0, 0],
        "tense":     [0, 0, 0, 0],
        "calm":      [0, 0, 0, 0],
        "lyrical":   [0, 0, 0, 0],
        "valence":   [0.50, 0.52, 0.55, 0.90],
        "arousal":   [0.50, 0.50, 0.50, 0.50],
    })


# ── Pre-built matrix fixtures ─────────────────────────────────────────────────

@pytest.fixture
def block_tag_matrix(tiny_df_disjoint_tags):
    """4×4 block-diagonal tag overlap DataFrame (2 groups of 2 tracks each)."""
    return build_tag_overlap_matrix(tiny_df_disjoint_tags)


@pytest.fixture
def tiny_R_block():
    """
    Pure numpy 4×4 block-diagonal array for MF tests.
    No dependency on matrix-building code — directly tests MF mechanics.
    """
    return np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    ], dtype=np.float32)


@pytest.fixture
def tiny_R_dense():
    """6×6 nearly-full matrix with one zero block, for training set size tests."""
    R = np.ones((6, 6), dtype=np.float32)
    R[4, 5] = 0.0
    R[5, 4] = 0.0
    return R


# ── MFModel fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def small_model():
    """Untrained MFModel(N=10, f=4) for shape and API tests."""
    return MFModel(N=10, f=4)


@pytest.fixture
def deterministic_model():
    """
    MFModel(N=5, f=3) with manually set weights so that the score ordering
    for query track 0 is deterministic:
      track 2 → dot product = 1.0 (highest)
      track 3 → dot product = 0.5 (second)
      track 4 → dot product = 0.1 (third)
      track 0 → self (excluded by recommend())
      track 1 → dot product = 0.0 (zero vector)
    Used to test recommend() output ordering without stochastic training.
    """
    torch.manual_seed(0)
    model = MFModel(N=5, f=3)
    model.eval()
    with torch.no_grad():
        model.P.weight.zero_()
        model.Q.weight.zero_()
        model.P.weight[0] = torch.tensor([1.0, 0.0, 0.0])
        model.Q.weight[2] = torch.tensor([1.0, 0.0, 0.0])
        model.Q.weight[3] = torch.tensor([0.5, 0.0, 0.0])
        model.Q.weight[4] = torch.tensor([0.1, 0.0, 0.0])
        # track 1 Q stays zero → dot = 0.0
    return model


# ── Integration fixtures (read real CSVs) ────────────────────────────────────

@pytest.fixture(scope="session")
def real_labels():
    """203-track pseudo_labels.csv, loaded once per test session."""
    path = os.path.join(DATA_DIR, "pseudo_labels.csv")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def real_R_tag():
    """203×203 tag overlap matrix (CSV), loaded once per test session."""
    path = os.path.join(DATA_DIR, "matrix_case1_tag_overlap.csv")
    return pd.read_csv(path, index_col="track_id")


@pytest.fixture(scope="session")
def real_R_va():
    """203×203 VA distance matrix at t=0.95 (CSV), loaded once per test session."""
    path = os.path.join(DATA_DIR, "matrix_case2_va_distance_t095.csv")
    return pd.read_csv(path, index_col="track_id")
