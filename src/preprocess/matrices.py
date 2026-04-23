"""
Interaction matrix builders.

In the paper (Koren et al. 2009), the input is an explicit ratings matrix r_ui.
Here we have no real user ratings, so we construct a proxy "implicit feedback"
matrix R from content features (§"Additional Input Sources"). Each entry
R[i,j] = 1 means track i and track j are considered "similar" under the
chosen signal definition — standing in for an observed interaction.

Two signal definitions:
  Case 1 — Tag Overlap:  R[i,j] = 1 if tracks share ≥1 character tag
  Case 2 — VA Distance:  R[i,j] = 1 if valence-arousal similarity ≥ threshold
"""

import numpy as np
import pandas as pd

CHARACTER_TAGS = ["energetic", "tense", "calm", "lyrical"]


# ── Case 1: Tag Overlap ───────────────────────────────────────────────────────

def _extract_tag_vectors(df: pd.DataFrame) -> np.ndarray:
    """Return binary tag matrix of shape (N, 4); columns = CHARACTER_TAGS."""
    return df[CHARACTER_TAGS].values.astype(np.float32)


def _tag_overlap_to_binary(tag_matrix: np.ndarray) -> np.ndarray:
    """
    R[i,j] = 1 if tracks i and j share at least one active tag, else 0.
    Dot product > 0 ↔ at least one shared dimension.
    """
    overlap = tag_matrix @ tag_matrix.T   # (N, N)
    return (overlap > 0).astype(np.float32)


def build_tag_overlap_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Case 1: binary implicit feedback from shared character tags."""
    tag_matrix = _extract_tag_vectors(df)
    binary     = _tag_overlap_to_binary(tag_matrix)
    return _wrap(binary, df["track_id"])


# ── Case 2: VA Distance ───────────────────────────────────────────────────────

def _extract_va_vectors(df: pd.DataFrame) -> np.ndarray:
    """Return valence-arousal matrix of shape (N, 2)."""
    return df[["valence", "arousal"]].values.astype(np.float32)


def _pairwise_va_similarity(va: np.ndarray) -> np.ndarray:
    """
    similarity(i,j) = 1 − euclidean_dist(VA_i, VA_j) / √2

    √2 is the max possible distance in the unit [0,1]^2 VA space,
    so similarity is normalized to [0, 1].

　　ex: diff = va[:, np.newaxis, :] - va[np.newaxis, :, :] 
    j →
        A     B     C

i  A  A-A   A-B   A-C
↓  B  B-A   B-B   B-C
   C  C-A   C-B   C-C

   ① va[:, np.newaxis, :]

    👉 shape: (N, 1, 2)

    [
    [[A]],
    [[B]],
    [[C]]
    ]

    ② va[np.newaxis, :, :]

    👉 shape: (1, N, 2)

    [
    [A, B, C]
    ]

    template
    X[:, None, :]  # ← row (i)
    X[None, :, :]  # ← column (j)
    """
    diff = va[:, np.newaxis, :] - va[np.newaxis, :, :]   # (N, N, 2)
    dist = np.sqrt((diff ** 2).sum(axis=2))               # (N, N)
    return 1.0 - dist / np.sqrt(2)


def _threshold_to_binary(similarity: np.ndarray, threshold: float) -> np.ndarray:
    """
    Binarize: R[i,j] = 1 if similarity ≥ threshold.
    Analogous to the confidence-level idea in the paper (§"Inputs with Varying
    Confidence Levels"): high-similarity pairs are treated as confident interactions.
    """
    return (similarity >= threshold).astype(np.float32)


def build_va_distance_matrix(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """Case 2: binary implicit feedback from valence-arousal proximity."""
    va         = _extract_va_vectors(df)
    similarity = _pairwise_va_similarity(va)
    binary     = _threshold_to_binary(similarity, threshold)
    return _wrap(binary, df["track_id"])


# ── Shared helper ─────────────────────────────────────────────────────────────

def _wrap(matrix: np.ndarray, track_ids: pd.Series) -> pd.DataFrame:
    ids = track_ids.tolist()
    df_out = pd.DataFrame(matrix, index=ids, columns=ids)
    df_out.index.name   = "track_id"
    df_out.columns.name = "track_id"
    return df_out
