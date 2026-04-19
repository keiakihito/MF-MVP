"""
Interaction matrix builders.

Each builder takes the pseudo_labels DataFrame and returns a square
DataFrame indexed and columned by track_id, with float values in [0, 1].
"""

import numpy as np
import pandas as pd

CHARACTER_TAGS = ["energetic", "tense", "calm", "lyrical"]


def build_tag_overlap_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Case 1 (baseline): binary overlap.
    interaction[i][j] = 1 if tracks share at least one active character tag.
    """
    tag_matrix = df[CHARACTER_TAGS].values.astype(np.float32)  # (n, 4)
    # dot product > 0 means at least one shared tag
    overlap = tag_matrix @ tag_matrix.T
    binary = (overlap > 0).astype(np.float32)
    return _wrap(binary, df["track_id"])


def build_va_distance_matrix(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """
    Case 2: VA-distance-based soft similarity, thresholded to binary.

    similarity(i,j) = 1 - euclidean_distance(VA_i, VA_j) / sqrt(2)
    interaction[i][j] = 1 if similarity >= threshold, else 0.

    sqrt(2) is the max possible distance in the unit [0,1]^2 VA space.
    """
    va = df[["valence", "arousal"]].values.astype(np.float32)  # (n, 2)
    # pairwise squared distances via broadcast
    diff = va[:, np.newaxis, :] - va[np.newaxis, :, :]         # (n, n, 2)
    dist = np.sqrt((diff ** 2).sum(axis=2))                     # (n, n)
    similarity = 1.0 - dist / np.sqrt(2)
    binary = (similarity >= threshold).astype(np.float32)
    return _wrap(binary, df["track_id"])


def _wrap(matrix: np.ndarray, track_ids: pd.Series) -> pd.DataFrame:
    ids = track_ids.tolist()
    df_out = pd.DataFrame(matrix, index=ids, columns=ids)
    df_out.index.name = "track_id"
    df_out.columns.name = "track_id"
    return df_out
