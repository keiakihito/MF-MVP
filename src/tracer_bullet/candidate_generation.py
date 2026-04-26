"""
MF-based candidate generation for the tracer bullet pipeline.

Role in the pipeline
--------------------
MF is the CANDIDATE GENERATOR — it retrieves a shortlist of top-K items per
user based on learned latent factors. It is NOT the final recommender.
Downstream stages (reranking.py) apply additional features and models to
re-order the candidates into a final recommendation list.

This separation reflects the two-stage RecSys architecture:
    Stage 1 — Retrieval (this module):  fast, approximate, high recall
    Stage 2 — Ranking  (reranking.py):  slower, precise, uses rich features

Reuses from src/mf/mf_experiment.py
------------------------------------
    MFModel            — the latent factor model (Koren 2009 Eq. 1)
    train_mf           — SGD training loop
    build_training_set — positive + negative sampling
    recommend          — top-K scoring via Q @ p_u

Key difference from the music MVP
----------------------------------
MovieLens is USER × ITEM (not item × item). The interaction matrix R has
shape (n_users, n_items) — rows are users, columns are movies. In the music
MVP, both axes were tracks. Here P[u] = user latent factor, Q[i] = item
latent factor, as in the original paper.

Public API
----------
build_interaction_matrix(events_df, n_users, n_items) -> np.ndarray
    Convert interaction events into a dense R matrix of shape (n_users, n_items).
    Values are the interaction strengths from the chosen signal strategy.

generate_candidates(model, user_idx, k=100) -> list[int]
    Return the top-k item indices for a given user (wraps mf.recommend).
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

# Reuse the existing MF implementation — do not rewrite
from mf.mf_experiment import MFModel, train_mf, recommend


def build_interaction_matrix(
    events_df: pd.DataFrame,
    n_users: int,
    n_items: int,
) -> np.ndarray:
    """
    Convert interaction events into a dense R matrix of shape (n_users, n_items).

    Args:
        events_df: DataFrame with columns [user_id, movie_id, interaction].
                   user_id and movie_id must be 0-indexed integers.
        n_users:   Total number of users (row dimension of R).
        n_items:   Total number of items (column dimension of R).

    Returns:
        np.ndarray of shape (n_users, n_items), dtype float32.
        R[u, i] = interaction strength from events_df, 0.0 if not present.

    TODO: implement
    """
    raise NotImplementedError


def generate_candidates(
    model: MFModel,
    user_idx: int,
    k: int = 100,
    track_ids: list | None = None,
) -> List[int]:
    """
    Return the top-k item indices for a given user using the trained MF model.

    Thin wrapper around mf.mf_experiment.recommend — exists to make the
    pipeline's dependency on MF explicit and swappable.

    Args:
        model:     Trained MFModel.
        user_idx:  0-indexed user ID.
        k:         Number of candidates to retrieve.
        track_ids: List mapping index → item ID. If None, uses integer indices.

    Returns:
        List of k item IDs (or indices if track_ids is None), ranked by score.

    TODO: implement (thin wrapper around recommend())
    """
    raise NotImplementedError
