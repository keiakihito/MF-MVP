"""
LightGBM-based re-ranker for the tracer bullet pipeline.

Role in the pipeline
--------------------
The re-ranker is Stage 2 of the two-stage RecSys architecture. It takes the
shortlist produced by candidate_generation.py and re-orders it using richer
features than MF alone can access.

Planned features
----------------
User features  (from users.dat):
    gender, age, occupation

Item features  (from movies.dat):
    genre one-hot encoding

Interaction features:
    MF score from candidate_generation (the latent factor dot product)
    user history length (number of rated items)

Dependency
----------
    pip install lightgbm

This dependency is deferred until reranking is implemented. Do not add it to
requirements.txt yet.

Public API
----------
build_feature_matrix(candidates, user_df, movie_df, mf_scores) -> np.ndarray
    Construct a feature matrix X where each row is a (user, item) candidate pair.

train_ranker(X_train, y_train) -> ranker
    Train a LightGBM ranker on labeled (user, item) pairs.
    y_train is a binary relevance label (1 = relevant in ground truth).

rerank(ranker, candidates, X) -> list[int]
    Apply a trained ranker to re-order a list of candidate item IDs.
    Returns item IDs sorted by predicted relevance score (highest first).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def build_feature_matrix(
    candidates: List[int],
    user_id: int,
    user_df: pd.DataFrame,
    movie_df: pd.DataFrame,
    mf_scores: np.ndarray,
) -> np.ndarray:
    """
    Construct a feature matrix for a set of candidate items for one user.

    Args:
        candidates: List of candidate item IDs (from candidate_generation).
        user_id:    The query user ID.
        user_df:    DataFrame with user features (from dataset.load_users).
        movie_df:   DataFrame with movie features (from dataset.load_movies).
        mf_scores:  Array of MF scores for each candidate (same order as candidates).

    Returns:
        np.ndarray of shape (len(candidates), n_features), dtype float32.

    TODO: implement after dataset and candidate_generation are complete
    """
    raise NotImplementedError


def train_ranker(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Any:
    """
    Train a LightGBM ranker on labeled (user, item) feature rows.

    Args:
        X_train: Feature matrix of shape (n_samples, n_features).
        y_train: Binary relevance labels of shape (n_samples,).

    Returns:
        Trained LightGBM model object.

    TODO: implement (requires: pip install lightgbm)
    """
    raise NotImplementedError


def rerank(
    ranker: Any,
    candidates: List[int],
    X: np.ndarray,
) -> List[int]:
    """
    Re-order candidate items by predicted relevance score.

    Args:
        ranker:     Trained LightGBM ranker (from train_ranker).
        candidates: List of candidate item IDs (same order as rows of X).
        X:          Feature matrix of shape (len(candidates), n_features).

    Returns:
        candidates re-ordered by predicted relevance (highest first).

    TODO: implement
    """
    raise NotImplementedError
