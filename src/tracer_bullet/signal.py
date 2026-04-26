"""
Modular, swappable interaction signal construction strategies.

This is the core experimental module. Different signal definitions produce
different interaction DataFrames, which downstream build different R matrices,
MF embeddings, and recommendation behaviors.

Mirrors the research design in the music MVP (tag overlap vs. VA distance),
now applied to MovieLens 1M ratings data.

Paper connection (Koren 2009 §"Additional Input Sources"):
    "implicit feedback usually denotes the presence or absence of an event"
    Each strategy is a different definition of what counts as an "event".

Signal strategies
-----------------
1. threshold  — rating >= threshold → interaction = 1.
               Rows below threshold are DROPPED (not encoded as 0).
               Negative sampling is handled separately, not here.

2. weighted   — normalized raw rating → interaction in [0, 1].
               All rows kept; strength reflects rating magnitude.

3. time_decay — interaction weight decays exponentially with age.
               More recent interactions are weighted closer to 1.0.

Public API
----------
build_interactions(ratings_df, signal_type="threshold", **kwargs) -> pd.DataFrame
    Dispatcher: routes to the appropriate build_*_signal function.
    Output columns: [user_id, movie_id, interaction, timestamp]

build_threshold_signal(ratings_df, threshold=4) -> pd.DataFrame
build_weighted_signal(ratings_df) -> pd.DataFrame
build_time_decay_signal(ratings_df, decay_rate=0.001) -> pd.DataFrame

train_test_split_by_time(events_df, test_ratio=0.2) -> tuple[pd.DataFrame, pd.DataFrame]
    Temporal split: all events up to the (1-test_ratio) quantile of timestamps
    go to train; the rest go to test. No random shuffling — preserves causality.
"""

from __future__ import annotations

import pandas as pd


def build_interactions(
    ratings_df: pd.DataFrame,
    signal_type: str = "threshold",
    **kwargs,
) -> pd.DataFrame:
    """
    Dispatcher for signal construction strategies.

    Args:
        ratings_df:  DataFrame with columns [user_id, movie_id, rating, timestamp].
        signal_type: One of "threshold", "weighted", "time_decay".
        **kwargs:    Passed through to the chosen strategy function.

    Returns:
        DataFrame with columns [user_id, movie_id, interaction, timestamp].

    Raises:
        ValueError: if signal_type is not recognized.
    """
    if signal_type == "threshold":
        return build_threshold_signal(ratings_df, **kwargs)
    elif signal_type == "weighted":
        return build_weighted_signal(ratings_df, **kwargs)
    elif signal_type == "time_decay":
        return build_time_decay_signal(ratings_df, **kwargs)
    else:
        raise ValueError(f"Unknown signal_type: {signal_type!r}. Choose from 'threshold', 'weighted', 'time_decay'.")


def build_threshold_signal(
    ratings_df: pd.DataFrame,
    threshold: float = 4.0,
) -> pd.DataFrame:
    """
    Keep only ratings >= threshold; set interaction = 1 for all kept rows.

    Rows below threshold are DROPPED entirely — they are treated as unobserved,
    not as negative interactions. Negative sampling is handled downstream.

    Args:
        ratings_df: DataFrame with columns [user_id, movie_id, rating, timestamp].
        threshold:  Minimum rating to count as a positive interaction.

    Returns:
        DataFrame with columns [user_id, movie_id, interaction, timestamp]
        where interaction = 1.0 for all rows.
    """
    kept = ratings_df[ratings_df["rating"] >= threshold].copy()
    kept["interaction"] = 1.0
    return kept[["user_id", "movie_id", "interaction", "timestamp"]].reset_index(drop=True)


def build_weighted_signal(
    ratings_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize raw ratings to [0, 1] and use as soft interaction strength.

    Normalization: interaction = (rating - min_rating) / (max_rating - min_rating)
    All rows are kept. A rating of 1 → interaction ≈ 0.0; rating of 5 → 1.0.

    Args:
        ratings_df: DataFrame with columns [user_id, movie_id, rating, timestamp].

    Returns:
        DataFrame with columns [user_id, movie_id, interaction, timestamp]
        where interaction ∈ [0.0, 1.0].
    """
    result = ratings_df.copy()
    min_r = result["rating"].min()
    max_r = result["rating"].max()
    denom = max_r - min_r
    result["interaction"] = (result["rating"] - min_r) / denom if denom > 0 else 0.0
    return result[["user_id", "movie_id", "interaction", "timestamp"]].reset_index(drop=True)


def build_time_decay_signal(
    ratings_df: pd.DataFrame,
    decay_rate: float = 0.001,
) -> pd.DataFrame:
    """
    Weight interactions by recency: newer events are closer to 1.0.

    Decay formula: interaction = exp(-decay_rate * days_since_event)
    where days_since_event = (max_timestamp - timestamp) / 86400.

    All rows are kept. A very recent event → interaction ≈ 1.0;
    an event from long ago → interaction → 0.0.

    Args:
        ratings_df:  DataFrame with columns [user_id, movie_id, rating, timestamp].
        decay_rate:  Controls how fast weight drops with age (per day).

    Returns:
        DataFrame with columns [user_id, movie_id, interaction, timestamp]
        where interaction ∈ (0.0, 1.0].
    """
    import math
    result = ratings_df.copy()
    max_ts = result["timestamp"].max()
    days_since = (max_ts - result["timestamp"]) / 86400.0
    result["interaction"] = days_since.apply(lambda d: math.exp(-decay_rate * d))
    return result[["user_id", "movie_id", "interaction", "timestamp"]].reset_index(drop=True)


def train_test_split_by_time(
    events_df: pd.DataFrame,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal train/test split — no random shuffling.

    All events whose timestamp is at or below the (1 - test_ratio) quantile
    go to train. Events above that quantile go to test.
    This ensures the model never sees future interactions during training.

    Args:
        events_df:  DataFrame with a `timestamp` column.
        test_ratio: Fraction of events (by time) reserved for test.

    Returns:
        (train_df, test_df)
    """
    cutoff = events_df["timestamp"].quantile(1.0 - test_ratio)
    train = events_df[events_df["timestamp"] <= cutoff].reset_index(drop=True)
    test  = events_df[events_df["timestamp"] >  cutoff].reset_index(drop=True)
    return train, test
