"""
Tests for signal.py — all three signal strategies and the dispatcher.

All tests use tiny_ratings_df from conftest.py (entirely in-memory).
No MovieLens data required.

TDD: contracts defined here; signal.py raises NotImplementedError until implemented.
"""

import pytest
import pandas as pd

from tracer_bullet.signal import (
    build_interactions,
    build_threshold_signal,
    build_weighted_signal,
    build_time_decay_signal,
    train_test_split_by_time,
)

pytestmark = pytest.mark.unit

EXPECTED_COLUMNS = {"user_id", "movie_id", "interaction", "timestamp"}


# ── build_threshold_signal ────────────────────────────────────────────────────

def test_threshold_output_columns(tiny_ratings_df):
    result = build_threshold_signal(tiny_ratings_df, threshold=4)
    assert set(result.columns) == EXPECTED_COLUMNS


def test_threshold_drops_low_ratings(tiny_ratings_df):
    # Ratings < 4 must be DROPPED (not encoded as 0)
    result = build_threshold_signal(tiny_ratings_df, threshold=4)
    assert len(result) < len(tiny_ratings_df)
    # All kept rows have original rating >= 4
    kept_users_movies = set(zip(result["user_id"], result["movie_id"]))
    low_rating_pairs = set(
        zip(
            tiny_ratings_df[tiny_ratings_df["rating"] < 4]["user_id"],
            tiny_ratings_df[tiny_ratings_df["rating"] < 4]["movie_id"],
        )
    )
    assert kept_users_movies.isdisjoint(low_rating_pairs)


def test_threshold_all_interactions_are_one(tiny_ratings_df):
    result = build_threshold_signal(tiny_ratings_df, threshold=4)
    assert (result["interaction"] == 1.0).all()


def test_threshold_exact_row_count(tiny_ratings_df):
    # tiny_ratings_df has ratings [5,4,2,5,3,4,4,1] — 5 rows are >= 4
    result = build_threshold_signal(tiny_ratings_df, threshold=4)
    assert len(result) == 5


def test_threshold_custom_threshold(tiny_ratings_df):
    # With threshold=3, rows with rating >= 3 are kept (rating 2 and 1 dropped)
    result = build_threshold_signal(tiny_ratings_df, threshold=3)
    assert len(result) == len(tiny_ratings_df[tiny_ratings_df["rating"] >= 3])


# ── build_weighted_signal ─────────────────────────────────────────────────────

def test_weighted_output_columns(tiny_ratings_df):
    result = build_weighted_signal(tiny_ratings_df)
    assert set(result.columns) == EXPECTED_COLUMNS


def test_weighted_keeps_all_rows(tiny_ratings_df):
    result = build_weighted_signal(tiny_ratings_df)
    assert len(result) == len(tiny_ratings_df)


def test_weighted_interaction_range(tiny_ratings_df):
    result = build_weighted_signal(tiny_ratings_df)
    assert result["interaction"].between(0.0, 1.0).all()


def test_weighted_monotonic(tiny_ratings_df):
    # Higher rating → higher interaction value
    result = build_weighted_signal(tiny_ratings_df)
    merged = tiny_ratings_df.merge(result[["user_id", "movie_id", "interaction"]],
                                    on=["user_id", "movie_id"])
    corr = merged["rating"].corr(merged["interaction"])
    assert corr > 0.99  # near-perfect monotone relationship


def test_weighted_max_rating_maps_to_one(tiny_ratings_df):
    result = build_weighted_signal(tiny_ratings_df)
    assert result["interaction"].max() == pytest.approx(1.0)


def test_weighted_min_rating_maps_to_zero(tiny_ratings_df):
    result = build_weighted_signal(tiny_ratings_df)
    assert result["interaction"].min() == pytest.approx(0.0)


# ── build_time_decay_signal ───────────────────────────────────────────────────

def test_time_decay_output_columns(tiny_ratings_df):
    result = build_time_decay_signal(tiny_ratings_df)
    assert set(result.columns) == EXPECTED_COLUMNS


def test_time_decay_keeps_all_rows(tiny_ratings_df):
    result = build_time_decay_signal(tiny_ratings_df)
    assert len(result) == len(tiny_ratings_df)


def test_time_decay_range(tiny_ratings_df):
    result = build_time_decay_signal(tiny_ratings_df)
    # Interaction must be strictly positive and at most 1.0
    assert (result["interaction"] > 0.0).all()
    assert (result["interaction"] <= 1.0).all()


def test_time_decay_monotonic(tiny_ratings_df):
    # More recent timestamp → higher interaction weight
    result = build_time_decay_signal(tiny_ratings_df)
    merged = tiny_ratings_df.merge(result[["user_id", "movie_id", "interaction"]],
                                    on=["user_id", "movie_id"])
    corr = merged["timestamp"].corr(merged["interaction"])
    assert corr > 0.99  # recent = high weight


def test_time_decay_most_recent_is_one(tiny_ratings_df):
    result = build_time_decay_signal(tiny_ratings_df)
    assert result["interaction"].max() == pytest.approx(1.0)


# ── build_interactions dispatcher ────────────────────────────────────────────

def test_dispatcher_threshold(tiny_ratings_df):
    result = build_interactions(tiny_ratings_df, signal_type="threshold", threshold=4)
    assert set(result.columns) == EXPECTED_COLUMNS
    assert (result["interaction"] == 1.0).all()


def test_dispatcher_weighted(tiny_ratings_df):
    result = build_interactions(tiny_ratings_df, signal_type="weighted")
    assert len(result) == len(tiny_ratings_df)


def test_dispatcher_time_decay(tiny_ratings_df):
    result = build_interactions(tiny_ratings_df, signal_type="time_decay")
    assert len(result) == len(tiny_ratings_df)


def test_dispatcher_unknown_signal_type(tiny_ratings_df):
    with pytest.raises(ValueError):
        build_interactions(tiny_ratings_df, signal_type="nonexistent")


# ── train_test_split_by_time ──────────────────────────────────────────────────

def test_split_output_columns(tiny_ratings_df):
    train, test = train_test_split_by_time(tiny_ratings_df, test_ratio=0.2)
    assert "timestamp" in train.columns
    assert "timestamp" in test.columns


def test_split_no_temporal_leakage(tiny_ratings_df):
    # max timestamp in train must be < min timestamp in test
    train, test = train_test_split_by_time(tiny_ratings_df, test_ratio=0.2)
    assert train["timestamp"].max() < test["timestamp"].min()


def test_split_covers_all_rows(tiny_ratings_df):
    train, test = train_test_split_by_time(tiny_ratings_df, test_ratio=0.2)
    assert len(train) + len(test) == len(tiny_ratings_df)


def test_split_test_ratio_approximate(tiny_ratings_df):
    train, test = train_test_split_by_time(tiny_ratings_df, test_ratio=0.25)
    actual_ratio = len(test) / len(tiny_ratings_df)
    # Allow ±1 row of tolerance given integer rounding
    assert 0.1 <= actual_ratio <= 0.4
