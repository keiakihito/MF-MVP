"""
conftest.py — Shared fixtures for the tracer_bullet test suite.

All fixtures are entirely in-memory (no file I/O, no MovieLens download required).
They use synthetic data designed to produce predictable, easy-to-reason-about results.

Fixture taxonomy:
  tiny_ratings_df        — 5 users × 4 movies, ratings 1–5 with ascending timestamps
  tiny_threshold_events  — result of build_threshold_signal(tiny_ratings_df, threshold=4)
  tiny_weighted_events   — result of build_weighted_signal(tiny_ratings_df)
  tiny_recs              — {user_id: [movie_id, ...]} for evaluation tests
  tiny_ground_truth      — {user_id: {movie_id, ...}} for evaluation tests
"""

import pytest
import pandas as pd


@pytest.fixture
def tiny_ratings_df():
    """
    5 users × 4 movies, ratings 1–5, timestamps strictly ascending.

    Designed so that:
    - Ratings >= 4: (u=1,m=1), (u=1,m=2), (u=2,m=3), (u=3,m=1), (u=4,m=4)
    - Ratings <  4: (u=2,m=1,r=2), (u=3,m=2,r=3), (u=5,m=3,r=1)
    This makes threshold filtering predictable in tests.
    """
    return pd.DataFrame({
        "user_id":   [1, 1, 2, 2, 3, 3, 4, 5],
        "movie_id":  [1, 2, 1, 3, 1, 2, 4, 3],
        "rating":    [5, 4, 2, 5, 3, 4, 4, 1],
        "timestamp": [100, 200, 300, 400, 500, 600, 700, 800],
    })


@pytest.fixture
def tiny_threshold_events(tiny_ratings_df):
    """
    Threshold signal (threshold=4) applied to tiny_ratings_df.
    Only rows with rating >= 4 are kept; interaction = 1.0 for all rows.
    Expected kept rows: indices 0,1,3,5,6 → 5 rows.
    """
    from tracer_bullet.signal import build_threshold_signal
    return build_threshold_signal(tiny_ratings_df, threshold=4)


@pytest.fixture
def tiny_weighted_events(tiny_ratings_df):
    """
    Weighted signal applied to tiny_ratings_df.
    All 8 rows kept; interaction normalized to [0, 1].
    """
    from tracer_bullet.signal import build_weighted_signal
    return build_weighted_signal(tiny_ratings_df)


@pytest.fixture
def tiny_recs():
    """
    Synthetic recommendation lists for 3 users.
    Designed to make hit@k and ndcg@k tests deterministic.
    """
    return {
        1: [10, 20, 30, 40, 50],   # item 20 is relevant → hit@3=1, hit@1=0
        2: [10, 20, 30],           # no relevant item → hit@3=0
        3: [99, 10, 20],           # item 99 is relevant → hit@3=1, hit@1=1
    }


@pytest.fixture
def tiny_ground_truth():
    """
    Ground truth relevant items per user, paired with tiny_recs.
    """
    return {
        1: {20},         # item 20 is in recs at rank 2
        2: {99},         # item 99 is not in recs at all
        3: {99},         # item 99 is in recs at rank 1
    }
