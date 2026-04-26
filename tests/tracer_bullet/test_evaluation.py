"""
Tests for evaluation.py — Hit@K, NDCG@K, and evaluate().

All tests are pure math: no file I/O, no model training, no data dependencies.
Contracts are defined here first (TDD); evaluation.py raises NotImplementedError
until implemented.
"""

import math
import pytest

from tracer_bullet.evaluation import hit_at_k, ndcg_at_k, evaluate

pytestmark = pytest.mark.unit


# ── hit_at_k ─────────────────────────────────────────────────────────────────

def test_hit_at_k_found_within_cutoff():
    # Relevant item is at rank 2, k=3 → hit
    assert hit_at_k([10, 20, 30], {20}, k=3) == 1.0


def test_hit_at_k_not_found_within_cutoff():
    # Relevant item is at rank 4, k=3 → miss
    assert hit_at_k([10, 20, 30, 99], {99}, k=3) == 0.0


def test_hit_at_k_found_at_exactly_k():
    # Relevant item is at rank k → hit
    assert hit_at_k([10, 20, 99], {99}, k=3) == 1.0


def test_hit_at_k_empty_recommended():
    assert hit_at_k([], {10}, k=5) == 0.0


def test_hit_at_k_empty_relevant():
    # Nothing is relevant → always miss
    assert hit_at_k([10, 20, 30], set(), k=3) == 0.0


def test_hit_at_k_multiple_relevant_one_found():
    # Two relevant items; one is in top-k
    assert hit_at_k([10, 20, 30], {20, 99}, k=3) == 1.0


# ── ndcg_at_k ────────────────────────────────────────────────────────────────

def test_ndcg_perfect_ranking():
    # Relevant item at rank 1 → perfect score = 1.0
    assert ndcg_at_k([10, 20, 30], {10}, k=3) == pytest.approx(1.0)


def test_ndcg_relevant_at_last_rank():
    # Relevant item at rank k → low but nonzero score
    score = ndcg_at_k([10, 20, 99], {99}, k=3)
    assert 0.0 < score < 1.0


def test_ndcg_relevant_earlier_scores_higher():
    # Same item, different rank → earlier rank scores higher
    score_rank1 = ndcg_at_k([99, 10, 20], {99}, k=3)
    score_rank3 = ndcg_at_k([10, 20, 99], {99}, k=3)
    assert score_rank1 > score_rank3


def test_ndcg_not_found():
    assert ndcg_at_k([10, 20, 30], {99}, k=3) == pytest.approx(0.0)


def test_ndcg_empty_relevant():
    assert ndcg_at_k([10, 20, 30], set(), k=3) == pytest.approx(0.0)


def test_ndcg_items_beyond_k_ignored():
    # Item at rank 4 should not affect ndcg@3
    score_without = ndcg_at_k([10, 20, 30], {99}, k=3)
    score_with    = ndcg_at_k([10, 20, 30, 99], {99}, k=3)
    assert score_without == pytest.approx(score_with)


# ── evaluate ─────────────────────────────────────────────────────────────────

def test_evaluate_returns_four_keys(tiny_recs, tiny_ground_truth):
    result = evaluate(tiny_recs, tiny_ground_truth, k=3)
    assert set(result.keys()) == {"hit@k_mean", "hit@k_std", "ndcg@k_mean", "ndcg@k_std"}


def test_evaluate_mean_correct(tiny_recs, tiny_ground_truth):
    # user 1: hit@3=1 (item 20 at rank 2)
    # user 2: hit@3=0 (item 99 not in list)
    # user 3: hit@3=1 (item 99 at rank 1)
    # mean = (1+0+1)/3 = 0.667
    result = evaluate(tiny_recs, tiny_ground_truth, k=3)
    assert result["hit@k_mean"] == pytest.approx(2 / 3, abs=1e-6)


def test_evaluate_std_zero_when_uniform():
    # All users get identical hit@k → std = 0
    recs = {1: [10], 2: [10], 3: [10]}
    gt   = {1: {10}, 2: {10}, 3: {10}}
    result = evaluate(recs, gt, k=1)
    assert result["hit@k_std"] == pytest.approx(0.0, abs=1e-6)


def test_evaluate_std_nonzero_when_mixed(tiny_recs, tiny_ground_truth):
    # Users have different hit scores → std > 0
    result = evaluate(tiny_recs, tiny_ground_truth, k=3)
    assert result["hit@k_std"] > 0.0


def test_evaluate_single_user():
    recs = {1: [10, 20]}
    gt   = {1: {10}}
    result = evaluate(recs, gt, k=2)
    assert result["hit@k_mean"] == pytest.approx(1.0)
    assert result["hit@k_std"]  == pytest.approx(0.0)
