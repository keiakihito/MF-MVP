"""
Tests for MFModel, build_training_set, train_mf, recommend, and _get_cv_entries.

RESEARCH CONCEPT: Implements Koren et al. (2009) Eq. 1:
    r̂_ui = q_i^T p_u
The model learns latent vectors P (query tracks) and Q (candidate tracks).
Training minimizes MSE over observed interactions κ plus sampled negatives
(implicit feedback framing: unobserved = no preference target).

This is an item-to-item setup: tracks act as both "users" and "items",
so P and Q both have shape (N, f) — unlike classic user/item MF.
"""

import pytest
import numpy as np
import torch

from mf.mf_experiment import (
    MFModel,
    build_training_set,
    train_mf,
    recommend,
    _get_cv_entries,
)

pytestmark = pytest.mark.unit


# ── MFModel — shape and initialization ───────────────────────────────────────

def test_mfmodel_embedding_shapes(small_model):
    # N=10, f=4 → both P and Q are (10, 4)
    # Both index the same N tracks (item-to-item symmetry)
    assert small_model.P.weight.shape == (10, 4)
    assert small_model.Q.weight.shape == (10, 4)


def test_mfmodel_is_nn_module(small_model):
    assert isinstance(small_model, torch.nn.Module)


# ── MFModel.forward — Eq. 1: r̂_ui = q_i^T p_u ────────────────────────────

def test_mfmodel_forward_output_shape(small_model):
    u = torch.tensor([0, 1, 2])
    i = torch.tensor([3, 4, 5])
    out = small_model(u, i)
    assert out.shape == (3,)


def test_mfmodel_forward_output_dtype(small_model):
    u = torch.tensor([0])
    i = torch.tensor([1])
    out = small_model(u, i)
    assert out.dtype == torch.float32


def test_mfmodel_forward_is_dot_product_positive():
    # P[0] = [1,0,0], Q[1] = [1,0,0] → dot = 1.0
    model = MFModel(N=3, f=3)
    with torch.no_grad():
        model.P.weight.zero_()
        model.Q.weight.zero_()
        model.P.weight[0] = torch.tensor([1.0, 0.0, 0.0])
        model.Q.weight[1] = torch.tensor([1.0, 0.0, 0.0])
    out = model(torch.tensor([0]), torch.tensor([1]))
    assert out.item() == pytest.approx(1.0)


def test_mfmodel_forward_is_dot_product_zero():
    # P[0] = [1,0,0], Q[1] = [0,1,0] → dot = 0.0 (orthogonal)
    model = MFModel(N=3, f=3)
    with torch.no_grad():
        model.P.weight.zero_()
        model.Q.weight.zero_()
        model.P.weight[0] = torch.tensor([1.0, 0.0, 0.0])
        model.Q.weight[1] = torch.tensor([0.0, 1.0, 0.0])
    out = model(torch.tensor([0]), torch.tensor([1]))
    assert out.item() == pytest.approx(0.0, abs=1e-6)


def test_mfmodel_forward_general_dot_product():
    # P[0] = [2,3], Q[1] = [4,5] → dot = 2*4 + 3*5 = 23
    model = MFModel(N=2, f=2)
    with torch.no_grad():
        model.P.weight[0] = torch.tensor([2.0, 3.0])
        model.Q.weight[1] = torch.tensor([4.0, 5.0])
    out = model(torch.tensor([0]), torch.tensor([1]))
    assert out.item() == pytest.approx(23.0)


# ── MFModel.predict_scores ────────────────────────────────────────────────────

def test_mfmodel_predict_scores_shape(small_model):
    # predict_scores(u) returns r̂_ui for all N items
    scores = small_model.predict_scores(0)
    assert scores.shape == (10,)


def test_mfmodel_predict_scores_equals_forward_for_all_items(small_model):
    # Batch predict_scores must agree with individual forward calls
    u_idx = 2
    scores = small_model.predict_scores(u_idx)
    for i in range(10):
        single = small_model(torch.tensor([u_idx]), torch.tensor([i]))
        assert scores[i].item() == pytest.approx(single.item(), abs=1e-5)


# ── build_training_set ────────────────────────────────────────────────────────

def test_build_training_set_returns_three_arrays(tiny_R_block):
    result = build_training_set(tiny_R_block)
    assert len(result) == 3


def test_build_training_set_dtypes(tiny_R_block):
    u_idx, i_idx, r_ui = build_training_set(tiny_R_block)
    assert u_idx.dtype == np.int64
    assert i_idx.dtype == np.int64
    assert r_ui.dtype == np.float32


def test_build_training_set_target_values_are_binary(tiny_R_block):
    _, _, r_ui = build_training_set(tiny_R_block)
    assert set(r_ui.tolist()) == {0.0, 1.0}


def test_build_training_set_sizes_confidence_ratio_one(tiny_R_block):
    # confidence_ratio=1.0 → #negatives = #positives → total = 2 * n_positives
    n_pos = int(tiny_R_block.sum())
    u_idx, _, _ = build_training_set(tiny_R_block, confidence_ratio=1.0)
    assert len(u_idx) == 2 * n_pos


def test_build_training_set_sizes_confidence_ratio_half(tiny_R_block):
    # confidence_ratio=0.5 → #negatives = 0.5 * #positives → total = 1.5 * n_positives
    n_pos = int(tiny_R_block.sum())
    u_idx, _, _ = build_training_set(tiny_R_block, confidence_ratio=0.5)
    assert len(u_idx) == int(n_pos * 1.5)


def test_build_training_set_positives_match_matrix(tiny_R_block):
    u_idx, i_idx, r_ui = build_training_set(tiny_R_block)
    pos_mask = r_ui == 1.0
    for u, i in zip(u_idx[pos_mask], i_idx[pos_mask]):
        assert tiny_R_block[u, i] == 1.0, f"Positive pair ({u},{i}) not in R"


def test_build_training_set_negatives_are_from_zero_entries(tiny_R_block):
    u_idx, i_idx, r_ui = build_training_set(tiny_R_block)
    neg_mask = r_ui == 0.0
    for u, i in zip(u_idx[neg_mask], i_idx[neg_mask]):
        assert tiny_R_block[u, i] == 0.0, f"Negative pair ({u},{i}) was actually R=1"


def test_build_training_set_lengths_consistent(tiny_R_block):
    u_idx, i_idx, r_ui = build_training_set(tiny_R_block)
    assert len(u_idx) == len(i_idx) == len(r_ui)


# ── train_mf ──────────────────────────────────────────────────────────────────

def test_train_mf_returns_model_and_history(tiny_R_block):
    model, history = train_mf(tiny_R_block, f=4, epochs=3, batch_size=4)
    assert isinstance(model, MFModel)
    assert isinstance(history, list)


def test_train_mf_history_length_equals_epochs(tiny_R_block):
    _, history = train_mf(tiny_R_block, f=4, epochs=5, batch_size=4)
    assert len(history) == 5


def test_train_mf_model_is_in_eval_mode_after_training(tiny_R_block):
    # train_mf calls model.eval() at the end — important for inference
    model, _ = train_mf(tiny_R_block, f=4, epochs=3, batch_size=4)
    assert model.training is False


def test_train_mf_loss_values_are_finite(tiny_R_block):
    _, history = train_mf(tiny_R_block, f=4, epochs=5, batch_size=4)
    for loss in history:
        assert np.isfinite(loss)


def test_train_mf_loss_decreases_on_learnable_signal(tiny_R_block):
    # A clear block-diagonal signal should be learnable; loss must decrease
    _, history = train_mf(tiny_R_block, f=8, epochs=30, lr=0.05, batch_size=4)
    assert history[-1] < history[0], (
        f"Loss did not decrease: start={history[0]:.4f}, end={history[-1]:.4f}"
    )


# ── recommend ─────────────────────────────────────────────────────────────────

def test_recommend_returns_k_items(deterministic_model):
    result = recommend(deterministic_model, 0, list(range(5)), k=3)
    assert len(result) == 3


def test_recommend_excludes_self(deterministic_model):
    # The query track (index 0) must never appear in its own recommendations
    result = recommend(deterministic_model, 0, list(range(5)), k=4)
    assert 0 not in result


def test_recommend_sorted_by_score(deterministic_model):
    # Weights set in fixture: Q[2]=1.0, Q[3]=0.5, Q[4]=0.1 for P[0]=[1,0,0]
    # Expected order: track 2 first, then 3, then 4
    result = recommend(deterministic_model, 0, list(range(5)), k=3)
    assert result == [2, 3, 4]


def test_recommend_returns_original_track_ids():
    # recommend must map back to original IDs, not array indices
    model = MFModel(N=4, f=2)
    with torch.no_grad():
        model.P.weight.zero_()
        model.Q.weight.zero_()
        model.P.weight[0] = torch.tensor([1.0, 0.0])
        model.Q.weight[2] = torch.tensor([1.0, 0.0])
    track_ids = ["alpha", "beta", "gamma", "delta"]
    result = recommend(model, 0, track_ids, k=2)
    for tid in result:
        assert tid in track_ids


def test_recommend_k_larger_than_n_minus_one_clips(small_model):
    # N=10, requesting k=9 (max possible excluding self): should return 9 items
    result = recommend(small_model, 0, list(range(10)), k=9)
    assert len(result) == 9


# ── _get_cv_entries ───────────────────────────────────────────────────────────

def test_get_cv_entries_returns_two_arrays(tiny_R_block):
    rows, cols = _get_cv_entries(tiny_R_block)
    assert isinstance(rows, np.ndarray)
    assert isinstance(cols, np.ndarray)
    assert len(rows) == len(cols)


def test_get_cv_entries_upper_triangle_only(tiny_R_block):
    # All returned pairs must satisfy row < col (strict upper triangle)
    rows, cols = _get_cv_entries(tiny_R_block)
    assert np.all(rows < cols), "Some CV entries are not in the strict upper triangle"


def test_get_cv_entries_no_diagonal(tiny_R_block):
    # Diagonal is excluded via np.triu(R, k=1) — k=1 skips the diagonal
    rows, cols = _get_cv_entries(tiny_R_block)
    assert np.all(rows != cols)


def test_get_cv_entries_only_positive_entries(tiny_R_block):
    # CV entries must all be from observed interactions (R[r,c] == 1)
    rows, cols = _get_cv_entries(tiny_R_block)
    for r, c in zip(rows, cols):
        assert tiny_R_block[r, c] == 1.0


def test_get_cv_entries_count_matches_upper_triangle_positives(tiny_R_block):
    # tiny_R_block has 2 positive pairs in upper triangle: (0,1) and (2,3)
    rows, cols = _get_cv_entries(tiny_R_block)
    assert len(rows) == 2
