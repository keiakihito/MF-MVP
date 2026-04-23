"""
Integration tests: train MF on the real 203-track matrices.

RESEARCH CONCEPT VALIDATION:
  - MF model learns from both Case 1 and Case 2 interaction signals
  - Loss decreases over training (SGD convergence on real data)
  - Recommendations are valid: correct number, no self, from known track IDs
  - K-fold CV produces finite, bounded MSE values (0 < MSE < 1)
  - (Soft) Case 1 MSE tends to be lower than Case 2 (block signal is easier to fit)
"""

import pytest
import numpy as np

from mf.mf_experiment import MFModel, train_mf, recommend, kfold_cv

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def trained_tag_model(real_R_tag):
    """Train a fast MF model on Case 1 matrix; shared across tests in this module."""
    R = real_R_tag.values.astype(np.float32)
    model, loss_history = train_mf(R, f=8, epochs=20, lr=0.01, desc="[test] Case 1")
    return model, loss_history, R


@pytest.fixture(scope="module")
def trained_va_model(real_R_va):
    """Train a fast MF model on Case 2 matrix; shared across tests in this module."""
    R = real_R_va.values.astype(np.float32)
    model, loss_history = train_mf(R, f=8, epochs=20, lr=0.01, desc="[test] Case 2")
    return model, loss_history, R


# ── Training convergence ──────────────────────────────────────────────────────

def test_train_mf_tag_loss_decreases(trained_tag_model):
    _, history, _ = trained_tag_model
    assert history[-1] < history[0], (
        f"Case 1 loss did not decrease: start={history[0]:.4f}, end={history[-1]:.4f}"
    )


def test_train_mf_va_loss_decreases(trained_va_model):
    _, history, _ = trained_va_model
    assert history[-1] < history[0], (
        f"Case 2 loss did not decrease: start={history[0]:.4f}, end={history[-1]:.4f}"
    )


def test_train_mf_loss_history_length(trained_tag_model):
    _, history, _ = trained_tag_model
    assert len(history) == 20


def test_train_mf_embedding_shapes_after_training(trained_tag_model):
    model, _, _ = trained_tag_model
    # N=203, f=8
    assert model.P.weight.shape == (203, 8)
    assert model.Q.weight.shape == (203, 8)


# ── Recommendation pipeline ───────────────────────────────────────────────────

def test_recommend_returns_k_items(trained_tag_model, real_R_tag):
    model, _, _ = trained_tag_model
    track_ids = real_R_tag.index.tolist()
    result = recommend(model, 0, track_ids, k=10)
    assert len(result) == 10


def test_recommend_all_ids_are_valid(trained_tag_model, real_R_tag):
    model, _, _ = trained_tag_model
    track_ids = real_R_tag.index.tolist()
    result = recommend(model, 0, track_ids, k=10)
    for tid in result:
        assert tid in track_ids, f"Returned track_id {tid!r} not in known track_ids"


def test_recommend_excludes_self_for_query_zero(trained_tag_model, real_R_tag):
    model, _, _ = trained_tag_model
    track_ids = real_R_tag.index.tolist()
    result = recommend(model, 0, track_ids, k=10)
    assert track_ids[0] not in result


def test_recommend_excludes_self_for_multiple_queries(trained_tag_model, real_R_tag):
    model, _, _ = trained_tag_model
    track_ids = real_R_tag.index.tolist()
    for q_idx in [0, 50, 100, 150, 202]:
        result = recommend(model, q_idx, track_ids, k=10)
        assert track_ids[q_idx] not in result, (
            f"Self-recommendation found for query index {q_idx}"
        )


# ── K-Fold Cross-Validation ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cv_tag_result(real_R_tag):
    R = real_R_tag.values.astype(np.float32)
    return kfold_cv(R, f=4, epochs=5, lr=0.01, n_splits=3, desc="[test] CV Tag")


@pytest.fixture(scope="module")
def cv_va_result(real_R_va):
    R = real_R_va.values.astype(np.float32)
    return kfold_cv(R, f=4, epochs=5, lr=0.01, n_splits=3, desc="[test] CV VA")


def test_kfold_cv_returns_dict(cv_tag_result):
    assert isinstance(cv_tag_result, dict)


def test_kfold_cv_required_keys(cv_tag_result):
    assert set(cv_tag_result.keys()) == {"mse_per_fold", "mean", "std"}


def test_kfold_cv_fold_count(cv_tag_result):
    assert len(cv_tag_result["mse_per_fold"]) == 3


def test_kfold_cv_mean_is_correct(cv_tag_result):
    expected = float(np.mean(cv_tag_result["mse_per_fold"]))
    assert cv_tag_result["mean"] == pytest.approx(expected, abs=1e-5)


def test_kfold_cv_std_is_correct(cv_tag_result):
    expected = float(np.std(cv_tag_result["mse_per_fold"]))
    assert cv_tag_result["std"] == pytest.approx(expected, abs=1e-5)


def test_kfold_cv_mse_values_finite_and_positive(cv_tag_result):
    for mse in cv_tag_result["mse_per_fold"]:
        assert 0.0 < mse < 1.0, f"MSE value {mse:.4f} outside expected range (0, 1)"


def test_kfold_cv_va_mse_values_finite_and_positive(cv_va_result):
    for mse in cv_va_result["mse_per_fold"]:
        assert 0.0 < mse < 1.0, f"MSE value {mse:.4f} outside expected range (0, 1)"


def test_mf_case1_lower_mse_than_case2_soft(cv_tag_result, cv_va_result):
    # RESEARCH HYPOTHESIS: Case 1 (tag overlap) produces a more consistently
    # learnable signal than Case 2 (VA distance), reflected in lower CV-MSE.
    # A tolerance of 0.05 absorbs stochasticity from the short training run.
    tag_mean = cv_tag_result["mean"]
    va_mean  = cv_va_result["mean"]
    assert tag_mean <= va_mean + 0.05, (
        f"Case 1 MSE ({tag_mean:.4f}) should be ≤ Case 2 MSE ({va_mean:.4f}) + 0.05. "
        f"If this fails consistently, the research hypothesis may need re-examination."
    )
