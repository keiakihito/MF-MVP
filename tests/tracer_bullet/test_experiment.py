"""
Tests for experiment.py — run_experiments() interface.

These tests validate the structure and contract of the experiment runner
without executing real training. The pipeline is mocked so tests remain fast.

TDD: contracts defined here; experiment.py raises NotImplementedError until implemented.
"""

import pytest
from unittest.mock import patch

from tracer_bullet.experiment import run_experiments

pytestmark = pytest.mark.unit

FAKE_METRICS = {
    "hit@k_mean":  0.3,
    "hit@k_std":   0.05,
    "ndcg@k_mean": 0.2,
    "ndcg@k_std":  0.03,
}


def _mock_run_pipeline(*args, **kwargs):
    return FAKE_METRICS


# ── run_experiments structure ─────────────────────────────────────────────────

def test_run_experiments_returns_all_default_signal_types():
    with patch("tracer_bullet.experiment.run_pipeline", side_effect=_mock_run_pipeline):
        results = run_experiments(signal_types=["threshold", "weighted", "time_decay"])
    assert set(results.keys()) == {"threshold", "weighted", "time_decay"}


def test_run_experiments_each_result_has_metric_keys():
    with patch("tracer_bullet.experiment.run_pipeline", side_effect=_mock_run_pipeline):
        results = run_experiments(signal_types=["threshold"])
    assert set(results["threshold"].keys()) == {
        "hit@k_mean", "hit@k_std", "ndcg@k_mean", "ndcg@k_std"
    }


def test_run_experiments_custom_signal_list():
    with patch("tracer_bullet.experiment.run_pipeline", side_effect=_mock_run_pipeline):
        results = run_experiments(signal_types=["threshold"])
    assert list(results.keys()) == ["threshold"]


def test_run_experiments_pipeline_called_once_per_signal():
    with patch("tracer_bullet.experiment.run_pipeline", side_effect=_mock_run_pipeline) as mock:
        run_experiments(signal_types=["threshold", "weighted"])
    assert mock.call_count == 2


def test_run_experiments_passes_k_to_pipeline():
    with patch("tracer_bullet.experiment.run_pipeline", side_effect=_mock_run_pipeline) as mock:
        run_experiments(signal_types=["threshold"], k=20)
    call_kwargs = mock.call_args_list[0][1]
    assert call_kwargs.get("k") == 20
