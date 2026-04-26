"""
End-to-end tracer bullet RecSys pipeline.

The pipeline accepts signal_type as a parameter so the exact same architecture
can be evaluated under different interaction signal definitions. This is the
key design choice: signal construction is explicit and swappable, not hardcoded.

Pipeline steps
--------------
1. Load MovieLens 1M             (dataset.py)
2. Build interactions            (signal.py — signal_type selects strategy)
3. Temporal train/test split     (signal.py)
4. Train MF on train events      (candidate_generation.py → mf.mf_experiment)
5. Generate top-K candidates     (candidate_generation.py)
6. Re-rank with LightGBM         (reranking.py) [future — skipped for now]
7. Evaluate Hit@K and NDCG@K     (evaluation.py)

Usage (from project root)
--------------------------
    PYTHONPATH=src .venv/bin/python src/tracer_bullet/pipeline.py

Public API
----------
run_pipeline(data_dir, signal_type="threshold", k=10) -> dict[str, float]
    Run the full pipeline for a single signal strategy.
    Returns evaluation metrics: hit@k_mean, hit@k_std, ndcg@k_mean, ndcg@k_std.
"""

from __future__ import annotations

import os
from typing import Dict

from tracer_bullet.dataset import load_ratings
from tracer_bullet.signal import build_interactions, train_test_split_by_time
from tracer_bullet.candidate_generation import build_interaction_matrix, generate_candidates
from tracer_bullet.evaluation import evaluate

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "raw", "movielens_1m",
)


def run_pipeline(
    data_dir: str = DATA_DIR,
    signal_type: str = "threshold",
    k: int = 10,
    mf_f: int = 16,
    mf_epochs: int = 100,
    mf_lr: float = 0.01,
    **signal_kwargs,
) -> Dict[str, float]:
    """
    Run the end-to-end tracer bullet pipeline for a single signal strategy.

    Args:
        data_dir:      Path to the directory containing MovieLens 1M .dat files.
        signal_type:   Interaction signal strategy ("threshold", "weighted", "time_decay").
        k:             Cutoff rank for evaluation metrics.
        mf_f:          MF latent factor dimensionality.
        mf_epochs:     MF training epochs.
        mf_lr:         MF learning rate.
        **signal_kwargs: Passed through to build_interactions (e.g. threshold=4).

    Returns:
        {
            "hit@k_mean":  float,
            "hit@k_std":   float,
            "ndcg@k_mean": float,
            "ndcg@k_std":  float,
        }

    TODO: implement after individual modules are complete
    """
    raise NotImplementedError


if __name__ == "__main__":
    results = run_pipeline(signal_type="threshold", k=10)
    print(results)
