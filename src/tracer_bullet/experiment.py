"""
Experiment runner: compare multiple signal strategies side by side.

Mirrors the research design of the music MVP — where tag overlap vs. VA distance
signals were compared — now applied to MovieLens 1M with standard RecSys metrics.

This module answers the core research question:
    How does the choice of interaction signal affect MF learnability,
    generalization, and recommendation quality?

Usage (from project root)
--------------------------
    PYTHONPATH=src .venv/bin/python src/tracer_bullet/experiment.py

Public API
----------
run_experiments(
    data_dir,
    signal_types=["threshold", "weighted", "time_decay"],
    k=10,
) -> dict[str, dict[str, float]]

    Runs run_pipeline() for each signal_type and returns:
    {
        "threshold":  {"hit@k_mean": ..., "hit@k_std": ..., "ndcg@k_mean": ..., "ndcg@k_std": ...},
        "weighted":   { ... },
        "time_decay": { ... },
    }

    Also prints a comparison table to stdout:
    Signal        hit@10_mean  hit@10_std  ndcg@10_mean  ndcg@10_std
    threshold     0.312        0.041       0.198         0.033
    weighted      0.287        0.055       0.181         0.042
    time_decay    0.298        0.047       0.192         0.039
"""

from __future__ import annotations

import os
from typing import Dict, List

from tracer_bullet.pipeline import run_pipeline

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "raw", "movielens_1m",
)

DEFAULT_SIGNAL_TYPES = ["threshold", "weighted", "time_decay"]


def run_experiments(
    data_dir: str = DATA_DIR,
    signal_types: List[str] = DEFAULT_SIGNAL_TYPES,
    k: int = 10,
    **pipeline_kwargs,
) -> Dict[str, Dict[str, float]]:
    """
    Run the pipeline for each signal strategy and return a comparison dict.

    Args:
        data_dir:       Path to MovieLens 1M .dat files.
        signal_types:   List of signal strategy names to compare.
        k:              Cutoff rank for evaluation metrics.
        **pipeline_kwargs: Forwarded to run_pipeline (e.g. mf_epochs=50).

    Returns:
        {signal_type: {metric_name: value, ...}, ...}
    """
    results = {}
    for signal_type in signal_types:
        results[signal_type] = run_pipeline(
            data_dir=data_dir, signal_type=signal_type, k=k, **pipeline_kwargs
        )
    _print_comparison_table(results, k)
    return results


def _print_comparison_table(
    results: Dict[str, Dict[str, float]],
    k: int,
) -> None:
    """Print a formatted comparison table of experiment results to stdout."""
    header = f"{'Signal':<14} {'hit@'+str(k)+'_mean':>14} {'hit@'+str(k)+'_std':>12} {'ndcg@'+str(k)+'_mean':>14} {'ndcg@'+str(k)+'_std':>12}"
    print(header)
    print("-" * len(header))
    for signal_type, metrics in results.items():
        print(
            f"{signal_type:<14} "
            f"{metrics['hit@k_mean']:>14.4f} "
            f"{metrics['hit@k_std']:>12.4f} "
            f"{metrics['ndcg@k_mean']:>14.4f} "
            f"{metrics['ndcg@k_std']:>12.4f}"
        )


if __name__ == "__main__":
    results = run_experiments(k=10)
    _print_comparison_table(results, k=10)
