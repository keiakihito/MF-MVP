"""
Offline evaluation metrics for recommendation lists.

Implements Hit@K and NDCG@K with per-user variance — critical for comparing
signal strategies: a signal that is better on average but highly unstable
across users is less desirable than a slightly weaker but consistent one.

Public API
----------
hit_at_k(recommended, relevant, k) -> float
    1.0 if any of the top-k recommended items is in relevant, else 0.0.

ndcg_at_k(recommended, relevant, k) -> float
    Normalized Discounted Cumulative Gain at K.
    Rewards relevant items appearing earlier in the ranked list.

evaluate(recommendations, ground_truth, k) -> dict[str, float]
    Aggregates metrics across all users. Returns:
    {
        "hit@k_mean":  float,
        "hit@k_std":   float,
        "ndcg@k_mean": float,
        "ndcg@k_std":  float,
    }
"""

from __future__ import annotations

import math
from typing import Dict, List, Set


def hit_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """
    Return 1.0 if any of the first k recommended items is in relevant, else 0.0.

    Args:
        recommended: Ordered list of recommended item IDs (best first).
        relevant:    Set of ground-truth relevant item IDs for this user.
        k:           Cutoff rank.

    Returns:
        1.0 or 0.0.

    """
    if not relevant:
        return 0.0
    return 1.0 if any(item in relevant for item in recommended[:k]) else 0.0


def ndcg_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at K.

    DCG@K  = Σ_{r=1}^{K} rel_r / log2(r + 1)   where rel_r ∈ {0, 1}
    IDCG@K = DCG of a perfect ranking (all relevant items first)
    NDCG@K = DCG@K / IDCG@K  (returns 0.0 if relevant is empty)

    Args:
        recommended: Ordered list of recommended item IDs (best first).
        relevant:    Set of ground-truth relevant item IDs for this user.
        k:           Cutoff rank.

    Returns:
        Float in [0.0, 1.0].
    """
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, item in enumerate(recommended[:k])
        if item in relevant
    )
    # IDCG: best case — all relevant items appear first
    n_relevant_in_k = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(n_relevant_in_k))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(
    recommendations: Dict[int, List[int]],
    ground_truth: Dict[int, Set[int]],
    k: int,
) -> Dict[str, float]:
    """
    Aggregate Hit@K and NDCG@K across all users.

    Args:
        recommendations: {user_id: [item_id, ...]} — ordered recommendation lists.
        ground_truth:    {user_id: {item_id, ...}} — relevant items per user.
        k:               Cutoff rank.

    Returns:
        {
            "hit@k_mean":  float,
            "hit@k_std":   float,
            "ndcg@k_mean": float,
            "ndcg@k_std":  float,
        }
    """
    hits, ndcgs = [], []
    for user_id, recs in recommendations.items():
        gt = ground_truth.get(user_id, set())
        hits.append(hit_at_k(recs, gt, k))
        ndcgs.append(ndcg_at_k(recs, gt, k))

    n = len(hits)
    def mean(xs): return sum(xs) / n
    def std(xs, m): return math.sqrt(sum((x - m) ** 2 for x in xs) / n)

    hit_mean = mean(hits)
    ndcg_mean = mean(ndcgs)
    return {
        "hit@k_mean":  hit_mean,
        "hit@k_std":   std(hits, hit_mean),
        "ndcg@k_mean": ndcg_mean,
        "ndcg@k_std":  std(ndcgs, ndcg_mean),
    }
