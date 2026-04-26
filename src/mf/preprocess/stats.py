"""
Summary statistics for an interaction matrix DataFrame.
"""

import numpy as np
import pandas as pd


def summarize(matrix_df: pd.DataFrame, name: str) -> dict:
    m = matrix_df.values.astype(float)
    n = m.shape[0]
    total = n * n

    row_sums = m.sum(axis=1)
    is_symmetric = np.allclose(m, m.T)

    stats = {
        "name":         name,
        "shape":        f"{n} x {n}",
        "density":      round(m.sum() / total, 4),
        "symmetric":    is_symmetric,
        "min":          round(m.min(), 4),
        "max":          round(m.max(), 4),
        "mean":         round(m.mean(), 4),
        "row_sum_min":  round(row_sums.min(), 1),
        "row_sum_max":  round(row_sums.max(), 1),
        "row_sum_mean": round(row_sums.mean(), 2),
        "row_sum_std":  round(row_sums.std(), 2),
    }
    return stats


def print_summary(stats: dict) -> None:
    print(f"\n{'='*45}")
    print(f"  {stats['name']}")
    print(f"{'='*45}")
    print(f"  Shape          : {stats['shape']}")
    print(f"  Density        : {stats['density']}")
    print(f"  Symmetric      : {stats['symmetric']}")
    print(f"  Value min/max  : {stats['min']} / {stats['max']}")
    print(f"  Value mean     : {stats['mean']}")
    print(f"  Row-sum min    : {stats['row_sum_min']}")
    print(f"  Row-sum max    : {stats['row_sum_max']}")
    print(f"  Row-sum mean   : {stats['row_sum_mean']} ± {stats['row_sum_std']}")
