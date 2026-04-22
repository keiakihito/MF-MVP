"""
Diagnose the VA-distance similarity distribution to inform threshold selection.

Usage (from project root):
    .venv/bin/python src/analysis/diagnose_va_similarity.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR  = os.path.join(BASE_DIR, "figures")

CSV_PATH   = os.path.join(DATA_DIR, "pseudo_labels.csv")
OUT_FIGURE = os.path.join(FIG_DIR, "va_similarity_diagnosis.png")

THRESHOLDS = [0.80, 0.85, 0.90, 0.95]


def main():
    df = pd.read_csv(CSV_PATH)
    va = df[["valence", "arousal"]].values.astype(np.float32)

    diff = va[:, np.newaxis, :] - va[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    similarity = 1.0 - dist / np.sqrt(2)

    n = len(va)
    upper = similarity[np.triu_indices(n, k=1)]

    print("=== VA similarity distribution (upper triangle, no diagonal) ===")
    print(f"  count  : {len(upper):,}")
    print(f"  min    : {upper.min():.4f}")
    print(f"  max    : {upper.max():.4f}")
    print(f"  mean   : {upper.mean():.4f}")
    print(f"  median : {np.median(upper):.4f}")
    print(f"  p90    : {np.percentile(upper, 90):.4f}")
    print(f"  p95    : {np.percentile(upper, 95):.4f}")
    print(f"  p99    : {np.percentile(upper, 99):.4f}")

    print("\n=== Density by threshold ===")
    for t in THRESHOLDS:
        binary = (similarity >= t).astype(float)
        density = binary.sum() / (n * n)
        print(f"  threshold={t:.2f}  →  density={density:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor("#F8F8F8")
    fig.suptitle("VA Similarity Distribution — choosing a threshold", fontsize=13)

    ax = axes[0]
    ax.hist(upper, bins=60, color="#2A9D8F", edgecolor="white", linewidth=0.4, alpha=0.85)
    for t in THRESHOLDS:
        ax.axvline(t, linewidth=1.3, linestyle="--", label=f"t={t}")
    ax.set_xlabel("Similarity (1 − dist/√2)", fontsize=10)
    ax.set_ylabel("# pairs", fontsize=10)
    ax.set_title("All pairwise similarities", fontsize=11)
    ax.legend(fontsize=9)
    sns.despine(ax=ax)

    ax2 = axes[1]
    ax2.hist(upper[upper >= 0.75], bins=50,
             color="#264653", edgecolor="white", linewidth=0.4, alpha=0.85)
    for t in THRESHOLDS:
        ax2.axvline(t, linewidth=1.3, linestyle="--", label=f"t={t}")
    ax2.set_xlabel("Similarity", fontsize=10)
    ax2.set_ylabel("# pairs", fontsize=10)
    ax2.set_title("Zoom: similarity ≥ 0.75", fontsize=11)
    ax2.legend(fontsize=9)
    sns.despine(ax=ax2)

    plt.tight_layout()
    plt.savefig(OUT_FIGURE, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nSaved: {OUT_FIGURE}")


if __name__ == "__main__":
    main()
