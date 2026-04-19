"""
Diagnose the VA-distance similarity distribution before choosing a threshold.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CSV_PATH = "pseudo_labels.csv"
THRESHOLDS = [0.80, 0.85, 0.90, 0.95]

df = pd.read_csv(CSV_PATH)
va = df[["valence", "arousal"]].values.astype(np.float32)

# Pairwise similarity (upper triangle only, excluding diagonal)
diff = va[:, np.newaxis, :] - va[np.newaxis, :, :]
dist = np.sqrt((diff ** 2).sum(axis=2))
similarity = 1.0 - dist / np.sqrt(2)

n = len(va)
upper = similarity[np.triu_indices(n, k=1)]  # (n*(n-1)/2,) values

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

# ── Histogram of similarity values ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.patch.set_facecolor("#F8F8F8")
fig.suptitle("VA Similarity Distribution — choosing a threshold", fontsize=13)

# Full distribution
ax = axes[0]
ax.hist(upper, bins=60, color="#2A9D8F", edgecolor="white", linewidth=0.4, alpha=0.85)
for t in THRESHOLDS:
    ax.axvline(t, linewidth=1.3, linestyle="--", label=f"t={t}")
ax.set_xlabel("Similarity (1 − dist/√2)", fontsize=10)
ax.set_ylabel("# pairs", fontsize=10)
ax.set_title("All pairwise similarities", fontsize=11)
ax.legend(fontsize=9)
sns.despine(ax=ax)

# Zoom on the high end (>= 0.75) where thresholds live
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
plt.savefig("va_similarity_diagnosis.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("\nSaved: va_similarity_diagnosis.png")
