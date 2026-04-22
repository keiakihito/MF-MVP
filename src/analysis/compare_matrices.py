"""
Compare the two pre-built interaction matrices visually.
Reads CSVs from data/ — run generate_matrices.py first if they don't exist.

Usage (from project root):
    .venv/bin/python src/analysis/compare_matrices.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR  = os.path.join(BASE_DIR, "figures")

CSV_PATH   = os.path.join(DATA_DIR, "pseudo_labels.csv")
CASE1_PATH = os.path.join(DATA_DIR, "matrix_case1_tag_overlap.csv")
CASE2_PATH = os.path.join(DATA_DIR, "matrix_case2_va_distance_t095.csv")
OUT_FIGURE = os.path.join(FIG_DIR, "matrix_comparison.png")

CHARACTER_TAGS = ["energetic", "tense", "calm", "lyrical"]

TAG_COLORS = {
    "energetic": "#E76F51",
    "tense":     "#264653",
    "calm":      "#2A9D8F",
    "lyrical":   "#E9C46A",
    "none":      "#AAAAAA",
}

SUMMARY_TEXT = (
    "Case 1 (tag overlap) shows a hard block structure: tracks with the same\n"
    "dominant tag form dense clusters, while cross-tag pairs are almost always 0.\n"
    "The row-sum distribution is bimodal — tracks are either highly connected\n"
    "(large tag group) or nearly isolated (lyrical / none).\n\n"
    "Case 2 (VA distance, t=0.95) produces a softer, more distributed pattern.\n"
    "Interactions spread across tag boundaries because VA position is continuous\n"
    "and tag-group boundaries do not perfectly align with VA proximity.\n"
    "Row sums are unimodal and more uniform — every track has a similar number\n"
    "of neighbors. This reduces bias toward dominant tag groups, but the\n"
    "reduced variance may also weaken the discriminative signal for MF."
)


# ── helpers ───────────────────────────────────────────────────────────────────

def dominant_tag(row):
    for tag in CHARACTER_TAGS:
        if row[tag] == 1:
            return tag
    return "none"


def sorted_ids_by_tag(df):
    df = df.copy()
    df["dominant_tag"] = df.apply(dominant_tag, axis=1)
    tag_order = CHARACTER_TAGS + ["none"]
    return (
        df.sort_values("dominant_tag",
                       key=lambda s: s.map({t: i for i, t in enumerate(tag_order)}))
          ["track_id"].tolist()
    )


def tag_boundaries(df):
    df = df.copy()
    df["dominant_tag"] = df.apply(dominant_tag, axis=1)
    tag_order = CHARACTER_TAGS + ["none"]
    counts = df["dominant_tag"].value_counts().reindex(tag_order).fillna(0).astype(int)
    return np.cumsum(counts.values)[:-1]


def draw_strip(ax, track_ids, tag_map, strip_w=4):
    for i, tid in enumerate(track_ids):
        ax.add_patch(mpatches.Rectangle(
            (-strip_w, i), strip_w, 1,
            color=TAG_COLORS[tag_map[tid]], clip_on=False, transform=ax.transData
        ))


def plot_heatmap(ax, matrix_df, sorted_ids, boundaries, tag_map, title, cmap):
    m = matrix_df.loc[sorted_ids, sorted_ids]
    sns.heatmap(m, ax=ax, cmap=cmap,
                xticklabels=False, yticklabels=False,
                cbar_kws={"shrink": 0.55, "label": "Interaction"},
                linewidths=0, vmin=0, vmax=1)
    for b in boundaries:
        ax.axhline(b, color="white", linewidth=1.0, alpha=0.8)
        ax.axvline(b, color="white", linewidth=1.0, alpha=0.8)
    draw_strip(ax, sorted_ids, tag_map)
    density = m.values.sum() / (len(sorted_ids) ** 2)
    ax.set_title(f"{title}\ndensity={density:.3f}", fontsize=11, pad=12)
    ax.set_xlabel("Track (item)", fontsize=9)
    ax.set_ylabel("Track (user)", fontsize=9)


def plot_spy(ax, matrix_df, sorted_ids, boundaries, title):
    m = matrix_df.loc[sorted_ids, sorted_ids].values
    n = len(sorted_ids)
    rows, cols = np.where(m > 0)
    ax.scatter(cols, rows, s=0.4, color="#264653", alpha=0.5, linewidths=0)
    for b in boundaries:
        ax.axhline(b, color="#E76F51", linewidth=0.7, alpha=0.6)
        ax.axvline(b, color="#E76F51", linewidth=0.7, alpha=0.6)
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    ax.set_facecolor("#F0F0F0")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    density = m.sum() / (n * n)
    ax.text(0.97, 0.03, f"Density: {density:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
    ax.set_title(title, fontsize=11, pad=12)
    ax.set_xlabel("Track (item)", fontsize=9)
    ax.set_ylabel("Track (user)", fontsize=9)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    df    = pd.read_csv(CSV_PATH)
    case1 = pd.read_csv(CASE1_PATH, index_col="track_id")
    case2 = pd.read_csv(CASE2_PATH, index_col="track_id")
    case1.columns = case1.columns.astype(int)
    case2.columns = case2.columns.astype(int)
    print(f"Loaded {len(df)} tracks")

    s_ids   = sorted_ids_by_tag(df)
    bounds  = tag_boundaries(df)
    tag_map = {row["track_id"]: dominant_tag(row) for _, row in df.iterrows()}

    legend_patches = [
        mpatches.Patch(color=TAG_COLORS[t], label=t.capitalize())
        for t in CHARACTER_TAGS + ["none"]
    ]

    fig = plt.figure(figsize=(20, 15))
    fig.patch.set_facecolor("#F8F8F8")
    fig.suptitle(
        "Interaction Matrix Comparison\n"
        "Case 1: Tag Overlap (baseline)   vs   Case 2: VA Distance (t=0.95)",
        fontsize=14, y=0.99
    )

    gs = fig.add_gridspec(3, 3, wspace=0.38, hspace=0.45,
                          width_ratios=[1, 1, 0.55],
                          height_ratios=[1, 1, 0.38])

    ax_h1  = fig.add_subplot(gs[0, 0])
    ax_h2  = fig.add_subplot(gs[0, 1])
    ax_leg = fig.add_subplot(gs[0, 2])
    ax_s1  = fig.add_subplot(gs[1, 0])
    ax_s2  = fig.add_subplot(gs[1, 1])
    ax_rs  = fig.add_subplot(gs[1, 2])
    ax_txt = fig.add_subplot(gs[2, :])

    plot_heatmap(ax_h1, case1, s_ids, bounds, tag_map,
                 "Case 1 — Tag Overlap",
                 sns.light_palette("#264653", as_cmap=True))
    plot_heatmap(ax_h2, case2, s_ids, bounds, tag_map,
                 "Case 2 — VA Distance (t=0.95)",
                 sns.light_palette("#2A9D8F", as_cmap=True))

    ax_leg.axis("off")
    ax_leg.legend(handles=legend_patches, title="Dominant tag",
                  loc="center", fontsize=10, title_fontsize=10, framealpha=0.9)

    plot_spy(ax_s1, case1, s_ids, bounds, "Case 1 — Sparsity Pattern")
    plot_spy(ax_s2, case2, s_ids, bounds, "Case 2 — Sparsity Pattern")

    rs1 = case1.values.sum(axis=1)
    rs2 = case2.values.sum(axis=1)
    ax_rs.hist(rs1, bins=25, color="#264653", alpha=0.65, edgecolor="white",
               label=f"Case 1  μ={rs1.mean():.0f}")
    ax_rs.hist(rs2, bins=25, color="#2A9D8F", alpha=0.65, edgecolor="white",
               label=f"Case 2  μ={rs2.mean():.0f}")
    ax_rs.axvline(rs1.mean(), color="#264653", linewidth=1.5, linestyle="--")
    ax_rs.axvline(rs2.mean(), color="#2A9D8F", linewidth=1.5, linestyle="--")
    ax_rs.set_title("Row-sum distribution", fontsize=11, pad=8)
    ax_rs.set_xlabel("# interactions per track", fontsize=9)
    ax_rs.set_ylabel("# tracks", fontsize=9)
    ax_rs.legend(fontsize=9)
    ax_rs.set_facecolor("#F8F8F8")
    sns.despine(ax=ax_rs)

    ax_txt.axis("off")
    ax_txt.text(0.01, 0.95, "Summary", fontsize=11, fontweight="bold",
                va="top", transform=ax_txt.transAxes)
    ax_txt.text(0.01, 0.75, SUMMARY_TEXT, fontsize=9.5, va="top",
                transform=ax_txt.transAxes, linespacing=1.6,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor="#CCCCCC", alpha=0.9))

    plt.savefig(OUT_FIGURE, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved: {OUT_FIGURE}")


if __name__ == "__main__":
    main()
