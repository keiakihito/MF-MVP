import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

CSV_PATH = "pseudo_labels.csv"
MATRIX_PATH = "pseudo_interaction_matrix.csv"
CHARACTER_TAGS = ["energetic", "tense", "calm", "lyrical"]

# Tag palette — one color per dominant tag, grey for no active tag
TAG_COLORS = {
    "energetic": "#E76F51",
    "tense":     "#264653",
    "calm":      "#2A9D8F",
    "lyrical":   "#E9C46A",
    "none":      "#AAAAAA",
}

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
matrix_df = pd.read_csv(MATRIX_PATH, index_col="track_id")
matrix_df.columns = matrix_df.columns.astype(int)

# Dominant tag = first active tag in the priority order; "none" if all zero
def dominant_tag(row):
    for tag in CHARACTER_TAGS:
        if row[tag] == 1:
            return tag
    return "none"

df["dominant_tag"] = df.apply(dominant_tag, axis=1)
tag_map = dict(zip(df["track_id"], df["dominant_tag"]))

# Sort tracks by dominant tag so same-tag blocks cluster together
tag_order = CHARACTER_TAGS + ["none"]
sorted_ids = (
    df.sort_values("dominant_tag", key=lambda s: s.map({t: i for i, t in enumerate(tag_order)}))
      ["track_id"]
      .tolist()
)
sorted_matrix = matrix_df.loc[sorted_ids, sorted_ids]

# Tag boundary positions for divider lines
tag_counts = df["dominant_tag"].value_counts().reindex(tag_order).fillna(0).astype(int)
boundaries = np.cumsum(tag_counts.values)[:-1]

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 8))
fig.patch.set_facecolor("#F8F8F8")
gs = fig.add_gridspec(1, 2, wspace=0.35)

ax_heat = fig.add_subplot(gs[0])
ax_spy  = fig.add_subplot(gs[1])

# ── 1. Sorted heatmap ─────────────────────────────────────────────────────────
sns.heatmap(
    sorted_matrix,
    ax=ax_heat,
    cmap=sns.light_palette("#2A9D8F", as_cmap=True),
    xticklabels=False,
    yticklabels=False,
    cbar_kws={"shrink": 0.6, "label": "Interaction (0 / 1)"},
    linewidths=0,
)

# Draw tag-group divider lines
for b in boundaries:
    ax_heat.axhline(b, color="white", linewidth=1.2, alpha=0.8)
    ax_heat.axvline(b, color="white", linewidth=1.2, alpha=0.8)

# Colored row-side strip showing dominant tag of each track
strip_width = 4
strip_colors = [TAG_COLORS[tag_map[tid]] for tid in sorted_ids]
for i, color in enumerate(strip_colors):
    ax_heat.add_patch(mpatches.Rectangle(
        (-strip_width, i), strip_width, 1,
        color=color, clip_on=False, transform=ax_heat.transData
    ))

ax_heat.set_title("Interaction Matrix\n(sorted by dominant character tag)", fontsize=13, pad=14)
ax_heat.set_xlabel("Track (item)", fontsize=10)
ax_heat.set_ylabel("Track (user)", fontsize=10)

# Tag legend
legend_patches = [
    mpatches.Patch(color=TAG_COLORS[t], label=t.capitalize())
    for t in tag_order
]
ax_heat.legend(
    handles=legend_patches,
    title="Dominant tag",
    bbox_to_anchor=(1.18, 1), loc="upper left",
    fontsize=9, title_fontsize=9,
    framealpha=0.9,
)

# ── 2. Spy / dot plot ─────────────────────────────────────────────────────────
n = len(sorted_ids)
rows, cols = np.where(sorted_matrix.values == 1)

ax_spy.scatter(cols, rows, s=0.4, color="#264653", alpha=0.5, linewidths=0)

for b in boundaries:
    ax_spy.axhline(b, color="#E76F51", linewidth=0.8, alpha=0.6)
    ax_spy.axvline(b, color="#E76F51", linewidth=0.8, alpha=0.6)

ax_spy.set_xlim(0, n)
ax_spy.set_ylim(n, 0)
ax_spy.set_aspect("equal")
ax_spy.set_title("Sparsity Pattern\n(each dot = 1 interaction)", fontsize=13, pad=14)
ax_spy.set_xlabel("Track (item)", fontsize=10)
ax_spy.set_ylabel("Track (user)", fontsize=10)
ax_spy.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax_spy.set_facecolor("#F0F0F0")

# Annotate density
density = sorted_matrix.values.sum() / (n * n)
ax_spy.text(
    0.97, 0.03, f"Density: {density:.3f}",
    transform=ax_spy.transAxes,
    ha="right", va="bottom", fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
)

# ── Save ──────────────────────────────────────────────────────────────────────
out = "interaction_matrix_viz.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out}")
plt.show()
