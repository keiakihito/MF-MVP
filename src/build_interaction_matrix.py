import pandas as pd
import numpy as np

CSV_PATH = "pseudo_labels.csv"
CHARACTER_TAGS = ["energetic", "tense", "calm", "lyrical"]

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} tracks")

# Active tags per track: set of tags where value == 1
tag_sets = {
    row["track_id"]: set(tag for tag in CHARACTER_TAGS if row[tag] == 1)
    for _, row in df.iterrows()
}

track_ids = sorted(tag_sets.keys())
n = len(track_ids)
print(f"Building {n} x {n} interaction matrix...")

# Build matrix: interaction[i][j] = 1 if tracks share at least one active tag
matrix = np.zeros((n, n), dtype=np.int8)
for i, tid_i in enumerate(track_ids):
    for j, tid_j in enumerate(track_ids):
        if tag_sets[tid_i] & tag_sets[tid_j]:
            matrix[i][j] = 1

interaction_df = pd.DataFrame(matrix, index=track_ids, columns=track_ids)
interaction_df.index.name = "track_id"
interaction_df.columns.name = "track_id"

out_path = "pseudo_interaction_matrix.csv"
interaction_df.to_csv(out_path)
print(f"Saved to {out_path}")

# Sanity checks
print(f"\nMatrix shape: {interaction_df.shape}")
print(f"Non-zero interactions: {matrix.sum()} / {n * n}")
print(f"Density: {matrix.sum() / (n * n):.4f}")

# Example rows for inspection
sample_ids = track_ids[:5]
print("\nExample rows (first 5 tracks, first 8 columns):")
print(interaction_df.loc[sample_ids, track_ids[:8]])

print("\nTag sets for first 5 tracks:")
for tid in sample_ids:
    print(f"  track {tid}: {tag_sets[tid]}")
