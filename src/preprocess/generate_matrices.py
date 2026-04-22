"""
Build both interaction matrices from pseudo_labels.csv and save to data/.

Usage (from project root):
    .venv/bin/python src/preprocess/generate_matrices.py
"""

import os
import pandas as pd

from preprocess.matrices import build_tag_overlap_matrix, build_va_distance_matrix
from preprocess.stats import summarize, print_summary

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")

CSV_PATH  = os.path.join(DATA_DIR, "pseudo_labels.csv")
OUT_CASE1 = os.path.join(DATA_DIR, "matrix_case1_tag_overlap.csv")
OUT_CASE2 = os.path.join(DATA_DIR, "matrix_case2_va_distance_t095.csv")

VA_THRESHOLD = 0.95


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} tracks")

    case1 = build_tag_overlap_matrix(df)
    case2 = build_va_distance_matrix(df, threshold=VA_THRESHOLD)

    print_summary(summarize(case1, "Case 1 — Tag Overlap"))
    print_summary(summarize(case2, f"Case 2 — VA Distance (threshold={VA_THRESHOLD})"))

    case1.to_csv(OUT_CASE1)
    case2.to_csv(OUT_CASE2)
    print(f"\nSaved: {OUT_CASE1}")
    print(f"Saved: {OUT_CASE2}")


if __name__ == "__main__":
    main()
