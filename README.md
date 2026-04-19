# iPalpiti MF Experiment — MVP

Applying Matrix Factorization (Koren et al., 2009) to the **iPalpiti dataset**, a collection of 203 classical music tracks annotated with emotional character tags (energetic, tense, calm, lyrical) derived from valence/arousal predictions.

---

## Repository structure

```
MVP/
├── data/               CSVs: source labels and interaction matrices
├── figures/            Diagnostic and comparison plots
├── src/                Python scripts for building and analysing matrices
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Two interaction matrix definitions

Because this dataset uses **pseudo-labels** derived from a model (Music2Emo), the definition of "interaction" is not given — it must be constructed. Two versions are provided so MF experiments can test how sensitive results are to the interaction definition.

### Case 1 — Tag Overlap (baseline)

`data/matrix_case1_tag_overlap.csv`

Two tracks interact (value = 1) if they share **at least one active character tag** among {energetic, tense, calm, lyrical}.

- Simple and interpretable
- Produces a hard block structure: tracks within the same tag group are fully connected, cross-group pairs are almost always 0
- Row-sum distribution is bimodal — large tag groups (calm) dominate; rare tags (lyrical, none) are nearly isolated
- Density: **0.283**

### Case 2 — VA Distance (t = 0.95)

`data/matrix_case2_va_distance_t095.csv`

Two tracks interact (value = 1) if their **valence/arousal similarity** exceeds 0.95, where similarity is defined as:

```
similarity(i, j) = 1 − euclidean_distance(VA_i, VA_j) / √2
```

`√2` is the maximum possible distance in the unit [0,1]² VA space, so similarity is always in [0, 1].

The threshold 0.95 was chosen by inspecting the full pairwise similarity distribution (see `figures/va_similarity_diagnosis.png`): all 203 tracks cluster in a narrow VA region (min similarity = 0.63), so only thresholds above ~0.93 produce a density comparable to Case 1.

- Grounded in the continuous audio predictions rather than the discretised tag rules
- Interactions cross tag-group boundaries, producing a softer and more distributed pattern
- Row-sum distribution is unimodal and more uniform (σ = 21 vs 31 for Case 1) — every track has a comparable number of neighbors, which reduces bias toward dominant tag groups but may also reduce the discriminative signal available to MF
- Density: **0.248**

### Why two versions?

The character tags in this dataset are themselves derived from VA thresholds (quantile-based rules applied to Music2Emo outputs). Case 1 interactions therefore inherit the discretisation artefacts of that tagging step. Case 2 bypasses the tag layer entirely and works directly from the continuous VA scores, making it a useful sanity check: if MF learns similar factors from both matrices, the tag discretisation is not distorting the signal. If results diverge, it indicates the block structure in Case 1 is driving the learned factors rather than genuine audio similarity.

## Running the scripts

```bash
# Rebuild both matrices and generate comparison figure
.venv/bin/python src/compare_matrices.py

# Diagnose VA similarity distribution (threshold selection)
.venv/bin/python src/diagnose_va_similarity.py
```

## Reference

Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *IEEE Computer*, 42(8), 30–37.
