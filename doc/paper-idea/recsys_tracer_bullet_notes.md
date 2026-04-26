# RecSys Tracer Bullet — Minimal Design Notes

## Research Question
- How does the design of pseudo-interaction signals affect the structure of learned embedding space and downstream recommendation quality in MF-based systems?

## Hypothesis
1. Continuous signals (VA-based) will produce smoother embedding structures,
while discrete signals (tag-based) will produce clustered representations.
2.  The effectiveness of MF depends more on signal design than model capacity.
3.  Different signal constructions lead to different retrieval behaviors:
tag-based signals favor precision, while VA-based signals enable smoother similarity exploration.
 
## Temporary Conclusion
MF successfully learns meaningful embedding structures from pseudo-interaction data,
but the quality and nature of these structures are strongly determined by the signal design.
We show that in MF-based recommendation, the geometry of the embedding space is primarily determined by signal design rather than model complexity.

## Goal
- Extend current MF MVP into an end-to-end RecSys pipeline
- Keep MF as fixed baseline (candidate generation)
- Systematically swap **signal (R construction)**

---

## Current MVP (baseline)

Pipeline (implicit):

data (tracks)
  → build R (Tag or VA)
  → MF
  → recommendations

Key property:
- Model fixed
- Signal changes → behavior changes

---

## Target Pipeline (Tracer Bullet)

data (MovieLens or music)
  → events (implicit feedback)
  → build R
  → MF (candidate generation)
  → rerank (LightGBM)
  → evaluation (Hit@K, NDCG@K)

---

## What to Replace / Extend

### 1. Data
- MVP: iPalpiti (203 tracks)
- TB: MovieLens 1M (+ optional music dataset later)

---

### 2. Signal (CORE EXPERIMENT VARIABLE)

Replace input matrix:
- Tag overlap
- VA distance

With generalized signal types:

- Block structure
- Continuous similarity
- Noisy / thresholded

→ ONLY this should change across experiments

---

### 3. Interaction Matrix (R)

MVP:
- item × item

Tracer Bullet:
- user × item

Conversion:
- rating ≥ 4 → interaction = 1

---

### 4. Candidate Generation

Reuse:
- src/mf

Adapt:
- item-item → user-item

---

### 5. Re-ranking (New)

Add:
- LightGBM ranker

Features:
- user features (age, gender)
- item features (genre)
- interaction features (MF score)

---

### 6. Evaluation (New)

Replace:
- MSE

With:
- Hit@K
- NDCG@K

---

## Experimental Axis

Only vary:

- R construction

Keep fixed:

- Model (MF)
- Dataset (per experiment)
- Evaluation

---

## Hypothesis

Signal structure determines:

- learnability (train loss)
- generalization (CV)
- ranking quality (NDCG)
- recommendation behavior

---

## Extension Plan

Phase 1:
- MovieLens tracer bullet (MF only)

Phase 2:
- Add reranking (LightGBM)

Phase 3:
- Multiple signal definitions
  - rating threshold
  - genre similarity
  - hybrid signals

Phase 4:
- Compare:
  - ranking metrics
  - stability
  - qualitative behavior

---

## Research Direction

Reframe as:

- Not model comparison
- Signal structure analysis

Potential framing:

"Effect of interaction signal structure on MF-based recommendation systems"

---

## Notes

- Keep MF untouched
- Keep changes isolated to tracer_bullet/
- Treat pipeline as experimental scaffold
- Every change = controlled variable


