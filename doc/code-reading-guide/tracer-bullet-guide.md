# Reading Guide: Tracer Bullet RecSys Pipeline

A map of the `src/tracer_bullet/` package — what each file does, how the
pieces connect, and how the design mirrors the signal-design research question
explored in the music MF MVP.

---

## How to use this guide

```
Research question  →  The experiment this code is designed to answer
Design decision    →  Why the code is structured the way it is
Code anchor        →  Exact file and function to open
What to notice     →  The key thing to look for in that function
```

---

## 1. The Central Research Question

The music MVP tested one question:
> "Does tag overlap vs. VA distance produce different MF embeddings and
> recommendations?"

The tracer bullet asks the same question on a real-world dataset:
> "How does the choice of interaction signal affect MF learnability,
> generalization, and recommendation quality?"

The three signals are:

| Signal | Definition | Research analogy |
|---|---|---|
| `threshold` | rating ≥ 4 → interaction = 1, rest dropped | Like tag overlap — binary, hard cutoff |
| `weighted` | normalized rating → interaction ∈ [0, 1] | Like VA similarity — soft, graded |
| `time_decay` | exp(-λ · days_since) → weights recency | Time dimension not present in music MVP |

**Code anchor:** `src/tracer_bullet/experiment.py` — `run_experiments()`

```python
results = run_experiments(signal_types=["threshold", "weighted", "time_decay"])
# Each key in results is one signal's evaluation metrics
# {"threshold": {"hit@k_mean": ..., "ndcg@k_mean": ..., ...}, ...}
```

> **Key question to hold while reading:** Does a harder signal (threshold)
> generalize better, or does the softer signal (weighted) give the model more
> useful gradient to learn from?

---

## 2. Data Loading — `dataset.py`

MovieLens 1M uses an unusual `.dat` format with `::` separators and `latin-1`
encoding. The three loaders hide this from the rest of the codebase.

**Code anchor:** `src/tracer_bullet/dataset.py`

```python
load_ratings(data_dir) -> DataFrame  # [user_id, movie_id, rating, timestamp]
load_users(data_dir)   -> DataFrame  # [user_id, gender, age, occupation, zip]
load_movies(data_dir)  -> DataFrame  # [movie_id, title, genres (list[str])]
```

**What to notice in `load_movies`:**
```python
df["genres"] = df["genres"].apply(lambda g: g.split("|"))
# "Action|Adventure|Sci-Fi"  →  ["Action", "Adventure", "Sci-Fi"]
```
Genres are stored as a pipe-separated string in the file. The loader splits
them into a list so downstream code can treat them as categorical features.
Users and movies are separate `.dat` files that join on `user_id` / `movie_id`
when features are needed (e.g., for the LightGBM re-ranker in Stage 2).

**Data directory:** `data/raw/movielens_1m/` — place `ratings.dat`, `users.dat`,
and `movies.dat` here after downloading from
https://grouplens.org/datasets/movielens/1m/

---

## 3. Signal Construction — `signal.py`

This is the core experimental module — the direct equivalent of
`build_tag_overlap_matrix` and `build_va_distance_matrix` in the music MVP.

**All three functions produce the same output schema:**

```
[user_id, movie_id, interaction, timestamp]
```

The downstream pipeline never needs to know which signal was used — it just
sees a DataFrame with an `interaction` column. This is the design that makes
controlled comparison possible.

### 3a. Threshold signal

```python
def build_threshold_signal(ratings_df, threshold=4.0) -> DataFrame:
```

Ratings ≥ threshold → `interaction = 1.0`. Rows below threshold are **dropped
entirely** — they are not encoded as `interaction = 0`. Those pairs are treated
as unobserved, not as disconfirmed preferences. Negative sampling (adding
explicit 0-interaction pairs) is a separate downstream concern.

> **Why drop instead of encode as 0?**
> Rating 2 out of 5 is not the same as "this user has never seen this movie."
> Encoding it as 0 would tell the model "I know this user dislikes this item,"
> which is stronger than the data actually says. Dropping preserves the
> uncertainty.

### 3b. Weighted signal

```python
def build_weighted_signal(ratings_df) -> DataFrame:
```

Min-max normalizes the raw rating to `[0, 1]`:

```
interaction = (rating - min_rating) / (max_rating - min_rating)
```

All rows are kept. Rating 1 → `0.0`; rating 5 → `1.0`. The model gets a
gradient even on weak interactions, which may help it learn nuanced preferences
— or introduce noise from borderline ratings.

### 3c. Time-decay signal

```python
def build_time_decay_signal(ratings_df, decay_rate=0.001) -> DataFrame:
```

Weights each interaction by how recent it is:

```
interaction = exp(-decay_rate × days_since_event)
```

where `days_since_event = (max_timestamp - timestamp) / 86400`. The most recent
event always gets `interaction = 1.0`; older events decay toward 0 but stay
strictly positive.

> **Paper connection (Koren 2009, p. 34, "Temporal Dynamics"):**
> The paper discusses time-varying biases `b_i(t)` and `p_u(t)`. Our
> `time_decay` signal is a simpler form of the same idea — we encode recency
> directly into the interaction weight rather than into the model parameters.

### 3d. Dispatcher

```python
def build_interactions(ratings_df, signal_type="threshold", **kwargs) -> DataFrame:
    # Routes to the correct build_*_signal function
    # Raises ValueError for unknown signal_type
```

The pipeline always calls `build_interactions` — it never calls the individual
strategy functions directly. This is what allows `experiment.py` to swap
strategies with a single parameter change.

### 3e. Temporal split

```python
def train_test_split_by_time(events_df, test_ratio=0.2) -> (train_df, test_df):
```

Splits on the timestamp quantile at `(1 - test_ratio)`. Every event in train
happened before every event in test — no temporal leakage.

> **Why not a random split?**
> Random splits let the model train on ratings from 2003 and predict ratings
> from 2001. That would be evaluated on data from the past, which is
> unrealistic and inflates metrics. Temporal split simulates real deployment:
> the model is trained on history and evaluated on the future.

---

## 4. Candidate Generation — `candidate_generation.py`

Wraps the existing MF code from `src/mf/mf_experiment.py` to work on the
MovieLens user-item matrix instead of the music item-item matrix.

**Code anchor:** `src/tracer_bullet/candidate_generation.py`

```python
from mf.mf_experiment import train_mf, recommend   # reuses existing MF code

def build_interaction_matrix(events_df, n_users, n_items) -> np.ndarray:
    # (N_users × N_items) float32 matrix
    # R[u, i] = interaction value from signal.py

def generate_candidates(model, user_idx, k=100, track_ids=None) -> list[int]:
    # Returns top-k item IDs for user_idx, scored by r̂_ui = q_i^T p_u
```

**What is different from the music MVP:**

| Music MVP | Tracer bullet |
|---|---|
| Item × item matrix (R is square) | User × item matrix (R is rectangular) |
| Both P and Q index tracks | P indexes users, Q indexes items |
| 203 × 203 = ~41K entries | 6040 users × 3883 movies = ~23M entries |
| One model per signal | One model per signal per experiment run |

The MF math is identical — it is still Eq. 1 from the paper (`r̂_ui = q_i^T p_u`).
Only the interpretation of the rows and columns changes.

---

## 5. Evaluation — `evaluation.py`

Replaces the MSE metric from the music MVP with ranking metrics that directly
measure recommendation quality.

**Code anchor:** `src/tracer_bullet/evaluation.py`

### Hit@K

```python
def hit_at_k(recommended, relevant, k) -> float:
    # 1.0 if any item in recommended[:k] is in relevant, else 0.0
```

Binary: did the user's ground-truth item appear in the top-K list?
Useful for measuring whether the system surfaces at least one relevant item.

### NDCG@K

```python
def ndcg_at_k(recommended, relevant, k) -> float:
    # DCG@K / IDCG@K
    # DCG@K  = Σ  1 / log2(rank + 2)   for each relevant item in top-K
    # IDCG@K = DCG of perfect ranking (all relevant items first)
```

Position-sensitive: a relevant item at rank 1 scores higher than at rank K.
The `log2(rank + 2)` denominator is the standard DCG discount — rank 1
contributes `1/log2(2) = 1.0`, rank 2 contributes `1/log2(3) ≈ 0.63`, etc.

### evaluate()

```python
def evaluate(recommendations, ground_truth, k) -> dict:
    # Returns per-user mean AND std of both metrics
    {
        "hit@k_mean":  float,
        "hit@k_std":   float,   # ← per-user variance — critical for signal comparison
        "ndcg@k_mean": float,
        "ndcg@k_std":  float,
    }
```

**Why std matters for signal comparison:**

A signal that produces `hit@10_mean = 0.31, hit@10_std = 0.04` is preferable
to one with `hit@10_mean = 0.31, hit@10_std = 0.12` — equal average quality
but the first is consistent across users. High std means the signal helps some
users a lot while failing others, which is a fairness concern.

> **Contrast with music MVP:** The music MVP used MSE as the metric (Eq. 2 from
> the paper). MSE measures how well the model reconstructed the interaction
> matrix — a training-time quality measure. Hit@K and NDCG@K measure whether
> the top-K list given to a real user contains something useful — a deployment
> quality measure. These are fundamentally different questions.

---

## 6. The Pipeline — `pipeline.py`

Wires steps 1–7 together with `signal_type` as the controllable parameter.

**Code anchor:** `src/tracer_bullet/pipeline.py`

```python
def run_pipeline(data_dir, signal_type="threshold", k=10,
                 mf_f=16, mf_epochs=100, mf_lr=0.01, **signal_kwargs):

    # Step 1: Load data
    ratings_df = load_ratings(data_dir)

    # Step 2: Build signal-specific interactions
    events_df = build_interactions(ratings_df, signal_type=signal_type, **signal_kwargs)

    # Step 3: Temporal split
    train_df, test_df = train_test_split_by_time(events_df, test_ratio=0.2)

    # Step 4: Build interaction matrix and train MF
    R = build_interaction_matrix(train_df, n_users, n_items)
    model, _ = train_mf(R, f=mf_f, epochs=mf_epochs, lr=mf_lr)

    # Step 5: Generate top-K candidates per user
    recommendations = {u: generate_candidates(model, u, k=k) for u in test_users}

    # Step 6: [future] Re-rank with LightGBM

    # Step 7: Evaluate
    ground_truth = {u: set(test_df[test_df.user_id == u]["movie_id"]) for u in test_users}
    return evaluate(recommendations, ground_truth, k=k)
```

Everything in the pipeline is fixed except `signal_type`. Running the same
pipeline with `"threshold"`, `"weighted"`, and `"time_decay"` gives three
comparable results because all other variables are held constant.

---

## 7. The Experiment Runner — `experiment.py`

```python
def run_experiments(signal_types=["threshold", "weighted", "time_decay"], k=10):
    results = {}
    for signal_type in signal_types:
        results[signal_type] = run_pipeline(signal_type=signal_type, k=k)
    _print_comparison_table(results, k)
    return results
```

This is the outermost loop. It calls `run_pipeline` once per signal, collects
results, and prints a comparison table:

```
Signal         hit@10_mean   hit@10_std  ndcg@10_mean  ndcg@10_std
-----------------------------------------------------------------
threshold           0.3120       0.0410        0.1980       0.0330
weighted            0.2870       0.0550        0.1810       0.0420
time_decay          0.2980       0.0470        0.1920       0.0390
```

Reading the table:
- Higher `hit@k_mean` and `ndcg@k_mean` → better average recommendation quality
- Lower `hit@k_std` and `ndcg@k_std` → more consistent across users
- Compare rows to see which signal design wins on this dataset

> **Hypothesis to test:** The threshold signal may win on NDCG (cleaner positive
> examples → model learns stronger preferences), but the weighted signal may
> win on std (softer signal gives the model information about borderline users
> it would otherwise ignore).

---

## 8. File Map and Reading Order

```
src/tracer_bullet/
  dataset.py             ← Step 1: I/O boundary — read data in, clean it up
  signal.py              ← Step 2–3: Core experiment variable — swap here to compare
  candidate_generation.py← Step 4–5: Reuses MF from music MVP
  evaluation.py          ← Step 7: Pure math — easiest to read first
  pipeline.py            ← Wires steps 1–7, signal_type is the only free variable
  experiment.py          ← Outer loop — runs pipeline N times, prints table
  reranking.py           ← Step 6: Future LightGBM stage (not yet implemented)
```

**Suggested reading order (easiest → most complex):**

1. `evaluation.py` — pure math, no dependencies. Read `hit_at_k` and `ndcg_at_k`
   to understand what "better recommendation" means here.

2. `signal.py` — the experimental heart. Read all three `build_*_signal`
   functions and notice what each one changes and what stays the same.

3. `dataset.py` — boring but necessary. Skim it; the key detail is the
   `genres` parsing and `latin-1` encoding.

4. `candidate_generation.py` — shows how the music MF code is reused on a
   different matrix shape.

5. `pipeline.py` — read the docstring step list, then trace through how
   `signal_type` flows from argument → `build_interactions` dispatcher.

6. `experiment.py` — the outermost loop. Read `run_experiments` and
   `_print_comparison_table` to understand what the final output looks like.

---

## 9. Connecting to the Music MVP

| Music MVP | Tracer bullet | What's the same |
|---|---|---|
| `build_tag_overlap_matrix` | `build_threshold_signal` | Binary interaction signal |
| `build_va_distance_matrix` | `build_weighted_signal` | Graded/soft signal |
| — | `build_time_decay_signal` | New: temporal dimension |
| `train_mf(R_tag)` vs `train_mf(R_va)` | `run_experiments(signal_types=[...])` | Same controlled comparison |
| MSE from K-Fold CV | Hit@K, NDCG@K from temporal split | Both measure generalization; different lens |
| `visualize_embeddings()` — PCA of P | — (future) | Latent space geometry per signal |
| Music item×item (N=203) | MovieLens user×item (6040×3883) | Same MF math, different scale |

The tracer bullet is the music MVP's hypothesis — *different signals → different
behavior* — tested on a dataset where ground truth is real user ratings rather
than synthesized from content features.

---

## 10. Quick Reference

| Symbol / Term | Definition | Code location |
|---|---|---|
| `interaction` | Signal-specific interaction weight for (user, item) pair | `signal.py` output column |
| `signal_type` | Which of the three strategies to use | `pipeline.py` argument |
| `R[u, i]` | Entry in the interaction matrix | `candidate_generation.build_interaction_matrix` |
| `r̂_ui` | Predicted interaction (Koren Eq. 1) | `model.forward()` in `mf_experiment.py` |
| `hit@k` | 1 if any relevant item in top-K, else 0 | `evaluation.hit_at_k` |
| `ndcg@k` | DCG@K / IDCG@K — position-weighted relevance | `evaluation.ndcg_at_k` |
| `test_ratio=0.2` | 20% of events (by time) held out for evaluation | `signal.train_test_split_by_time` |
| `k=10` | Cutoff for top-K recommendation list | `pipeline.run_pipeline` argument |
