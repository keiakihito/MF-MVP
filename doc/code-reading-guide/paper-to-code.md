# Reading Guide: Koren et al. (2009) × This Codebase

A side-by-side map of the paper's concepts and the code that implements them.
Read one section of the paper, then open the corresponding file and look for
the anchors listed here.

---

## How to use this guide

```
Paper section  →  What the paper says
Code anchor    →  Where to find it in src/
What to notice →  The key insight to confirm in the code
```

---

## 1. The Big Picture — What Problem Are We Solving?

**Paper (pp. 30–31, "Recommender System Strategies")**

The paper contrasts two approaches:
- *Content filtering*: build a profile of each item from its attributes
- *Collaborative filtering*: infer preferences from past behavior patterns

Matrix Factorization is a **latent factor model** — a subtype of collaborative
filtering that compresses the user-item interaction matrix into two low-dimensional
embedding matrices.

**This codebase**

We have no real users. Instead, tracks play both roles:
- A track acting as a **"user"** expresses preferences via its interaction row
- A track acting as an **"item"** receives scores via its interaction column

This is item-to-item collaborative filtering — the same math, applied to a
synthetic interaction matrix built from content features (tags or VA distance).

```
src/preprocess/matrices.py    ← where the proxy interaction matrix is built
src/mf/mf_experiment.py       ← load_data() — where it's loaded as R
```

> **Key question to hold while reading:** The paper uses real ratings (Netflix
> stars). We use pseudo-interactions from music features. How does that
> change what the embeddings mean?

---

## 2. The Interaction Matrix — r_ui

**Paper (p. 32, "A Basic Matrix Factorization Model")**

> "Recommender systems rely on different types of input data, which are often
> placed in a matrix with one dimension representing users and the other
> representing items of interest."

The paper's matrix is **sparse** — most (user, item) pairs have no rating.
Our matrix is **dense-ish** (~25–28% non-zero) because every track pair gets
a definite 0 or 1 from the content signal.

| Paper concept | Paper notation | Code location |
|---|---|---|
| Observed rating | r_ui | `R` in `train_mf()` |
| Known pairs for training | κ (kappa, Eq. 2) | `_sample_observed(R)` in `build_training_set()` |
| Unobserved (implicit 0) | — (implicit feedback framing) | `_sample_unobserved(R, n)` |

**Code anchor:** `src/mf/mf_experiment.py`, Section 3

```python
def _sample_observed(R):
    # These are the κ pairs in Eq. 2 — entries where r_ui = 1
    u_idx, i_idx = np.where(R == 1)

def _sample_unobserved(R, n_samples):
    # Zero entries treated as "no preference" — implicit feedback assumption
    neg_u, neg_i = np.where(R == 0)
```

**Two matrices, two signals:**

| Signal | File | Structure | Expected MF behavior |
|---|---|---|---|
| Tag overlap | `matrix_case1_tag_overlap.csv` | Block-diagonal | Clean clusters in embedding space |
| VA distance | `matrix_case2_va_distance_t095.csv` | Smooth/distributed | Gradient manifold in embedding space |

**Code anchor:** `src/preprocess/matrices.py`

```python
def build_tag_overlap_matrix(df):
    # R[i,j] = 1 if tracks share ≥1 tag
    # → hard block structure, bimodal row-sum distribution

def build_va_distance_matrix(df, threshold=0.95):
    # R[i,j] = 1 if VA similarity ≥ threshold
    # → softer structure, unimodal row-sum distribution
```

> **What to notice:** Run `src/analysis/compare_matrices.py` and look at
> `figures/matrix_comparison.png`. The block structure of Case 1 is exactly
> what will produce cluster-like embeddings.

---

## 3. The Model — Eq. 1: r̂_ui = q_i^T p_u

**Paper (p. 32, "A Basic Matrix Factorization Model")**

> "Each item i is associated with a vector q_i ∈ ℝ^f, and each user u is
> associated with a vector p_u ∈ ℝ^f … The resulting dot product, q_i^T p_u,
> captures the interaction between user u and item i."

This is the entire model in one equation. Everything else is about how to
learn q_i and p_u well.

| Paper term | Paper notation | Code |
|---|---|---|
| User latent factor vector | p_u ∈ ℝ^f | `model.P` (nn.Embedding) |
| Item latent factor vector | q_i ∈ ℝ^f | `model.Q` (nn.Embedding) |
| Latent dimensionality | f | `f=16` parameter in `train_mf()` |
| Predicted rating | r̂_ui | return value of `model.forward()` |

**Code anchor:** `src/mf/mf_experiment.py`, Section 2 — `class MFModel`

```python
class MFModel(nn.Module):
    # P[u] = p_u   (user latent factor, paper Eq. 1)
    # Q[i] = q_i   (item latent factor, paper Eq. 1)

    def forward(self, u_idx, i_idx):
        return (self.P(u_idx) * self.Q(i_idx)).sum(dim=-1)
        # This is exactly q_i^T p_u (element-wise product then sum = dot product)
```

> **What to notice:** `f=16` means each track gets a 16-dimensional vector.
> The paper uses 20–200 factors for Netflix. We use 16 because our dataset
> is tiny (203 tracks). Try changing `f` and see how CV-MSE changes.

---

## 4. Learning — Eq. 2: Minimizing Squared Error

**Paper (p. 32, "Learning Algorithms")**

The system learns p_u and q_i by minimizing regularized squared error over
the known pairs κ:

```
min  Σ_(u,i)∈κ  (r_ui − q_i^T p_u)²  +  λ(‖q_i‖² + ‖p_u‖²)
```

Our MVP drops regularization (λ=0) for simplicity — we just minimize MSE.
The paper discusses two solvers: **SGD** and **ALS**. We use Adam (a variant
of SGD with adaptive learning rates).

| Paper concept | Paper notation | Code |
|---|---|---|
| Loss function | Eq. 2 (MSE term) | `nn.MSELoss()` in `_init_model()` |
| Learning rate | γ (gamma) | `lr=0.01` in `train_mf()` |
| Training pairs | κ | `build_training_set()` output |
| One SGD step | Eq. 3 update rules | `optimizer.step()` in `_train_one_epoch()` |
| Regularization | λ (lambda) | **not implemented** in MVP |

**Code anchor:** `src/mf/mf_experiment.py`, Section 4

```python
def _train_one_epoch(model, loader, optimizer, loss_fn, device):
    for u, i, r in loader:
        loss = loss_fn(model(u, i), r)   # MSE: (r_ui − r̂_ui)²
        optimizer.zero_grad()
        loss.backward()                   # compute gradients
        optimizer.step()                  # SGD update (Adam variant)
```

**Training loss plot:** After running the script, open
`figures/mf_training_curves.png`. You should see both curves start high
and decrease — confirming the model is minimizing Eq. 2.

> **What to notice:** The Tag model (Case 1) typically converges faster and
> lower than the VA model (Case 2). This is because the block structure of
> R_tag is easier to factorize — the model quickly learns "tracks with the
> same tag should have similar p_u vectors."

---

## 5. The Latent Space — What Do the Factors Mean?

**Paper (p. 31, Figure 2)**

> "Consider two hypothetical dimensions characterized as female- versus
> male-oriented and serious versus escapist."

The paper shows that Netflix movie embeddings naturally cluster by genre
along interpretable axes — even though the model was never told what genres
are. The factors emerge from the rating patterns.

In our case, the "genres" are emotional tags (energetic, tense, calm, lyrical).
If MF works correctly on R_tag, the learned p_u vectors should cluster by tag
without ever being told what the tags mean.

**Code anchor:** `src/mf/mf_experiment.py`, Section 8 — `visualize_embeddings()`

```python
def _plot_one_embedding(ax, model, colors, title):
    emb    = model.P.weight.detach().cpu().numpy()  # all p_u vectors, shape (N, f)
    coords = pca.fit_transform(emb)                  # project f dims → 2D
    # Color each point by its emotional tag
    # → if clusters appear, MF has recovered the tag structure
```

Open `figures/mf_embeddings.png` after running:
- **Case 1 (Tag):** You should see 4 colored clusters. This is the paper's
  Figure 2 analogy — the factors have captured the emotional categories.
- **Case 2 (VA):** Points spread more continuously. The factors capture
  smooth VA proximity rather than hard categories.

> **Key insight from the paper:** The factors are *latent* — they are not
> predefined dimensions like "energetic". The model discovers them from
> co-occurrence patterns in R. The fact that they align with our tags in
> Case 1 is the validation that MF works.

---

## 6. Implicit Feedback — Why We Need Negative Sampling

**Paper (p. 32, "A Basic Matrix Factorization Model" + p. 34, "Additional Input Sources")**

> "When explicit feedback is not available, recommender systems can infer user
> preferences using implicit feedback, which indirectly reflects opinion by
> observing user behavior."

Our entire R matrix is implicit feedback — we never asked anyone whether
track A is similar to track B. We inferred it from tags or VA distance.

A subtlety: if you only train on the positive entries (R[i,j]=1), the model
has never seen a negative example. It will learn to predict 1 for everything.
We must also show it pairs that should score 0.

**Code anchor:** `src/mf/mf_experiment.py`, `build_training_set()`

```python
def build_training_set(R, confidence_ratio=1.0):
    pos_u, pos_i = _sample_observed(R)      # r_ui = 1  (confirmed interactions)
    neg_u, neg_i = _sample_unobserved(R, …) # r_ui = 0  (no interaction observed)
    # confidence_ratio=1.0 means equal positives and negatives
    # → analogous to the paper's confidence weighting c_ui (Eq. 8)
```

> **What to try:** Change `confidence_ratio` to 2.0 (twice as many negatives)
> and observe whether CV-MSE improves. More negatives = stricter penalty for
> false positives in recommendation.

---

## 7. Recommendation — Scoring All Items

**Paper (p. 32, Eq. 1)**

Once p_u and q_i are learned, predicting r̂_ui for a new (u, i) pair is just
a dot product. To recommend top-K items for user u, compute r̂_ui for all i
and rank them.

**Code anchor:** `src/mf/mf_experiment.py`, Section 5

```python
def recommend(model, u_idx, track_ids, k=10):
    r_hat_u = model.predict_scores(u_idx)   # Q @ p_u — scores for all items
    r_hat_u[u_idx] = -inf                   # exclude self
    top_k = argsort(r_hat_u)[::-1][:k]     # rank by score
```

```python
def predict_scores(self, u_idx):
    p_u = self.P.weight[u_idx]   # (f,)  — the query track's latent vector
    return self.Q.weight @ p_u   # (N,)  — dot product with every item
```

**Comparison output:** `compare_recommendations()` prints results for the same
query track under both models side by side. Look for:

| Case 1 (Tag) | Case 2 (VA) |
|---|---|
| All top-K share the same tag as the query | Top-K may include tracks from adjacent VA region |
| Cluster-like, categorical | Smoother, crosses tag boundaries |

---

## 8. Cross-Validation — Measuring Generalization

**Paper (p. 32, Eq. 2 + "Learning Algorithms")**

The paper minimizes error on *known* ratings κ but warns about overfitting.
Cross-validation tests whether the learned factors generalize to *unseen* pairs.

Our K-Fold CV:
1. Split observed positive entries (upper triangle of R) into 5 folds
2. For each fold: train on 4/5 of interactions, test on the held-out 1/5
3. Report mean ± std of MSE across folds

**Code anchor:** `src/mf/mf_experiment.py`, Section 9

```python
def kfold_cv(R, f=16, epochs=100, lr=0.01, n_splits=5):
    rows, cols = _get_cv_entries(R)   # upper triangle only → no symmetric leakage
    for train_idx, test_idx in KFold(n_splits).split(rows):
        R_train = R.copy()
        R_train[held_out] = 0         # hide test entries from model
        model, _ = train_mf(R_train, …)
        mse = _evaluate_fold(model, …)
```

**Typical result:**
```
Case 1 Tag Overlap:  MSE = 0.10 ± 0.005
Case 2 VA Distance:  MSE = 0.13 ± 0.004
```

Lower MSE in Case 1 confirms: the **block structure of the tag matrix is
easier for MF to learn and generalize**, consistent with the paper's
observation that strong signal in the interaction matrix produces better
latent factors.

---

## 9. Things the MVP Does NOT Implement (yet)

These are in the paper but out of scope for this MVP. Knowing they exist
helps you see what "vanilla MF" is missing:

| Paper section | Concept | Why it matters |
|---|---|---|
| p. 33, "Adding Biases" (Eq. 3–5) | Global mean μ, item bias b_i, user bias b_u | Removes systematic rating tendencies (some items always rated higher) |
| p. 33, "Adding Biases" (Eq. 4) | r̂_ui = μ + b_i + b_u + q_i^T p_u | More accurate prediction by isolating the interaction term |
| p. 33, "Alternating Least Squares" | ALS optimizer | Better for implicit feedback datasets |
| p. 34, "Temporal Dynamics" | b_i(t), p_u(t) | User taste and item popularity change over time |
| p. 34–35, "Inputs with Varying Confidence" (Eq. 8) | c_ui weighting | High-frequency interactions get more weight |

---

## Quick Reference: Paper Notation → Code

| Paper | Meaning | Code |
|---|---|---|
| r_ui | observed interaction | `R[u, i]` (float32, 0 or 1) |
| r̂_ui | predicted interaction | `model(u_idx, i_idx)` |
| p_u | user latent vector | `model.P.weight[u]` |
| q_i | item latent vector | `model.Q.weight[i]` |
| f | latent dimensionality | `f=16` in `train_mf()` |
| κ | known training pairs | output of `_sample_observed()` |
| γ | learning rate | `lr=0.01` |
| λ | regularization weight | not implemented |
| N | number of tracks | `N = R.shape[0]` = 203 |

---

## Suggested Reading Order

1. **Paper pp. 30–31** → understand collaborative filtering vs. content filtering
   → then open `src/preprocess/matrices.py` and read the module docstring

2. **Paper p. 32, "A Basic Matrix Factorization Model"** → understand Eq. 1
   → then open `src/mf/mf_experiment.py` and read `class MFModel`

3. **Paper p. 32, "Learning Algorithms" → SGD** → understand Eq. 2 and the update rules
   → then read `_train_one_epoch()` and `train_mf()`

4. **Run the script**, watch the tqdm loss bars drop
   → open `figures/mf_training_curves.png` → confirm Eq. 2 is being minimized

5. **Paper p. 31, Figure 2** → the latent space geometry illustration
   → open `figures/mf_embeddings.png` → compare your PCA plot to the paper's figure

6. **Paper p. 32, "Additional Input Sources"** → implicit feedback
   → read `build_training_set()` and the `confidence_ratio` parameter

7. Read the CV results printout
   → connect to the paper's goal of generalizing to *unseen* ratings
