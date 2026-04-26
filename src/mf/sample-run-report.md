# Sample Run Report
## MF Signal Design Experiment — Koren et al. (2009)

---

## 1. How to Read the Output

### 1.1 Data Check (Line 4)

```
N=203 tracks  |  R_tag density=0.283  R_va density=0.248
```

| Field | Value | What it means |
|---|---|---|
| `N=203` | 203 tracks | P and Q are both (203 × 16) matrices |
| `R_tag density=0.283` | 28.3% of entries are 1 | Case 1 matrix — moderately sparse, block structure |
| `R_va density=0.248` | 24.8% of entries are 1 | Case 2 matrix — slightly sparser, smoother structure |

Both matrices are in the "sparse" range the paper assumes for collaborative filtering.
If density were near 1.0, there would be no signal — everything is similar to everything.
If density were near 0.0, nothing would be similar — the model has nothing to learn from.

---

### 1.2 Training Progress (Lines 5–6)

```
Case 1 (Tag Overlap): 100/100  loss=0.0005
Case 2 (VA Distance): 100/100  loss=0.0232
```

Each bar is one model training for 100 epochs. The `loss=` number is the **final
per-epoch mean MSE** — how far the model's predictions r̂_ui are from the true r_ui
on the training set (Eq. 2 in the paper).

**How to read the progress bar:**
- Bar filling left to right = epochs completing (0 → 100)
- `loss=` number dropping over time = the model is learning
- A flat loss late in training = convergence (updating P and Q is no longer helping)

**What the final loss numbers mean:**

| Model | Final train loss | Interpretation |
|---|---|---|
| Case 1 Tag | **0.0005** | Near-perfect fit — block structure is easy to compress into 16 factors |
| Case 2 VA  | **0.0232** | 46× higher — smooth continuous similarity is harder to factorize |

This gap already tells you the answer before looking at any recommendations.

---

### 1.3 Recommendation Output

Each recommendation block has this structure:

```
================================================================================
Query: id=6  [tense+calm]  V=0.26 A=0.35
  Prelude A L'Unisson
================================================================================

  --- Case 1: Tag Overlap ---
  id=12  [tense]  V=0.35 A=0.38  Serenade For Strings In G Minor ...
  ...

  --- Case 2: VA Distance ---
  id=39  [tense]  V=0.28 A=0.38  Minimax ...
  ...
```

**How to read each result row:**

```
id=12   [tense               ]   V=0.35  A=0.38   Serenade For Strings ...
  ↑          ↑                     ↑        ↑           ↑
track ID   emotional tag(s)      valence  arousal     title (truncated)
```

- **Tags** — what emotional category the track belongs to (from pseudo_labels.csv)
- **V (valence)** — 0=negative emotion, 1=positive emotion
- **A (arousal)** — 0=calm/slow, 1=energetic/fast
- The query track's tags/VA tell you what "neighborhood" the model should be
  recommending from

**What to look for:**
- **Case 1:** Do the recommended tags match the query's tag? (they should — cluster behavior)
- **Case 2:** Do the recommended V/A values sit close to the query's V/A? (they should — proximity behavior)
- **The interesting case:** When Case 2 recommends a *different tag* but similar V/A — that is the signal crossing a categorical boundary

---

### 1.4 CV Training Losses (Lines 101–110)

```
CV Tag fold 1/5: 50/50  loss=0.0230
CV Tag fold 2/5: 50/50  loss=0.0220
...
CV VA  fold 1/5: 50/50  loss=0.0367
...
```

Each line is one of the 10 models trained during K-Fold CV (5 folds × 2 cases).
These training losses are higher than the full-run losses because:
- Only 50 epochs (not 100) — less time to converge
- 20% fewer training entries per fold — less signal per model

Watch for **consistency across folds**: if all 5 tag folds finish near 0.022 and
all 5 VA folds finish near 0.037, the behavior is stable and not a fluke of one
particular data split.

---

### 1.5 K-Fold CV Results (Lines 115–116)

```
Case 1 Tag Overlap:  MSE = 0.1011 ± 0.0042
Case 2 VA Distance:  MSE = 0.1315 ± 0.0067
```

These are **test MSE** — measured on held-out entries the model never saw during
training. This is the generalization score: lower = better.

| Metric | Tag | VA | Difference |
|---|---|---|---|
| Mean MSE | 0.1011 | 0.1315 | VA is 30% worse |
| Std ± | 0.0042 | 0.0067 | VA is 60% more unstable |

- **Mean** — how well the model predicts unseen interactions on average
- **Std** — how consistent this performance is across different data splits.
  High std means the model's quality depends heavily on *which* entries happen
  to be in the training set vs. test set — a sign of structural ambiguity in the signal.

---

## 2. Key Findings

### Finding 1 — Tag signal is dramatically easier to fit

Training MSE: 0.0005 (Tag) vs. 0.0232 (VA) — a **46× gap**.

The tag interaction matrix has a clean block-diagonal structure: tracks with the
same tag always interact (R=1), tracks with different tags almost never interact
(R=0). This is exactly the kind of pattern MF is designed to exploit — the model
can learn four tight clusters in ℝ^16 and perfectly reproduce the matrix.

The VA matrix has a smoother, more continuous structure. There is no hard
boundary between "similar" and "not similar" — the threshold at 0.95 creates
many borderline cases. MF must learn a gradient manifold, which is harder to
compress and always leaves residual error.

---

### Finding 2 — Tag signal generalizes better (CV MSE: 0.1011 vs. 0.1315)

Even though both models are tested on *unseen* interactions, the tag model
predicts held-out pairs significantly better (30% lower MSE).

Why? Because the tag structure is **global and consistent** — if you know a
track is `[tense]`, you know it interacts with *every* other `[tense]` track,
including the ones held out. The model learns this rule from the training fold
and applies it correctly to the test fold.

The VA model's continuous similarity is **local** — knowing that track A is
close to track B doesn't tell you much about track A's relationship to track C,
which might be equidistant to B but on the other side. This local ambiguity
causes higher test error and higher variance across folds.

---

### Finding 3 — The two signals produce qualitatively different recommendations

The clearest example is **Query track 65 `[tense]` V=0.35, A=0.37**:

**Case 1 (Tag):** All top-10 are `[tense]` or `[tense+calm]`
```
[tense+calm]  V=0.31  A=0.33
[tense]       V=0.30  A=0.36
[tense+calm]  V=0.33  A=0.29
...
```
→ Stayed entirely within the emotional category. The model learned: "tense tracks
go together." V/A values vary widely (A ranges from 0.29 to 0.42).

**Case 2 (VA):** Top-10 includes `[lyrical]`, `[calm]`, and `[none]`
```
[lyrical]  V=0.39  A=0.39
[none]     V=0.39  A=0.41
[none]     V=0.37  A=0.42
[calm]     V=0.36  A=0.34
```
→ Crossed tag boundaries entirely. The model learned: "tracks at similar VA
coordinates go together." V/A values cluster tightly around the query's V=0.35,
A=0.37 — regardless of tag label.

This is the central experimental result: **same query, same model architecture,
different signal → qualitatively different recommendations**.

---

### Finding 4 — The VA model is more unstable (std: ±0.0067 vs. ±0.0042)

CV std is 60% higher for the VA model. This means the VA model's quality varies
more depending on which interactions are held out. When a borderline pair (VA
similarity just above 0.95) ends up in the test set, the model struggles —
because the training set gave it no guidance for such marginal cases. The tag
model never has this problem: a pair is either same-tag (R=1) or not (R=0),
with no ambiguity.

---

## 3. Mapping to the Paper

### 3.1 The core model — Eq. 1

> *"Each item i is associated with a vector q_i ∈ ℝ^f … the resulting dot
> product q_i^T p_u captures the interaction between user u and item i."*
> — p. 32

In the output, every recommendation score is exactly this dot product.
When `predict_scores(u_idx)` computes `Q @ p_u`, it is evaluating Eq. 1
for all 203 items simultaneously.

The **PCA scatter plot** (`figures/mf_embeddings.png`) visualizes the learned
p_u vectors in 2D. The paper's Figure 2 shows the same idea — movies
clustering by genre in the latent space. Our Case 1 plot should show the
same: tracks clustering by emotional tag without ever being told what tags are.

---

### 3.2 Learning via SGD — Eq. 2

> *"To learn the factor vectors, the system minimizes the regularized squared
> error on the set of known ratings κ."* — p. 32

The training loss numbers in the progress bars are exactly the MSE term of
Eq. 2 (we omit regularization λ in the MVP). The loss dropping from ~0.50
at epoch 1 to 0.0005 at epoch 100 is the optimizer solving Eq. 2.

The **training curve plot** (`figures/mf_training_curves.png`) shows this
minimization visually — the same type of convergence curve referenced in
the paper's Figure 4 (RMSE vs. number of parameters).

---

### 3.3 Implicit feedback — the unobserved entries

> *"Implicit feedback usually denotes the presence or absence of an event,
> so it is typically represented by a densely filled matrix."* — p. 32

Our entire experiment is implicit feedback. Neither R_tag nor R_va comes
from users explicitly expressing preferences — both are inferred from content
features. The zero entries in R do not mean "dislike"; they mean "no
observed similarity under this signal definition."

This is why `_sample_unobserved()` exists: without negative examples, the
model never learns that some pairs should score 0. The paper's §"Inputs with
Varying Confidence Levels" (Eq. 8) addresses the same problem with confidence
weights c_ui. Our `confidence_ratio=1.0` is the simplest version of this.

---

### 3.4 Generalization — the purpose of CV

> *"The system learns the model by fitting the previously observed ratings.
> However, the goal is to generalize those previous ratings in a way that
> predicts future, unknown ratings."* — p. 32

K-Fold CV directly measures this generalization. The test MSE (0.1011 vs.
0.1315) answers the paper's central question for our dataset: **which signal
definition produces factors that generalize better to unseen interactions?**
Answer: the tag signal, by a 30% margin.

---

### 3.5 What these results suggest about signal design

The paper was written for explicit ratings (Netflix stars). Our experiment
extends the question to a new setting: **when you have no ratings and must
construct a pseudo-interaction matrix from content features, the structural
properties of that matrix fundamentally shape what MF can learn.**

| Matrix property | Tag (Case 1) | VA (Case 2) |
|---|---|---|
| Structure | Block-diagonal | Smooth/continuous |
| Train MSE | 0.0005 (easy to fit) | 0.0232 (hard to fit) |
| CV MSE | 0.1011 (generalizes well) | 0.1315 (generalizes less) |
| Recommendation style | Categorical clusters | Continuous proximity |
| Paper analogy | Strong explicit signal | Noisy implicit signal |

The tag matrix behaves like strong explicit feedback — the signal is clean and
consistent. The VA matrix behaves like noisy implicit feedback — the signal is
real but ambiguous at the boundaries. The paper (Figure 4) shows that noisier
signals require more factors and more data to achieve the same accuracy. Our
results are consistent with this: with only f=16 and N=203, the cleaner signal
wins clearly.
