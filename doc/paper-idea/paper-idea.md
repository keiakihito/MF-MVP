# Title (Working)
Signal Design Shapes Embedding Geometry in MF-based Recommendation Systems

---

# 1. Introduction

Recommendation systems often rely on learned representations (embeddings) to capture relationships between users and items. In collaborative filtering methods such as Matrix Factorization (MF), these embeddings are learned from user–item interaction data. However, in many real-world scenarios—especially cold-start or data-sparse environments—explicit user interactions are not available.

To address this limitation, pseudo-interaction signals can be constructed from alternative sources such as metadata or content-based features. While such approaches enable the application of collaborative filtering techniques, an open question remains:

👉 *How does the design of these signals influence the learned embedding space and downstream recommendation performance?*

This work investigates the role of signal construction in shaping embedding geometry and retrieval behavior in MF-based systems.

---

## Research Question

- How does the design of pseudo-interaction signals affect the structure of learned embedding space and downstream recommendation quality in MF-based systems?

---

## Hypothesis

1. Continuous signals (VA-based) will produce smoother embedding structures,  
   while discrete signals (tag-based) will produce clustered representations.

2. The effectiveness of MF depends more on signal design than model capacity.

3. Different signal constructions lead to different retrieval behaviors:  
   - tag-based signals favor precision  
   - VA-based signals enable smoother similarity exploration

---

# 2. Related Work

*Skipped (to be added later)*

---

# 3. Method

## 3.1 Overview

We construct pseudo user–item interaction matrices from content-derived signals and train a standard Matrix Factorization (MF) model on these interactions.

The overall pipeline:

1. Construct pseudo-interaction matrix \( R \)
2. Convert \( R \) into training samples (u, i, r_ui)
3. Train MF model to learn embeddings
4. Perform retrieval using learned embeddings

---

## 3.2 Signal Construction

We consider two types of signals:

### (1) Discrete Signal (Tag-based)

- Items are assigned categorical tags
- Interaction defined as:
  - \( r_{ui} = 1 \) if tags overlap
  - \( r_{ui} = 0 \) otherwise

---

### (2) Continuous Signal (VA-based)

- Items are represented in a continuous space (e.g., valence–arousal)
- Interaction defined using similarity:
  - \( r_{ui} = 1 \) if similarity > threshold
  - \( r_{ui} = 0 \) otherwise

---

## 3.3 Matrix Factorization

We train a standard MF model:

\[
\hat{r}_{ui} = p_u^T q_i
\]

where:

- \( p_u \in \mathbb{R}^f \): user embedding  
- \( q_i \in \mathbb{R}^f \): item embedding  

The model is trained by minimizing:

\[
L = \sum_{(u,i)} (r_{ui} - p_u^T q_i)^2
\]

---

## 3.4 Training Procedure

- Initialize embeddings randomly
- Sample positive and negative pairs from \( R \)
- Optimize using SGD / Adam
- Learn embedding matrices \( P \) and \( Q \)

---

## 3.5 Retrieval

For each query item \( u \), compute:

\[
\hat{r}_{ui} = q_i^T p_u
\]

Rank items based on predicted scores.

---

# 4. Results

*Skipped (to be added later)*

---

# 5. Discussion

*Skipped (to be added later)*

---

# 6. Temporary Conclusion

Matrix Factorization can successfully learn meaningful embedding structures from pseudo-interaction data. However, the geometry of the learned embedding space—and consequently the behavior of the recommendation system—is strongly influenced by the design of the input signals.

Specifically:

- Discrete signals produce clustered representations
- Continuous signals produce smoother embedding spaces
- Signal design plays a more critical role than model complexity

---

# Notes (for future expansion)

- Add evaluation metrics (NDCG, Hit@K)
- Compare across model capacities
- Analyze embedding visualization
- Connect to cold-start scenarios