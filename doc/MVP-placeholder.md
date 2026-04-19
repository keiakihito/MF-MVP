## RQ & Hypothesis

**RQ1**: Can pseudo-interactions derived from content signals support collaborative filtering in a data-scarce music recommendation setting?
**RQ2**: How does the definition of pseudo-interaction affect matrix structure and downstream ranking behavior?
**RQ3**: How do MF-based recommenders compare with content-based embedding retrieval under different pseudo-interaction designs?

**Hypothesis**:

* **H1**: Pseudo-interactions can provide enough signal for MF to learn meaningful latent structure, even without real user logs.
* **H2**: Tag-overlap interactions will produce discrete, block-structured matrices, while VA-distance interactions will produce smoother and more distributed matrices.
* **H3**: MF performance will depend not only on model choice, but also on how pseudo-interactions are constructed.

---

## Method

* Construct two pseudo-interaction matrices from the same dataset:

  * **Case 1:** binary tag-overlap interactions based on VA-derived character tags
  * **Case 2:** thresholded VA-distance interactions based on continuous valence–arousal similarity
* Treat each track as a pseudo-user and each track as a candidate item.
* Analyze the structural properties of each matrix, including density, row-sum distribution, and interaction patterns.
* Train MF-based recommenders on both matrices.
* Compare against content-based baselines using pretrained CNN / Transformer embeddings with cosine similarity.
* Evaluate ranking quality with NDCG@K and Hit@K.

---

## Result (Placeholder)

* The two pseudo-interaction definitions produce markedly different matrix structures, even when matched at similar density.
* Tag-overlap interactions form clear block-diagonal clusters, reflecting discrete label agreement.
* VA-distance interactions form smoother and more distributed connectivity patterns, reflecting continuous affect similarity.
* These structural differences are expected to affect MF training dynamics and ranking behavior.

---

## Conclusion

Pseudo-interactions derived from content signals are not a single design choice but a family of possible constructions.
Their utility for collaborative filtering depends strongly on how the interaction matrix is defined.
In data-scarce settings, understanding the structure of pseudo-interactions may be as important as choosing the recommendation model itself.
