"""
mf_experiment.py — Signal Design × Matrix Factorization

Trains MF on two pseudo-interaction matrices and compares the resulting
embedding geometry and recommendations:

  Case 1 (Tag Overlap):  block-diagonal structure
    → MF learns cluster embeddings; top-K stays within the same emotional tag
  Case 2 (VA Distance):  smoother, distributed pattern
    → MF learns a gradient manifold; top-K may cross tag boundaries

Usage (from project root):
    .venv/bin/python src/mf/mf_experiment.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR  = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Section 1 — Data Loading
# ---------------------------------------------------------------------------

def load_data():
    labels = pd.read_csv(os.path.join(DATA_DIR, "pseudo_labels.csv"))

    R_tag = pd.read_csv(
        os.path.join(DATA_DIR, "matrix_case1_tag_overlap.csv"),
        index_col="track_id",
    )
    R_va = pd.read_csv(
        os.path.join(DATA_DIR, "matrix_case2_va_distance_t095.csv"),
        index_col="track_id",
    )

    track_ids = R_tag.index.tolist()
    R_tag_np  = R_tag.values.astype(np.float32)
    R_va_np   = R_va.values.astype(np.float32)

    return R_tag_np, R_va_np, track_ids, labels


# ---------------------------------------------------------------------------
# Section 2 — MF Model
# ---------------------------------------------------------------------------

class MFModel(nn.Module):
    """
    Vanilla MF: R_hat[i,j] = U[i] · V[j]
    Both U and V are N×k embeddings; a track acts as both "user" and "item"
    because the interaction matrix is item-to-item.
    """
    def __init__(self, n_items: int, k: int):
        super().__init__()
        self.U = nn.Embedding(n_items, k)
        self.V = nn.Embedding(n_items, k)
        nn.init.normal_(self.U.weight, std=0.01)
        nn.init.normal_(self.V.weight, std=0.01)

    def forward(self, u_idx: torch.Tensor, v_idx: torch.Tensor) -> torch.Tensor:
        return (self.U(u_idx) * self.V(v_idx)).sum(dim=-1)

    def score_all(self, u_idx: int) -> torch.Tensor:
        """Return scores for one query track against all item embeddings."""
        u_vec = self.U.weight[u_idx]   # (k,)
        return self.V.weight @ u_vec   # (N,)


# ---------------------------------------------------------------------------
# Section 3 — Training
# ---------------------------------------------------------------------------

def _build_triplets(R: np.ndarray, neg_ratio: float = 1.0):
    """
    Build (i, j, label) triplets.
    neg_ratio controls how many negatives to sample per positive.
    Without negatives the model collapses to predicting 1 everywhere.
    """
    pos_i, pos_j = np.where(R == 1)
    n_pos = len(pos_i)

    neg_i, neg_j = np.where(R == 0)
    n_neg = int(n_pos * neg_ratio)
    choice = np.random.choice(len(neg_i), size=n_neg, replace=False)
    neg_i, neg_j = neg_i[choice], neg_j[choice]

    all_i = np.concatenate([pos_i, neg_i]).astype(np.int64)
    all_j = np.concatenate([pos_j, neg_j]).astype(np.int64)
    all_r = np.concatenate([
        np.ones(n_pos, dtype=np.float32),
        np.zeros(n_neg, dtype=np.float32),
    ])

    return all_i, all_j, all_r


def train_mf(
    R: np.ndarray,
    k: int = 16,
    epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = 512,
    neg_ratio: float = 1.0,
    desc: str = "",
) -> MFModel:
    N = R.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_i, all_j, all_r = _build_triplets(R, neg_ratio)
    dataset = TensorDataset(
        torch.from_numpy(all_i),
        torch.from_numpy(all_j),
        torch.from_numpy(all_r),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MFModel(N, k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    pbar = tqdm(range(epochs), desc=desc or "Training", ncols=80)
    for _ in pbar:
        total_loss = 0.0
        for u, v, r in loader:
            u, v, r = u.to(device), v.to(device), r.to(device)
            loss = loss_fn(model(u, v), r)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        pbar.set_postfix(loss=f"{total_loss / len(loader):.4f}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Section 4 — Recommendation
# ---------------------------------------------------------------------------

def recommend(
    model: MFModel,
    query_idx: int,
    track_ids: list,
    k: int = 10,
) -> list:
    """Return top-k track_ids (excluding the query itself)."""
    with torch.no_grad():
        scores = model.score_all(query_idx).cpu().numpy()

    scores[query_idx] = -np.inf
    top_indices = np.argsort(scores)[::-1][:k]
    return [track_ids[i] for i in top_indices]


# ---------------------------------------------------------------------------
# Section 5 — Comparison Utility
# ---------------------------------------------------------------------------

def compare_recommendations(
    model_tag: MFModel,
    model_va: MFModel,
    query_track_id,
    track_ids: list,
    labels: pd.DataFrame,
    k: int = 10,
):
    query_idx = track_ids.index(query_track_id)
    recs_tag  = recommend(model_tag, query_idx, track_ids, k)
    recs_va   = recommend(model_va,  query_idx, track_ids, k)

    tag_cols = ["energetic", "tense", "calm", "lyrical"]

    def fmt_track(tid):
        row  = labels[labels["track_id"] == tid].iloc[0]
        tags = "+".join(c for c in tag_cols if row[c] == 1) or "none"
        return f"  id={tid:<4} [{tags:<20}] V={row['valence']:.2f} A={row['arousal']:.2f}  {row['title'][:40]}"

    query_row = labels[labels["track_id"] == query_track_id].iloc[0]
    q_tags = "+".join(c for c in tag_cols if query_row[c] == 1) or "none"
    print(f"\n{'='*80}")
    print(f"Query track: id={query_track_id}  [{q_tags}]  V={query_row['valence']:.2f} A={query_row['arousal']:.2f}")
    print(f"  Title: {query_row['title']}")
    print(f"{'='*80}")
    print(f"\n{'--- Case 1: Tag Overlap (expect same-cluster results) ---':^80}")
    for tid in recs_tag:
        print(fmt_track(tid))
    print(f"\n{'--- Case 2: VA Distance  (expect smoother, cross-tag results) ---':^80}")
    for tid in recs_va:
        print(fmt_track(tid))
    print()


# ---------------------------------------------------------------------------
# Section 6 — Visualization
# ---------------------------------------------------------------------------

TAG_COLORS = {
    "energetic": "#e74c3c",
    "tense":     "#9b59b6",
    "calm":      "#2ecc71",
    "lyrical":   "#3498db",
    "mixed":     "#95a5a6",
}
TAG_COLS = ["energetic", "tense", "calm", "lyrical"]


def _dominant_tag(row) -> str:
    active = [c for c in TAG_COLS if row[c] == 1]
    if len(active) == 1:
        return active[0]
    if len(active) > 1:
        return "mixed"
    return "mixed"


def visualize_embeddings(
    model_tag: MFModel,
    model_va: MFModel,
    track_ids: list,
    labels: pd.DataFrame,
    save_path: str = None,
):
    """
    PCA on U embeddings from both models.
    Tag model → tight clusters (geometry mirrors block structure).
    VA model  → smoother spread (geometry mirrors continuous similarity).
    """
    labels_indexed = labels.set_index("track_id")
    ordered_labels = labels_indexed.loc[track_ids]
    colors = [TAG_COLORS[_dominant_tag(row)] for _, row in ordered_labels.iterrows()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    pca = PCA(n_components=2)

    for ax, model, title in zip(
        axes,
        [model_tag, model_va],
        ["Case 1: Tag Overlap", "Case 2: VA Distance (t=0.95)"],
    ):
        emb    = model.U.weight.detach().cpu().numpy()  # (N, k)
        coords = pca.fit_transform(emb)                  # (N, 2)

        for tag, color in TAG_COLORS.items():
            mask = [c == color for c in colors]
            if any(mask):
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=color, label=tag, alpha=0.75, s=40, edgecolors="none",
                )

        ax.set_title(title, fontsize=13)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    handles = [mpatches.Patch(color=c, label=t) for t, c in TAG_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("MF Embedding Geometry by Signal Design\n"
                 "(Tag → clusters | VA distance → gradient)", fontsize=14, y=1.01)
    plt.tight_layout()

    path = save_path or os.path.join(FIG_DIR, "mf_embeddings.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nEmbedding plot saved → {path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading data...")
    R_tag, R_va, track_ids, labels = load_data()
    N = len(track_ids)
    print(f"  N={N} tracks  |  R_tag density={R_tag.mean():.3f}  R_va density={R_va.mean():.3f}")

    model_tag = train_mf(R_tag, k=16, epochs=100, lr=0.01, desc="Case 1 (Tag Overlap)")
    model_va  = train_mf(R_va,  k=16, epochs=100, lr=0.01, desc="Case 2 (VA Distance) ")

    example_ids = [track_ids[0], track_ids[50], track_ids[100]]
    for tid in example_ids:
        compare_recommendations(model_tag, model_va, tid, track_ids, labels, k=10)

    visualize_embeddings(model_tag, model_va, track_ids, labels)
