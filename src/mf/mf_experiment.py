"""
mf_experiment.py — Signal Design × Matrix Factorization
========================================================
Implements the basic MF model from Koren et al. (2009), §"A Basic Matrix
Factorization Model", trained on two pseudo-interaction matrices to compare
how signal design affects embedding geometry and recommendation behavior.

Notation follows the paper throughout:
  N     — number of tracks (acting as both users and items; item-to-item setup)
  f     — latent factor dimensionality (called 'k' in some implementations)
  p_u   — user latent factor vector  ∈ ℝ^f  (P embedding in code)
  q_i   — item latent factor vector  ∈ ℝ^f  (Q embedding in code)
  r_ui  — observed interaction (0 or 1 in our binary proxy)
  r̂_ui  — predicted rating: q_i^T p_u  (Eq. 1)
  κ     — set of known (u, i) pairs used for training  (Eq. 2)

Departure from the paper: there are no real users here. Both rows and columns
of R represent tracks, so p_u and q_i both index tracks. This is an
item-to-item collaborative filtering setup; the paper's user/item asymmetry
does not apply.

Usage (from project root):
    PYTHONPATH=src .venv/bin/python src/mf/mf_experiment.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR  = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

TAG_COLS = ["energetic", "tense", "calm", "lyrical"]

TAG_COLORS = {
    "energetic": "#e74c3c",
    "tense":     "#9b59b6",
    "calm":      "#2ecc71",
    "lyrical":   "#3498db",
    "mixed":     "#95a5a6",
}


# =============================================================================
# Section 1 — Data Loading
# =============================================================================

def load_data():
    """Load both interaction matrices and track metadata."""
    labels = pd.read_csv(os.path.join(DATA_DIR, "pseudo_labels.csv"))

    # R_tag, R_va are the observed interaction matrices r_ui (binary proxy)
    R_tag = pd.read_csv(os.path.join(DATA_DIR, "matrix_case1_tag_overlap.csv"),
                        index_col="track_id")
    R_va  = pd.read_csv(os.path.join(DATA_DIR, "matrix_case2_va_distance_t095.csv"),
                        index_col="track_id")

    track_ids = R_tag.index.tolist()
    return (
        R_tag.values.astype(np.float32),  # N×N observed ratings matrix
        R_va.values.astype(np.float32),
        track_ids,
        labels,
    )


# =============================================================================
# Section 2 — MF Model  (Koren et al. Eq. 1)
# =============================================================================

class MFModel(nn.Module):
    """
    Basic MF model: r̂_ui = q_i^T p_u  (Eq. 1, Koren et al. 2009)

    P[u] = p_u ∈ ℝ^f  — latent factor vector for "user" (query track)
    Q[i] = q_i ∈ ℝ^f  — latent factor vector for "item" (candidate track)

    Both P and Q have shape (N, f) because this is an item-to-item setup.
    """
    def __init__(self, N: int, f: int):
        super().__init__()
        self.P = nn.Embedding(N, f)   # p_u vectors
        self.Q = nn.Embedding(N, f)   # q_i vectors
        nn.init.normal_(self.P.weight, std=0.01)
        nn.init.normal_(self.Q.weight, std=0.01)

    def forward(self, u_idx: torch.Tensor, i_idx: torch.Tensor) -> torch.Tensor:
        # r̂_ui = q_i^T p_u  (element-wise product then sum = dot product)
        return (self.P(u_idx) * self.Q(i_idx)).sum(dim=-1)

    def predict_scores(self, u_idx: int) -> torch.Tensor:
        """
        Compute r̂_ui for all items i given a fixed query track u.
        Returns a vector of length N: Q @ p_u
        """
        p_u = self.P.weight[u_idx]   # (f,)
        return self.Q.weight @ p_u   # (N,)


# =============================================================================
# Section 3 — Training Set Construction  (κ in paper)
# =============================================================================

def _sample_observed(R: np.ndarray):
    """
    Extract all known positive interactions — the set κ in Eq. 2.
    Returns (u_idx, i_idx) index arrays for entries where R[u,i] = 1.
    """
    u_idx, i_idx = np.where(R == 1)
    return u_idx, i_idx


def _sample_unobserved(R: np.ndarray, n_samples: int):
    """
    Sample zero entries as negative examples.
    In the paper's implicit feedback framing (§"Additional Input Sources"),
    unobserved entries are treated as expressing no preference — we model
    them as target 0 to prevent the model from collapsing to all-ones.
    """
    neg_u, neg_i = np.where(R == 0)
    choice = np.random.choice(len(neg_u), size=n_samples, replace=False)
    return neg_u[choice], neg_i[choice]


def build_training_set(R: np.ndarray, confidence_ratio: float = 1.0):
    """
    Combine observed (r_ui=1) and unobserved (r_ui=0) entries into training
    triplets (u_idx, i_idx, r_ui). confidence_ratio controls how many
    negatives are sampled per positive (analogous to confidence weighting,
    §"Inputs with Varying Confidence Levels").

    Returns:
        u_idx  (int64 array) — user indices
        i_idx  (int64 array) — item indices
        r_ui   (float32 array) — target values (0 or 1)
    """
    pos_u, pos_i = _sample_observed(R)
    n_pos = len(pos_u)

    neg_u, neg_i = _sample_unobserved(R, int(n_pos * confidence_ratio))

    u_idx = np.concatenate([pos_u, neg_u]).astype(np.int64)
    i_idx = np.concatenate([pos_i, neg_i]).astype(np.int64)
    r_ui  = np.concatenate([
        np.ones(n_pos, dtype=np.float32),
        np.zeros(len(neg_u), dtype=np.float32),
    ])
    return u_idx, i_idx, r_ui


# =============================================================================
# Section 4 — Training  (SGD on Eq. 2)
# =============================================================================

def _init_model(N: int, f: int, device: torch.device) -> tuple:
    """Initialise model, Adam optimizer, and MSE loss (Eq. 2 without regularization)."""
    model     = MFModel(N, f).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # γ (learning rate) in paper
    loss_fn   = nn.MSELoss()
    return model, optimizer, loss_fn


def _make_dataloader(u_idx, i_idx, r_ui, batch_size: int) -> DataLoader:
    """Wrap training arrays in a shuffled DataLoader."""
    dataset = TensorDataset(
        torch.from_numpy(u_idx),
        torch.from_numpy(i_idx),
        torch.from_numpy(r_ui),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def _train_one_epoch(model, loader, optimizer, loss_fn, device) -> float:
    """Run one full pass over the training set. Returns mean batch loss."""
    total_loss = 0.0
    for u, i, r in loader:
        u, i, r = u.to(device), i.to(device), r.to(device)
        loss = loss_fn(model(u, i), r)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_mf(
    R: np.ndarray,
    f: int = 16,
    epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = 512,
    confidence_ratio: float = 1.0,
    desc: str = "",
) -> tuple:
    """
    Train an MF model minimizing Eq. 2 (Koren et al.) via SGD.

    Args:
        R                — N×N observed interaction matrix (r_ui)
        f                — latent factor dimensionality
        epochs           — number of full passes over κ
        lr               — learning rate γ
        confidence_ratio — negatives sampled per positive (confidence weighting proxy)
        desc             — label for tqdm progress bar

    Returns:
        (model, loss_history) — trained MFModel and list of per-epoch mean MSE
    """
    N      = R.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    u_idx, i_idx, r_ui = build_training_set(R, confidence_ratio)
    loader              = _make_dataloader(u_idx, i_idx, r_ui, batch_size)
    model, optimizer, loss_fn = _init_model(N, f, device)

    # Override lr if different from default
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    loss_history = []
    pbar = tqdm(range(epochs), desc=desc or "Training", ncols=80)
    for _ in pbar:
        epoch_loss = _train_one_epoch(model, loader, optimizer, loss_fn, device)
        loss_history.append(epoch_loss)
        pbar.set_postfix(loss=f"{epoch_loss:.4f}")

    model.eval()
    return model, loss_history


# =============================================================================
# Section 5 — Recommendation  (r̂_ui = q_i^T p_u, Eq. 1)
# =============================================================================

def recommend(
    model: MFModel,
    u_idx: int,
    track_ids: list,
    k: int = 10,
) -> list:
    """
    Return top-k track_ids for query track u_idx (excluding itself).
    Scores are r̂_ui = Q @ p_u for all i (Eq. 1).
    """
    with torch.no_grad():
        r_hat_u = model.predict_scores(u_idx).cpu().numpy()   # r̂_ui for all i

    r_hat_u[u_idx] = -np.inf   # exclude the query track itself
    top_indices = np.argsort(r_hat_u)[::-1][:k]
    return [track_ids[idx] for idx in top_indices]


# =============================================================================
# Section 6 — Recommendation Comparison
# =============================================================================

def _format_track(tid, labels: pd.DataFrame) -> str:
    """Format one track as a readable summary line."""
    row  = labels[labels["track_id"] == tid].iloc[0]
    tags = "+".join(c for c in TAG_COLS if row[c] == 1) or "none"
    return (f"  id={tid:<4} [{tags:<20}]"
            f" V={row['valence']:.2f} A={row['arousal']:.2f}"
            f"  {row['title'][:40]}")


def compare_recommendations(
    model_tag: MFModel,
    model_va: MFModel,
    query_track_id,
    track_ids: list,
    labels: pd.DataFrame,
    k: int = 10,
):
    """
    Print top-k recommendations for the same query track from both models,
    side by side, so you can see cluster-like (tag) vs. smooth (VA) behavior.
    """
    u_idx    = track_ids.index(query_track_id)
    recs_tag = recommend(model_tag, u_idx, track_ids, k)
    recs_va  = recommend(model_va,  u_idx, track_ids, k)

    query_row = labels[labels["track_id"] == query_track_id].iloc[0]
    q_tags    = "+".join(c for c in TAG_COLS if query_row[c] == 1) or "none"

    print(f"\n{'='*80}")
    print(f"Query: id={query_track_id}  [{q_tags}]"
          f"  V={query_row['valence']:.2f} A={query_row['arousal']:.2f}")
    print(f"  {query_row['title']}")
    print(f"{'='*80}")
    print(f"\n{'--- Case 1: Tag Overlap (expect same-cluster results) ---':^80}")
    for tid in recs_tag:
        print(_format_track(tid, labels))
    print(f"\n{'--- Case 2: VA Distance (expect smoother, cross-tag results) ---':^80}")
    for tid in recs_va:
        print(_format_track(tid, labels))
    print()


# =============================================================================
# Section 7 — Training Curve Plot
# =============================================================================

def plot_training_curves(
    loss_tag: list,
    loss_va: list,
    save_path: str = None,
):
    """
    Plot per-epoch MSE loss for both models.
    A smooth, decreasing curve means the model is learning; a flat curve
    means it converged quickly (or is stuck). Comparing the two cases shows
    which signal is easier to fit.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(loss_tag, label="Case 1: Tag Overlap",   color="#264653", linewidth=1.8)
    ax.plot(loss_va,  label="Case 2: VA Distance",   color="#2A9D8F", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("MF Training Loss — Tag Overlap vs VA Distance")
    ax.legend(fontsize=10)
    ax.set_facecolor("#F8F8F8")
    fig.patch.set_facecolor("#F8F8F8")

    path = save_path or os.path.join(FIG_DIR, "mf_training_curves.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Training curve saved → {path}")
    plt.close(fig)


# =============================================================================
# Section 8 — Embedding Visualization (PCA on p_u, Eq. 1 geometry)
# =============================================================================

def _get_point_colors(track_ids: list, labels: pd.DataFrame) -> list:
    """Assign a TAG_COLORS color to each track based on its dominant tag."""
    labels_idx = labels.set_index("track_id").loc[track_ids]
    def _color(row):
        active = [c for c in TAG_COLS if row[c] == 1]
        tag = active[0] if len(active) == 1 else "mixed"
        return TAG_COLORS[tag]
    return [_color(row) for _, row in labels_idx.iterrows()]


def _plot_one_embedding(ax, model: MFModel, colors: list, title: str):
    """
    Project p_u vectors (P embedding) to 2D via PCA and scatter-plot.
    The geometry of this plot reflects the structure of the interaction matrix:
    block matrix → tight clusters; smooth matrix → gradient spread.
    """
    pca    = PCA(n_components=2)
    emb    = model.P.weight.detach().cpu().numpy()   # p_u matrix (N, f)
    coords = pca.fit_transform(emb)                   # (N, 2)

    for tag, color in TAG_COLORS.items():
        mask = np.array([c == color for c in colors])
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=color, label=tag, alpha=0.75, s=40, edgecolors="none")

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")


def visualize_embeddings(
    model_tag: MFModel,
    model_va: MFModel,
    track_ids: list,
    labels: pd.DataFrame,
    save_path: str = None,
):
    """
    Side-by-side PCA plots of the learned p_u latent factors.
    Tag model → tight clusters (block structure reproduced in embedding space).
    VA model  → smoother gradient (continuous similarity reflected in geometry).
    """
    colors = _get_point_colors(track_ids, labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _plot_one_embedding(axes[0], model_tag, colors, "Case 1: Tag Overlap")
    _plot_one_embedding(axes[1], model_va,  colors, "Case 2: VA Distance (t=0.95)")

    handles = [mpatches.Patch(color=c, label=t) for t, c in TAG_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Latent Factor Geometry (PCA on p_u, Eq. 1)\n"
                 "Tag → clusters  |  VA distance → gradient", fontsize=13, y=1.01)
    plt.tight_layout()

    path = save_path or os.path.join(FIG_DIR, "mf_embeddings.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Embedding plot saved → {path}")
    plt.close(fig)


# =============================================================================
# Section 9 — K-Fold Cross-Validation
# =============================================================================

def _get_cv_entries(R: np.ndarray):
    """
    Extract upper-triangle positive entries to use as CV data points.
    Upper triangle only to avoid leaking symmetric pairs (R[i,j] == R[j,i])
    across train/test splits.
    Returns (row_idx, col_idx) arrays.
    """
    rows, cols = np.where(np.triu(R, k=1) == 1)
    return rows, cols


def _evaluate_fold(model: MFModel, R: np.ndarray,
                   test_u: np.ndarray, test_i: np.ndarray,
                   device: torch.device) -> float:
    """
    Compute MSE on held-out positive entries + equal number of sampled negatives.
    Returns scalar MSE.
    """
    # Positives
    n_test = len(test_u)

    # Sample negatives of equal size from zero entries
    neg_u, neg_i = np.where(R == 0)
    choice = np.random.choice(len(neg_u), size=n_test, replace=False)
    eval_u = np.concatenate([test_u, neg_u[choice]]).astype(np.int64)
    eval_i = np.concatenate([test_i, neg_i[choice]]).astype(np.int64)
    eval_r = np.concatenate([
        np.ones(n_test,  dtype=np.float32),
        np.zeros(n_test, dtype=np.float32),
    ])

    with torch.no_grad():
        u_t = torch.from_numpy(eval_u).to(device)
        i_t = torch.from_numpy(eval_i).to(device)
        r_t = torch.from_numpy(eval_r).to(device)
        preds = model(u_t, i_t)
        mse = nn.MSELoss()(preds, r_t).item()
    return mse


def kfold_cv(
    R: np.ndarray,
    f: int = 16,
    epochs: int = 100,
    lr: float = 0.01,
    n_splits: int = 5,
    desc: str = "",
) -> dict:
    """
    K-Fold cross-validation on the observed positive entries.

    For each fold:
      1. Hold out a subset of positive (u,i) entries
      2. Train MF on the remaining interactions
      3. Measure MSE on held-out positives + equal negatives

    Returns dict with keys: mse_per_fold, mean, std
    """
    rows, cols = _get_cv_entries(R)
    kf         = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mse_per_fold = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(rows)):
        # Build R_train: mask out held-out entries (both directions, symmetric)
        R_train = R.copy()
        R_train[rows[test_idx], cols[test_idx]] = 0
        R_train[cols[test_idx], rows[test_idx]] = 0

        fold_desc = f"{desc} fold {fold_idx+1}/{n_splits}"
        model, _ = train_mf(R_train, f=f, epochs=epochs, lr=lr, desc=fold_desc)

        mse = _evaluate_fold(model, R_train,
                             rows[test_idx], cols[test_idx], device)
        mse_per_fold.append(mse)

    return {
        "mse_per_fold": mse_per_fold,
        "mean": float(np.mean(mse_per_fold)),
        "std":  float(np.std(mse_per_fold)),
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # --- Load ---
    print("Loading data...")
    R_tag, R_va, track_ids, labels = load_data()
    N = len(track_ids)
    print(f"  N={N} tracks  |  R_tag density={R_tag.mean():.3f}"
          f"  R_va density={R_va.mean():.3f}")

    # --- Train (returns model + per-epoch loss history) ---
    model_tag, loss_tag = train_mf(R_tag, f=16, epochs=100, lr=0.01,
                                   desc="Case 1 (Tag Overlap)")
    model_va,  loss_va  = train_mf(R_va,  f=16, epochs=100, lr=0.01,
                                   desc="Case 2 (VA Distance)")

    # --- Training loss plot ---
    plot_training_curves(loss_tag, loss_va)

    # --- Recommendations for 3 example tracks ---
    for tid in [track_ids[0], track_ids[50], track_ids[100]]:
        compare_recommendations(model_tag, model_va, tid, track_ids, labels, k=10)

    # --- Embedding geometry ---
    visualize_embeddings(model_tag, model_va, track_ids, labels)

    # --- K-Fold CV (5 folds, fewer epochs for speed) ---
    print("\nRunning 5-fold CV (this trains 10 models)...")
    cv_tag = kfold_cv(R_tag, f=16, epochs=50, lr=0.01, n_splits=5,
                      desc="CV Tag")
    cv_va  = kfold_cv(R_va,  f=16, epochs=50, lr=0.01, n_splits=5,
                      desc="CV VA ")

    print(f"\n{'='*50}")
    print(f"  K-Fold CV Results (5 folds, f=16, epochs=50)")
    print(f"{'='*50}")
    print(f"  Case 1 Tag Overlap:  MSE = {cv_tag['mean']:.4f} ± {cv_tag['std']:.4f}")
    print(f"  Case 2 VA Distance:  MSE = {cv_va['mean']:.4f}  ± {cv_va['std']:.4f}")
    print(f"{'='*50}")
    print("  Lower MSE = signal is more consistently learnable by MF")
