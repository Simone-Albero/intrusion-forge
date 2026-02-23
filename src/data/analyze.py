from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.metrics.pairwise import paired_distances

from src.ml.projection import tsne_projection, create_subsample_mask
from src.plot.array import samples_plot


def sample_distances(
    X: np.ndarray,
    idx_a: npt.ArrayLike,
    idx_b: npt.ArrayLike | None = None,
    n_pairs: int = 200_000,
    metric: str = "euclidean",
) -> np.ndarray:
    """Sample pairwise distances between points.

    Args:
        X: Feature matrix (n_samples, n_features)
        idx_a: Indices for first set of points
        idx_b: Indices for second set (None for intra-class distances)
        n_pairs: Maximum number of distance pairs to compute

    Returns:
        Array of pairwise distances between sampled pairs
    """
    idx_a = np.asarray(idx_a)

    if idx_b is None:  # Intra-class distances
        if len(idx_a) < 2:
            return np.array([])

        max_pairs = len(idx_a) * (len(idx_a) - 1) // 2
        n_samples = min(n_pairs, max_pairs)

        if n_samples < max_pairs:
            # Oversample slightly to absorb i==j collisions, then trim
            size = int(n_samples * 1.1) + 32
            i = np.random.randint(0, len(idx_a), size=size)
            j = np.random.randint(0, len(idx_a), size=size)
            valid = i != j
            i, j = i[valid][:n_samples], j[valid][:n_samples]
        else:
            from itertools import combinations

            pairs = np.array(list(combinations(range(len(idx_a)), 2)))
            i, j = pairs[:, 0], pairs[:, 1]

        return paired_distances(X[idx_a[i]], X[idx_a[j]], metric=metric)

    else:  # Inter-class distances
        idx_b = np.asarray(idx_b)
        if len(idx_a) == 0 or len(idx_b) == 0:
            return np.array([])

        n_samples = min(n_pairs, len(idx_a) * len(idx_b))
        replace = n_samples > min(len(idx_a), len(idx_b))

        i = np.random.choice(len(idx_a), size=n_samples, replace=replace)
        j = np.random.choice(len(idx_b), size=n_samples, replace=replace)

        return paired_distances(X[idx_a[i]], X[idx_b[j]], metric=metric)


def compute_class_separability(X: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
    """Analyze class separability using intra/inter-class distances."""
    results = []
    for class_name in np.unique(y):
        idx_class = np.where(y == class_name)[0]
        idx_other = np.where(y != class_name)[0]

        intra_dist = sample_distances(X, idx_class)
        inter_dist = sample_distances(X, idx_class, idx_other)

        intra_mean = np.mean(intra_dist) if len(intra_dist) > 0 else np.nan
        inter_mean = np.mean(inter_dist) if len(inter_dist) > 0 else np.nan

        both_finite = np.isfinite(intra_mean) and np.isfinite(inter_mean)
        gap = inter_mean - intra_mean if both_finite else np.nan
        ratio = intra_mean / inter_mean if both_finite else np.nan

        try:
            sil = silhouette_score(X, y == class_name, sample_size=min(50_000, len(X)))
        except ValueError:
            sil = np.nan

        results.append(
            {
                "class": str(class_name),
                "n_samples": int(idx_class.size),
                "intra_mean": float(intra_mean),
                "inter_mean": float(inter_mean),
                "gap": float(gap),
                "ratio": float(ratio),
                "silhouette_score": float(sil),
            }
        )

    results.sort(key=lambda x: x["ratio"])
    return results


def distance_roc_auc(
    intra_a: np.ndarray, intra_b: np.ndarray, inter_ab: np.ndarray
) -> float:
    """Compute ROC-AUC using distance as score.

    Positive (1): inter-class pairs. Negative (0): intra-class pairs.
    AUC=1 means inter-class distances are always larger than intra-class.
    """
    y_true = np.concatenate(
        [np.zeros(len(intra_a) + len(intra_b)), np.ones(len(inter_ab))]
    )
    y_scores = np.concatenate([intra_a, intra_b, inter_ab])

    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_scores)


def compute_class_similarity(
    X: np.ndarray,
    idx_a: npt.ArrayLike,
    idx_b: npt.ArrayLike,
    n_pairs: int = 200_000,
) -> dict[str, float]:
    """Compute similarity between two classes based on distance distributions.

    Returns a dict with:
      intra_a_mean:  low → class A is compact
      intra_b_mean:  low → class B is compact
      inter_ab_mean: high → classes are well separated
      gap:           negative → classes are closer to each other than internally
      ratio:         low → good separation
      overlap_a:     P(inter < intra_a); high → class A overlaps with class B
      overlap_b:     P(inter < intra_b); high → class B overlaps with class A
      roc_auc:       high → good separation
    """
    intra_a = sample_distances(X, idx_a, n_pairs=n_pairs)
    intra_b = sample_distances(X, idx_b, n_pairs=n_pairs)
    inter_ab = sample_distances(X, idx_a, idx_b, n_pairs=n_pairs)

    intra_a_mean = np.mean(intra_a) if len(intra_a) > 0 else np.nan
    intra_b_mean = np.mean(intra_b) if len(intra_b) > 0 else np.nan
    inter_ab_mean = np.mean(inter_ab) if len(inter_ab) > 0 else np.nan

    all_finite = (
        np.isfinite(intra_a_mean)
        and np.isfinite(intra_b_mean)
        and np.isfinite(inter_ab_mean)
    )
    intra_mean = (intra_a_mean + intra_b_mean) / 2
    gap = inter_ab_mean - intra_mean if all_finite else np.nan
    ratio = intra_mean / inter_ab_mean if all_finite else np.nan

    overlap_a = np.mean(inter_ab[:, None] < intra_a[None, :])
    overlap_b = np.mean(inter_ab[:, None] < intra_b[None, :])

    return {
        "intra_a_mean": float(intra_a_mean),
        "intra_b_mean": float(intra_b_mean),
        "inter_ab_mean": float(inter_ab_mean),
        "gap": float(gap),
        "ratio": float(ratio),
        "overlap_a": float(overlap_a),
        "overlap_b": float(overlap_b),
        "roc_auc": float(distance_roc_auc(intra_a, intra_b, inter_ab)),
    }


def visualize(
    X: np.ndarray,
    y: np.ndarray,
    exclude_classes: list | None = None,
    n_samples: int = 3000,
) -> Any:
    """Create a t-SNE visualization, optionally excluding specific classes."""
    mask = ~np.isin(y, exclude_classes or [])
    vis_mask = create_subsample_mask(y[mask], n_samples=n_samples, stratify=False)
    reduced_x = tsne_projection(X[mask][vis_mask])
    return samples_plot(reduced_x, y[mask][vis_mask])
