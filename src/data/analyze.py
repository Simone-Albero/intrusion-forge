from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import paired_distances
from sklearn.decomposition import PCA

from src.ml.projection import tsne_projection, create_subsample_mask
from src.plot.array import samples_plot


def sample_distances(
    X: np.ndarray,
    idx_a: npt.ArrayLike,
    idx_b: npt.ArrayLike | None = None,
    n_pairs: int = 200_000,
    metric: str = "cosine",
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

    pca = PCA(n_components=0.95, svd_solver="full", whiten=True)
    X = pca.fit_transform(X)

    if idx_b is None:  # Intra-class distances
        if len(idx_a) < 2:
            return np.array([])

        max_pairs = len(idx_a) * (len(idx_a) - 1) // 2
        n_samples = min(n_pairs, max_pairs)

        if n_samples == 0:
            return np.array([])

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
        if n_samples == 0:
            return np.array([])

        replace = n_samples > min(len(idx_a), len(idx_b))

        i = np.random.choice(len(idx_a), size=n_samples, replace=replace)
        j = np.random.choice(len(idx_b), size=n_samples, replace=replace)

        return paired_distances(X[idx_a[i]], X[idx_b[j]], metric=metric)


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


def compute_class_separability(
    X: np.ndarray, y: np.ndarray, n_pairs: int = 50_000, metric: str = "cosine"
) -> list[dict[str, Any]]:
    """Analyze class separability using pairwise intra/inter-class distances.

    For each class, aggregates metrics across all pairs involving that class.
    Returns a list of dicts (one per class), each with:
      class:        class label
      pairs:         dict of other class -> {roc, ratio}
      mean_roc:     mean roc in [0, 1] across all pairs involving this class
      min_roc:      lowest roc among all pairs involving this class
      mean_ratio:   mean intra/inter ratio
      min_ratio:    lowest ratio among all pairs involving this class
    """
    classes = np.unique(y)

    pair_metrics: dict[tuple, dict] = {}
    for i, class_a in enumerate(classes):
        for class_b in classes[i + 1 :]:
            idx_a = np.where(y == class_a)[0]
            idx_b = np.where(y == class_b)[0]

            intra_a = sample_distances(X, idx_a, n_pairs=n_pairs, metric=metric)
            intra_b = sample_distances(X, idx_b, n_pairs=n_pairs, metric=metric)
            inter_ab = sample_distances(X, idx_a, idx_b, n_pairs=n_pairs, metric=metric)

            intra_a_mean = np.mean(intra_a) if len(intra_a) > 0 else np.nan
            intra_b_mean = np.mean(intra_b) if len(intra_b) > 0 else np.nan
            inter_ab_mean = np.mean(inter_ab) if len(inter_ab) > 0 else np.nan

            all_finite = (
                np.isfinite(intra_a_mean)
                and np.isfinite(intra_b_mean)
                and np.isfinite(inter_ab_mean)
            )
            intra_mean = (intra_a_mean + intra_b_mean) / 2
            ratio = intra_mean / inter_ab_mean if all_finite else np.nan

            raw_roc = distance_roc_auc(intra_a, intra_b, inter_ab)

            pair_metrics[(str(class_a), str(class_b))] = {
                "roc": raw_roc,
                "ratio": ratio,
            }

    results = []
    for cls in classes:
        cls = str(cls)
        pairs = {
            other: m
            for (a, b), m in pair_metrics.items()
            for other in ([b] if a == cls else [a] if b == cls else [])
        }

        rocs = np.array([m["roc"] for m in pairs.values()])
        ratios = np.array([m["ratio"] for m in pairs.values()])

        results.append(
            {
                "class": cls,
                "pairs": {
                    other: {
                        "ratio": float(m["ratio"]),
                        "roc": float(m["roc"]),
                    }
                    for other, m in sorted(
                        pairs.items(), key=lambda x: x[1]["ratio"], reverse=True
                    )
                },
                "mean_ratio": float(np.nanmean(ratios)),
                "mean_roc": float(np.nanmean(rocs)),
                "max_ratio": float(np.nanmax(ratios)),
                "min_roc": float(np.nanmin(rocs)),
            }
        )

    results.sort(key=lambda x: x["mean_roc"])
    return results
