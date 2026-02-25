from itertools import combinations
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import roc_auc_score, pairwise_distances
from sklearn.metrics.pairwise import paired_distances
from sklearn.decomposition import PCA

from src.ml.projection import tsne_projection, create_subsample_mask
from src.plot.array import samples_plot


def _pairwise_sample(
    X: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray | None,
    max_samples: int,
    metric: str,
    chunk_size: int = 256,
) -> np.ndarray:
    """Sample distances row-by-row (chunked) to avoid materialising a full N×N matrix."""
    collected: list[np.ndarray] = []
    budget = max_samples

    idx_a = idx_a[np.random.permutation(len(idx_a))]
    if idx_b is not None:
        idx_b = idx_b[np.random.permutation(len(idx_b))]

    if idx_b is None:  # intra-class: upper-triangle pairs
        n = len(idx_a)
        for start in range(0, n - 1, chunk_size):
            if budget <= 0:
                break
            end = min(start + chunk_size, n - 1)
            D = pairwise_distances(
                X[idx_a[start:end]], X[idx_a[start + 1 :]], metric=metric
            )
            flat = [
                D[li, max(gi + 1 - (start + 1), 0) :]
                for li, gi in enumerate(range(start, end))
            ]
            chunk = np.concatenate(flat) if flat else np.array([])
            if len(chunk) > budget:
                chunk = chunk[np.random.choice(len(chunk), size=budget, replace=False)]
            collected.append(chunk)
            budget -= len(chunk)
    else:  # inter-class
        for start in range(0, len(idx_a), chunk_size):
            if budget <= 0:
                break
            end = min(start + chunk_size, len(idx_a))
            D = pairwise_distances(X[idx_a[start:end]], X[idx_b], metric=metric).ravel()
            if len(D) > budget:
                D = D[np.random.choice(len(D), size=budget, replace=False)]
            collected.append(D)
            budget -= len(D)

    return np.concatenate(collected) if collected else np.array([])


def _paired_sample(
    X: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray | None,
    n_samples: int | None,
    metric: str,
) -> np.ndarray:
    """Sample distances via random 1-to-1 index pairs."""
    if idx_b is None:
        max_pairs = len(idx_a) * (len(idx_a) - 1) // 2
        if n_samples is None or n_samples >= max_pairs:
            pairs = np.array(list(combinations(range(len(idx_a)), 2)))
            i, j = pairs[:, 0], pairs[:, 1]
        else:
            size = int(n_samples * 1.1) + 32
            i = np.random.randint(0, len(idx_a), size=size)
            j = np.random.randint(0, len(idx_a), size=size)
            mask = i != j
            i, j = i[mask][:n_samples], j[mask][:n_samples]
        return paired_distances(X[idx_a[i]], X[idx_a[j]], metric=metric)
    else:
        max_pairs = len(idx_a) * len(idx_b)
        use_all = n_samples is None or n_samples >= max_pairs
        n = max_pairs if use_all else n_samples
        replace = not use_all and n_samples > min(len(idx_a), len(idx_b))
        i = np.random.choice(len(idx_a), size=n, replace=replace)
        j = np.random.choice(len(idx_b), size=n, replace=replace)
        return paired_distances(X[idx_a[i]], X[idx_b[j]], metric=metric)


def sample_distances(
    X: np.ndarray,
    idx_a: npt.ArrayLike,
    idx_b: npt.ArrayLike | None = None,
    max_pairs: int | None = 200_000,
    metric: str = "cosine",
    pairwise: bool = False,
) -> np.ndarray:
    """Sample pairwise distances between two sets of points.

    Args:
        X: Feature matrix (n_samples, n_features).
        idx_a: Indices for the first set of points.
        idx_b: Indices for the second set (None for intra-class distances).
        max_pairs: Maximum number of distance pairs to sample.
        metric: Distance metric to use.
        pairwise: If True, use chunked row iteration (memory-efficient).
                  If False, draw random 1-to-1 index pairs.
    """
    idx_a = np.asarray(idx_a)
    idx_b = np.asarray(idx_b) if idx_b is not None else None

    if idx_b is None:
        if len(idx_a) < 2:
            return np.array([])
        total_pairs = len(idx_a) * (len(idx_a) - 1) // 2
    else:
        if len(idx_a) == 0 or len(idx_b) == 0:
            return np.array([])
        total_pairs = len(idx_a) * len(idx_b)

    n_samples = total_pairs if max_pairs is None else min(max_pairs, total_pairs)
    if n_samples == 0:
        return np.array([])

    if pairwise:
        return _pairwise_sample(X, idx_a, idx_b, n_samples, metric)
    return _paired_sample(X, idx_a, idx_b, n_samples, metric)


def distance_roc_auc(
    intra_a: np.ndarray, intra_b: np.ndarray, inter_ab: np.ndarray
) -> float:
    """ROC-AUC using distance as score: AUC=1 means inter-class distances always exceed intra-class."""
    y_true = np.concatenate(
        [np.zeros(len(intra_a) + len(intra_b)), np.ones(len(inter_ab))]
    )
    y_scores = np.concatenate([intra_a, intra_b, inter_ab])
    return roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) >= 2 else np.nan


def visualize(
    X: np.ndarray,
    y: np.ndarray,
    exclude_classes: list | None = None,
    n_samples: int = 3000,
) -> Any:
    """Create a t-SNE visualization, optionally excluding specific classes."""
    mask = ~np.isin(y, exclude_classes or [])
    vis_mask = create_subsample_mask(y[mask], n_samples=n_samples, stratify=False)
    return samples_plot(tsne_projection(X[mask][vis_mask]), y[mask][vis_mask])


def compute_class_separability(
    X: np.ndarray,
    y: np.ndarray,
    max_pairs: int = 50_000,
    pairwise: bool = False,
    metric: str = "cosine",
    pca_variance: float = 0.95,
) -> list[dict[str, Any]]:
    """Analyze class separability using pairwise intra/inter-class distances.

    Returns a list of dicts (one per class) with keys:
      class, pairs, mean_ratio, max_ratio.
    """
    classes = np.unique(y)

    # Reduce once so sample_distances never operates on the raw high-dim matrix.
    X = PCA(n_components=pca_variance, svd_solver="full", whiten=True).fit_transform(X)

    pair_metrics: dict[tuple, dict] = {}
    for i, class_a in enumerate(classes):
        for class_b in classes[i + 1 :]:
            idx_a = np.where(y == class_a)[0]
            idx_b = np.where(y == class_b)[0]

            intra_a = sample_distances(
                X, idx_a, max_pairs=max_pairs, metric=metric, pairwise=pairwise
            )
            intra_b = sample_distances(
                X, idx_b, max_pairs=max_pairs, metric=metric, pairwise=pairwise
            )
            inter_ab = sample_distances(
                X, idx_a, idx_b, max_pairs=max_pairs, metric=metric, pairwise=pairwise
            )

            intra_mean = (
                np.mean([np.mean(intra_a), np.mean(intra_b)])
                if len(intra_a) > 0 and len(intra_b) > 0
                else np.nan
            )
            inter_mean = np.mean(inter_ab) if len(inter_ab) > 0 else np.nan
            ratio = (
                intra_mean / inter_mean
                if np.isfinite(intra_mean) and np.isfinite(inter_mean)
                else np.nan
            )

            pair_metrics[(str(class_a), str(class_b))] = {"ratio": ratio}

    results = []
    for cls in map(str, classes):
        pairs = {
            other: m
            for (a, b), m in pair_metrics.items()
            for other in ([b] if a == cls else [a] if b == cls else [])
        }

        ratios = np.array([m["ratio"] for m in pairs.values()])

        results.append(
            {
                "class": cls,
                "pairs": {
                    other: {"ratio": float(m["ratio"])}
                    for other, m in sorted(
                        pairs.items(), key=lambda x: x[1]["ratio"], reverse=True
                    )
                },
                "mean_ratio": float(np.nanmean(ratios)),
                "max_ratio": float(np.nanmax(ratios)),
            }
        )

    results.sort(key=lambda x: x["mean_ratio"])
    return results
