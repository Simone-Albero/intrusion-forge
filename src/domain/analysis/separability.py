"""Pairwise class separability helpers.

Not wired into the current pipeline; kept here for future re-introduction.
"""

from itertools import combinations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import paired_distances
from tqdm import tqdm


def _mean_distance_intra(
    X: np.ndarray,
    idx: np.ndarray,
    max_pairs: int | None,
    metric: str,
    chunk_size: int = 512,
) -> float:
    """Mean intra-class distance without materialising the full distance array.

    Iterates the upper-triangle in row chunks when the total number of pairs is
    within *max_pairs* (or max_pairs is None); otherwise draws random paired
    samples and accumulates a running sum.
    """
    n = len(idx)
    if n < 2:
        return np.nan

    total_pairs = n * (n - 1) // 2
    exact = max_pairs is None or total_pairs <= max_pairs

    running_sum = 0.0
    running_count = 0

    if exact:
        for start in range(0, n - 1, chunk_size):
            end = min(start + chunk_size, n - 1)
            D = pairwise_distances(
                X[idx[start:end]], X[idx[start + 1 :]], metric=metric
            )
            for li in range(end - start):
                row = D[li, li:]
                running_sum += float(row.sum())
                running_count += len(row)
    else:
        size = int(max_pairs * 1.1) + 32
        i = np.random.randint(0, n, size=size)
        j = np.random.randint(0, n, size=size)
        mask = i != j
        i, j = i[mask][:max_pairs], j[mask][:max_pairs]
        for start in range(0, len(i), chunk_size):
            end = min(start + chunk_size, len(i))
            d = paired_distances(
                X[idx[i[start:end]]], X[idx[j[start:end]]], metric=metric
            )
            running_sum += float(d.sum())
            running_count += len(d)

    return running_sum / running_count if running_count > 0 else np.nan


def _mean_distance_inter(
    X: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    max_pairs: int | None,
    metric: str,
    chunk_size: int = 512,
) -> float:
    """Mean inter-class distance without materialising the full distance array."""
    if len(idx_a) == 0 or len(idx_b) == 0:
        return np.nan

    total_pairs = len(idx_a) * len(idx_b)
    exact = max_pairs is None or total_pairs <= max_pairs

    running_sum = 0.0
    running_count = 0

    if exact:
        for start in range(0, len(idx_a), chunk_size):
            end = min(start + chunk_size, len(idx_a))
            D = pairwise_distances(X[idx_a[start:end]], X[idx_b], metric=metric)
            running_sum += float(D.sum())
            running_count += D.size
    else:
        n = max_pairs
        replace = n > min(len(idx_a), len(idx_b))
        i = np.random.choice(len(idx_a), size=n, replace=replace)
        j = np.random.choice(len(idx_b), size=n, replace=replace)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            d = paired_distances(
                X[idx_a[i[start:end]]], X[idx_b[j[start:end]]], metric=metric
            )
            running_sum += float(d.sum())
            running_count += len(d)

    return running_sum / running_count if running_count > 0 else np.nan


def compute_pairwise_separability(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_pairs: int | None = 50_000,
    metric: str = "cosine",
    pca_variance: float = 0.9,
) -> dict[str, dict[str, float]]:
    """Compute pairwise separability ratios between groups defined by `y`.

    For every pair of groups the ratio ``mean_intra / mean_inter`` is computed.
    Low values indicate well-separated groups.

    Returns ``{group: {other_group: ratio, "_mean_ratio": float}}`` sorted by
    mean ratio ascending.
    """
    classes = np.unique(y)

    X = PCA(n_components=pca_variance, svd_solver="full", whiten=True).fit_transform(X)

    pair_metrics: dict[tuple, float] = {}
    n_pairs = len(classes) * (len(classes) - 1) // 2
    for class_a, class_b in tqdm(
        combinations(classes, 2), total=n_pairs, desc="class pairs"
    ):
        idx_a = np.where(y == class_a)[0]
        idx_b = np.where(y == class_b)[0]

        intra_a = _mean_distance_intra(X, idx_a, max_pairs=max_pairs, metric=metric)
        intra_b = _mean_distance_intra(X, idx_b, max_pairs=max_pairs, metric=metric)
        inter_ab = _mean_distance_inter(
            X, idx_a, idx_b, max_pairs=max_pairs, metric=metric
        )

        intra_mean = (
            float(np.mean([intra_a, intra_b]))
            if np.isfinite(intra_a) and np.isfinite(intra_b)
            else np.nan
        )
        ratio = (
            intra_mean / inter_ab
            if np.isfinite(intra_mean) and np.isfinite(inter_ab) and inter_ab != 0.0
            else np.nan
        )

        pair_metrics[(str(class_a), str(class_b))] = float(ratio)

    result: dict[str, dict[str, float]] = {}
    for cls in map(str, classes):
        pairs = {
            b if a == cls else a: ratio
            for (a, b), ratio in pair_metrics.items()
            if a == cls or b == cls
        }
        pairs = dict(
            sorted(
                pairs.items(), key=lambda x: x[1] if np.isfinite(x[1]) else float("inf")
            )
        )
        mean_ratio = float(np.nanmean(list(pairs.values()))) if pairs else np.nan
        result[cls] = {"_mean_ratio": mean_ratio, **pairs}

    return dict(sorted(result.items(), key=lambda x: x[1]["_mean_ratio"]))
