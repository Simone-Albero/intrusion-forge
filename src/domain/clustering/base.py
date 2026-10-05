import itertools
import logging
import time
from collections.abc import Callable

import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from src.core.utils import timed
from src.domain.analysis.complexity.shared import l2_normalize
from src.domain.clustering.hardness import KdnNodes, compute_kdn

logger = logging.getLogger(__name__)

FitFn = Callable[..., np.ndarray]
ClusterFn = Callable[..., tuple[np.ndarray, int, int]]


def cluster_size_balance(labels: np.ndarray) -> float:
    """Normalized entropy of cluster sizes in [0, 1] (1 = uniform)."""
    _, counts = np.unique(labels, return_counts=True)
    n_clusters = counts.size
    if n_clusters < 2:
        return 0.0
    shares = counts / counts.sum()
    entropy = float(-(shares * np.log(shares)).sum())
    return entropy / float(np.log(n_clusters))


def subsample_indices(n: int, *, max_samples: int, random_state: int) -> np.ndarray:
    """Indices of a random subsample of `n` rows, unchanged (arange) when already small enough."""
    if n <= max_samples:
        return np.arange(n)
    rng = np.random.default_rng(random_state)
    return rng.choice(n, size=max_samples, replace=False)


def subsample_features(
    X: np.ndarray,
    *,
    max_samples: int,
    random_state: int,
) -> np.ndarray:
    """Random subsample of X, unchanged when already small enough."""
    rows = subsample_indices(
        X.shape[0], max_samples=max_samples, random_state=random_state
    )
    return X[rows]


def assign_nearest_centroid(
    X: np.ndarray,
    centroids: dict,
    *,
    metric: str,
    batch_size: int = 50_000,
) -> np.ndarray:
    """Assign each row to the nearest centroid id."""
    items = [(int(k), v) for k, v in centroids.items()]
    centroid_ids = np.array([k for k, _ in items], dtype=np.int64)
    centroid_matrix = np.array(
        [np.asarray(v, dtype=np.float64) for _, v in items], dtype=np.float64
    )
    result = np.empty(len(X), dtype=np.int64)
    for start in range(0, len(X), batch_size):
        batch = X[start : start + batch_size]
        distances = pairwise_distances(batch, centroid_matrix, metric=metric)
        result[start : start + batch_size] = centroid_ids[distances.argmin(axis=1)]
    return result


def compute_centroids(
    X: np.ndarray, labels: np.ndarray, *, metric: str
) -> dict[int, np.ndarray]:
    """Mean feature vector of every label."""
    centroids = {}
    for label in np.unique(labels):
        centroid = X[labels == label].mean(axis=0)
        # Under cosine a region's centre is a direction: the mean of the unit vectors
        # it was clustered on, renormalized.
        if metric == "cosine":
            centroid = l2_normalize(centroid[np.newaxis])[0]
        centroids[int(label)] = centroid
    return centroids


def merge_small_clusters(
    X: np.ndarray, labels: np.ndarray, *, min_size: float, metric: str
) -> tuple[np.ndarray, int, int]:
    """Fold noise and every cluster under `min_size` into its nearest surviving region."""
    ids, counts = np.unique(labels[labels != -1], return_counts=True)
    small_ids = ids[counts < min_size]
    absorbed = np.isin(labels, small_ids) | (labels == -1)
    n_merged_clusters = int(small_ids.size)
    n_merged = int(absorbed.sum())
    if not absorbed.any():
        return labels, 0, 0

    survivor_mask = ~absorbed
    if not survivor_mask.any():
        # No cluster reaches the floor: the whole class becomes one region.
        return (
            np.zeros(labels.shape[0], dtype=labels.dtype),
            n_merged_clusters,
            n_merged,
        )

    centroids = compute_centroids(
        X[survivor_mask], labels[survivor_mask], metric=metric
    )
    merged = labels.copy()
    merged[absorbed] = assign_nearest_centroid(X[absorbed], centroids, metric=metric)
    return merged, n_merged_clusters, n_merged


def _granularity_loss(
    labels: np.ndarray,
    hardness: np.ndarray,
    *,
    n_class: int,
    eval_rows_per_train_row: float,
) -> tuple[float, float, float]:
    """Expected squared error of reading each row's hardness off its region's rate:
    `within` is the hardness variance a region's mean hides, `noise` the sampling noise
    of a rate measured on the test rows the region will receive, at its full-class size.
    A finer partition lowers the first and raises the second."""
    ids, counts = np.unique(labels, return_counts=True)
    scale = n_class / labels.shape[0]
    weight = counts / counts.sum()
    groups = [hardness[labels == cluster] for cluster in ids]
    mean_hardness = np.array([group.mean() for group in groups])
    variance = np.array(
        [group.var(ddof=1) if group.size > 1 else np.nan for group in groups]
    )
    # A one-row region stands for about `scale` rows of the class whose spread is
    # unknown, not zero: it takes the spread of the regions that have one.
    known = ~np.isnan(variance)
    pooled = (
        np.average(variance[known], weights=counts[known])
        if known.any()
        else hardness.var(ddof=1)
    )
    within = float((weight * np.where(known, variance, pooled)).sum())
    noise = float(
        (
            weight
            * np.maximum(mean_hardness * (1 - mean_hardness), 1e-3)
            / np.maximum(counts * scale * eval_rows_per_train_row, 1e-9)
        ).sum()
    )
    return within + noise, within, noise


@timed
def grid_search(
    X: np.ndarray,
    fit_fn: FitFn,
    param_grid: dict[str, list],
    *,
    max_fit_samples: int,
    random_state: int,
    row_ids: np.ndarray,
    label: object,
    nodes: KdnNodes,
    hardness_k: int,
    eval_rows_per_train_row: float,
    min_cluster_floor: int,
    max_clusters: int,
    merge_metric: str,
    # merge_metric, not metric: a keyword named here is absorbed instead of reaching
    # fit_fn through **fixed_params, so it must not shadow an algorithm's own parameter.
    **fixed_params,
) -> tuple[dict, np.ndarray | None]:
    """Grid search keeping the candidate of least `_granularity_loss`."""
    scored_rows = subsample_indices(
        X.shape[0], max_samples=max_fit_samples, random_state=random_state
    )
    X_sub = X[scored_rows]
    n_class = X.shape[0]
    scale = n_class / X_sub.shape[0]
    scaled_floor = min_cluster_floor / scale
    hardness = compute_kdn(
        X_sub, row_ids=row_ids[scored_rows], label=label, nodes=nodes, k=hardness_k
    )

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    sweep: list[dict] = []
    sweep_labels: list[np.ndarray | None] = []
    failures: list[tuple[dict, Exception]] = []

    for combo_values in tqdm(
        itertools.product(*values),
        total=int(np.prod([len(v) for v in values])) if values else 1,
        desc="Grid search",
    ):
        combo = dict(zip(keys, combo_values))
        started = time.perf_counter()
        try:
            # Passed by hand: as grid_search's own parameters they never reach fit_fn
            # through **fixed_params.
            labels = fit_fn(
                X_sub,
                **combo,
                max_fit_samples=max_fit_samples,
                random_state=random_state,
                **fixed_params,
            )
        except TypeError:
            # A call the algorithm cannot take is a configuration error, not a degenerate
            # candidate: stop the sweep rather than let it pick among the survivors.
            raise
        except Exception as exc:
            failures.append((combo, exc))
            sweep.append(
                {
                    "combo": combo,
                    "n_clusters": 0,
                    "n_merged_clusters": 0,
                    "n_merged": 0,
                    "size_balance": 0.0,
                    "loss": float("nan"),
                    "loss_within": float("nan"),
                    "loss_noise": float("nan"),
                    "duration_s": time.perf_counter() - started,
                    "error": True,
                }
            )
            sweep_labels.append(None)
            continue

        labels, n_merged_clusters, n_merged = merge_small_clusters(
            X_sub, labels, min_size=scaled_floor, metric=merge_metric
        )
        loss, loss_within, loss_noise = _granularity_loss(
            labels,
            hardness,
            n_class=n_class,
            eval_rows_per_train_row=eval_rows_per_train_row,
        )
        sweep.append(
            {
                "combo": combo,
                "n_clusters": int(np.unique(labels).size),
                "n_merged_clusters": n_merged_clusters,
                "n_merged": n_merged,
                "size_balance": cluster_size_balance(labels),
                "loss": loss,
                "loss_within": loss_within,
                "loss_noise": loss_noise,
                "duration_s": time.perf_counter() - started,
            }
        )
        sweep_labels.append(labels)

    if failures:
        first_combo, first_exc = failures[0]
        logger.warning(
            "grid_search: %d of %d candidates failed and were skipped (first: %s, %r)",
            len(failures),
            len(sweep),
            first_combo,
            first_exc,
        )

    valid = [entry for entry in sweep if not entry.get("error")]
    if not valid:
        raise RuntimeError(
            "grid_search: no valid clustering found across all parameter combinations."
        )

    eligible = [entry for entry in valid if entry["n_clusters"] <= max_clusters]
    if not eligible:
        raise ValueError(
            f"No candidate partition of max_clusters={max_clusters} or fewer survives "
            "merging undersized regions. Raise clustering.max_regions or tighten the "
            "clustering grid."
        )

    best_entry = min(eligible, key=lambda entry: entry["loss"])

    # Without subsampling, the sweep's fit of the winner is the refit itself.
    best_idx = next(i for i, entry in enumerate(sweep) if entry is best_entry)
    subsampled = len(X) > max_fit_samples
    best_labels = None if subsampled else sweep_labels[best_idx]
    for i, entry in enumerate(sweep):
        entry["best"] = i == best_idx

    return {"best": best_entry, "sweep": sweep}, best_labels
