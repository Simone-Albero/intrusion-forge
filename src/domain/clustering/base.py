import itertools
import logging
import time
from collections.abc import Callable

import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from src.core.utils import timed
from src.domain.analysis.complexity.shared import l2_normalize
from src.domain.clustering.hardness import KdnReference, compute_kdn

logger = logging.getLogger(__name__)

FitFn = Callable[..., np.ndarray]
ClusterFn = Callable[..., tuple[np.ndarray, int, int]]


def cluster_size_balance(labels: np.ndarray) -> float:
    """Normalized entropy of cluster sizes in [0, 1] (1 = uniform)."""
    _, counts = np.unique(labels, return_counts=True)
    k = counts.size
    if k < 2:
        return 0.0
    p = counts / counts.sum()
    h = float(-(p * np.log(p)).sum())
    return h / float(np.log(k))


def subsample_indices(n: int, *, max_samples: int, random_state: int) -> np.ndarray:
    """Indices of a random subsample of `n` rows, unchanged (arange) when already small enough."""
    if n <= max_samples:
        return np.arange(n)
    rng = np.random.default_rng(random_state)
    return rng.choice(n, size=max_samples, replace=False)


def subsample_features(
    X_num: np.ndarray,
    *,
    max_samples: int,
    random_state: int,
) -> np.ndarray:
    """Random subsample of X_num, unchanged when already small enough."""
    idx = subsample_indices(
        X_num.shape[0], max_samples=max_samples, random_state=random_state
    )
    return X_num[idx]


def assign_nearest_centroid(
    X_num: np.ndarray,
    centroids: dict,
    *,
    metric: str,
    batch_size: int = 50_000,
) -> np.ndarray:
    """Assign each row to the nearest centroid id."""
    items = [(int(k), v) for k, v in centroids.items()]
    id_arr = np.array([k for k, _ in items], dtype=np.int64)
    C = np.array([np.asarray(v, dtype=np.float64) for _, v in items], dtype=np.float64)
    result = np.empty(len(X_num), dtype=np.int64)
    for start in range(0, len(X_num), batch_size):
        batch = X_num[start : start + batch_size]
        D = pairwise_distances(batch, C, metric=metric)
        result[start : start + batch_size] = id_arr[D.argmin(axis=1)]
    return result


def compute_centroids(
    X_num: np.ndarray, labels: np.ndarray, *, metric: str
) -> dict[int, np.ndarray]:
    """Mean feature vector of every label."""
    centroids = {}
    for cid in np.unique(labels):
        centroid = X_num[labels == cid].mean(axis=0)
        # Under cosine a region's centre is a direction: the mean of the unit vectors
        # it was clustered on, renormalized.
        if metric == "cosine":
            centroid = l2_normalize(centroid[np.newaxis])[0]
        centroids[int(cid)] = centroid
    return centroids


def merge_small_clusters(
    X_num: np.ndarray, labels: np.ndarray, *, min_size: float, metric: str
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
        X_num[survivor_mask], labels[survivor_mask], metric=metric
    )
    out = labels.copy()
    out[absorbed] = assign_nearest_centroid(X_num[absorbed], centroids, metric=metric)
    return out, n_merged_clusters, n_merged


def _predict_reliability(
    labels: np.ndarray,
    hardness: np.ndarray,
    *,
    n_class: int,
    eval_rows_per_train_row: float,
) -> tuple[float, float, float]:
    """Between/within-region variance of `hardness`, scaled to the rows classify will
    evaluate each region on. `V_b` is the size-weighted spread of each region's mean
    hardness; `V_n` is the size-weighted sampling noise of a rate measured on that many
    rows. Reliability is `V_b / (V_b + V_n)`: how much of the spread a measured error
    rate would actually reflect, rather than noise from too few evaluated rows."""
    ids, counts = np.unique(labels, return_counts=True)
    scale = n_class / labels.shape[0]
    n_full = counts * scale
    p = np.array([hardness[labels == cid].mean() for cid in ids])
    w = counts / counts.sum()
    grand_mean = float((w * p).sum())
    var_between = float((w * (p - grand_mean) ** 2).sum())
    var_sampling = float(
        (
            w
            * np.clip(p * (1 - p), 1e-3, None)
            / np.maximum(n_full * eval_rows_per_train_row, 1e-9)
        ).sum()
    )
    total = var_between + var_sampling
    reliability = var_between / total if total > 0 else 0.0
    return reliability, var_between, var_sampling


@timed
def grid_search(
    X_num: np.ndarray,
    fit_fn: FitFn,
    param_grid: dict[str, list],
    *,
    max_fit_samples: int,
    random_state: int,
    ids: np.ndarray,
    label: object,
    reference: KdnReference,
    hardness_k: int,
    eval_rows_per_train_row: float,
    reliability_target: float,
    min_cluster_floor: int,
    max_clusters: int,
    merge_metric: str,
    # merge_metric, not metric: a keyword named here is absorbed instead of reaching
    # fit_fn through **fixed_params, so it must not shadow an algorithm's own parameter.
    **fixed_params,
) -> tuple[dict, np.ndarray | None]:
    """Grid search scored by the reliability classify's evaluation would let it measure."""
    scored = subsample_indices(
        X_num.shape[0], max_samples=max_fit_samples, random_state=random_state
    )
    sub_num = X_num[scored]
    n_class = X_num.shape[0]
    scale = n_class / sub_num.shape[0]
    scaled_floor = min_cluster_floor / scale
    hardness = compute_kdn(
        sub_num, ids=ids[scored], label=label, reference=reference, k=hardness_k
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
        t0 = time.perf_counter()
        try:
            # Passed by hand: as grid_search's own parameters they never reach fit_fn
            # through **fixed_params.
            labels = fit_fn(
                sub_num,
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
                    "var_between": 0.0,
                    "var_sampling": 0.0,
                    "reliability": float("-inf"),
                    "duration_s": time.perf_counter() - t0,
                    "error": True,
                }
            )
            sweep_labels.append(None)
            continue

        labels, n_merged_clusters, n_merged = merge_small_clusters(
            sub_num, labels, min_size=scaled_floor, metric=merge_metric
        )
        reliability, var_between, var_sampling = _predict_reliability(
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
                "var_between": var_between,
                "var_sampling": var_sampling,
                "reliability": reliability,
                "duration_s": time.perf_counter() - t0,
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

    valid = [e for e in sweep if not e.get("error")]
    if not valid:
        raise RuntimeError(
            "grid_search: no valid clustering found across all parameter combinations."
        )

    eligible = [e for e in valid if e["n_clusters"] <= max_clusters]
    if not eligible:
        raise ValueError(
            f"No candidate partition of max_clusters={max_clusters} or fewer survives "
            "merging undersized regions. Raise max_complexity_samples, lower "
            "min_subsample_per_cluster, or tighten the clustering grid."
        )

    above_target = [e for e in eligible if e["reliability"] >= reliability_target]
    best_entry = (
        max(above_target, key=lambda e: (e["n_clusters"], e["reliability"]))
        if above_target
        else max(eligible, key=lambda e: e["reliability"])
    )

    # Without subsampling, the sweep's fit of the winner is the refit itself.
    best_idx = next(i for i, e in enumerate(sweep) if e is best_entry)
    subsampled = len(X_num) > max_fit_samples
    best_labels = None if subsampled else sweep_labels[best_idx]
    for i, entry in enumerate(sweep):
        entry["best"] = i == best_idx

    return {"best": best_entry, "sweep": sweep}, best_labels
