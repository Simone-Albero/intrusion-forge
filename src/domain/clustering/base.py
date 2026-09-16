import itertools
import logging
import time
from collections.abc import Callable, Sequence

import numpy as np
from sklearn.metrics import pairwise_distances, silhouette_score
from tqdm import tqdm

from src.core.utils import timed

logger = logging.getLogger(__name__)

FitFn = Callable[..., np.ndarray]
ClusterFn = Callable[[np.ndarray, np.ndarray | None], np.ndarray]


def cluster_size_balance(labels: np.ndarray) -> float:
    """Normalized entropy of non-noise cluster sizes in [0, 1] (1 = uniform)."""
    mask = labels != -1
    if not mask.any():
        return 0.0
    _, counts = np.unique(labels[mask], return_counts=True)
    k = counts.size
    if k < 2:
        return 0.0
    p = counts / counts.sum()
    h = float(-(p * np.log(p)).sum())
    return h / float(np.log(k))


def _measure(labels: np.ndarray, score: float, combo: dict, duration_s: float) -> dict:
    """Sweep entry describing one fitted partition."""
    n = int(labels.shape[0])
    n_noise = int((labels == -1).sum())
    n_clusters = int(np.unique(labels[labels != -1]).size) if n - n_noise > 0 else 0
    return {
        "combo": combo,
        "score": score,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": n_noise / n if n > 0 else 0.0,
        "size_balance": cluster_size_balance(labels),
        "duration_s": duration_s,
    }


def subsample_features(
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    max_samples: int,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Random subsample of the feature blocks, unchanged when already small enough."""
    n = X_num.shape[0]
    if n <= max_samples:
        return X_num, X_cat
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=max_samples, replace=False)
    return X_num[idx], (X_cat[idx] if X_cat is not None else None)


def _score_silhouette(
    X_num: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
) -> float:
    """Silhouette on non-noise points only. Returns -inf on failure or < 2 clusters."""
    mask = labels != -1
    if mask.sum() < 2:
        return float("-inf")
    unique = np.unique(labels[mask])
    if len(unique) < 2:
        return float("-inf")
    try:
        return float(silhouette_score(X_num[mask], labels[mask], metric=metric))
    except Exception:
        return float("-inf")


def assign_nearest_centroid(
    X_num: np.ndarray,
    centroids: dict,
    *,
    metric: str,
    candidate_ids: Sequence[int] | None = None,
    batch_size: int = 50_000,
) -> np.ndarray:
    """Assign each row to the nearest centroid id, restricted to `candidate_ids` if given."""
    items = [(int(k), v) for k, v in centroids.items()]
    if candidate_ids is not None:
        cand = {int(c) for c in candidate_ids}
        items = [(k, v) for k, v in items if k in cand]
    id_arr = np.array([k for k, _ in items], dtype=np.int64)
    C = np.array([np.asarray(v, dtype=np.float64) for _, v in items], dtype=np.float64)
    result = np.empty(len(X_num), dtype=np.int64)
    for start in range(0, len(X_num), batch_size):
        batch = X_num[start : start + batch_size]
        D = pairwise_distances(batch, C, metric=metric)
        result[start : start + batch_size] = id_arr[D.argmin(axis=1)]
    return result


def assign_clusters_within_class(
    X_num: np.ndarray,
    y_class: np.ndarray,
    centroids: dict,
    cluster_to_class: dict[int, int],
    *,
    metric: str,
    batch_size: int = 50_000,
) -> np.ndarray:
    """Assign each row to the nearest centroid of its own class, or -1 when it has none."""
    result = np.full(len(X_num), -1, dtype=np.int64)
    for cls in np.unique(y_class):
        candidate_ids = [cid for cid, c in cluster_to_class.items() if c == cls]
        if not candidate_ids:
            continue
        rows = np.where(y_class == cls)[0]
        result[rows] = assign_nearest_centroid(
            X_num[rows],
            centroids,
            metric=metric,
            candidate_ids=candidate_ids,
            batch_size=batch_size,
        )
    return result


@timed
def grid_search(
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    fit_fn: FitFn,
    param_grid: dict[str, list],
    *,
    max_fit_samples: int = 50_000,
    random_state: int = 0,
    noise_penalty: float = 3.0,
    resolution_weight: float = 0.1,
    min_clusters: int | None = None,
    **fixed_params,
) -> dict:
    """Grid search scored by silhouette − noise_penalty·noise_ratio + resolution tilt."""
    sub_num, sub_cat = subsample_features(X_num, X_cat, max_fit_samples, random_state)

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    sweep: list[dict] = []

    for combo_values in tqdm(
        itertools.product(*values),
        total=int(np.prod([len(v) for v in values])),
        desc="Grid search",
    ):
        combo = dict(zip(keys, combo_values))
        t0 = time.perf_counter()
        try:
            labels = fit_fn(sub_num, X_cat=sub_cat, **combo, **fixed_params)
        except Exception:
            sweep.append(
                {
                    "combo": combo,
                    "score": float("-inf"),
                    "n_clusters": 0,
                    "n_noise": 0,
                    "noise_ratio": 0.0,
                    "size_balance": 0.0,
                    "duration_s": time.perf_counter() - t0,
                    "error": True,
                }
            )
            continue

        duration = time.perf_counter() - t0
        sil = _score_silhouette(sub_num, labels)
        entry = _measure(labels, sil, combo, duration)
        entry["silhouette"] = sil
        sweep.append(entry)

    valid = [e for e in sweep if not e.get("error")]
    max_k = max((e["n_clusters"] for e in valid), default=0)
    for e in valid:
        tilt = resolution_weight * (e["n_clusters"] / max_k) if max_k > 0 else 0.0
        e["resolution_tilt"] = tilt
        e["score"] = e["silhouette"] - noise_penalty * e["noise_ratio"] + tilt

    candidates = valid
    if min_clusters is not None:
        above_floor = [e for e in valid if e["n_clusters"] >= min_clusters]
        if above_floor:
            candidates = above_floor
        elif valid:
            logger.warning(
                "grid_search: no candidate reaches min_clusters=%d (max available "
                "%d); scoring the full sweep instead.",
                min_clusters,
                max_k,
            )

    best_score = float("-inf")
    best_entry: dict | None = None
    for e in candidates:
        if e["score"] > best_score:
            best_score = e["score"]
            best_entry = e

    if best_entry is None:
        logger.warning(
            "grid_search: no scoreable clustering across %d combination(s); "
            "falling back to the first sweep entry (may be a failed fit).",
            len(sweep),
        )
        best_entry = sweep[0] if sweep else None

    if best_entry is None:
        raise RuntimeError(
            "grid_search: no valid clustering found across all parameter combinations."
        )

    return {"best": best_entry, "sweep": sweep}
