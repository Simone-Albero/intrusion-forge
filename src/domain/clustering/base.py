import itertools
import time
from collections.abc import Callable

import numpy as np
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from src.core.utils import timed

FitFn = Callable[..., np.ndarray]  # (X_num, X_cat=None, **params) -> labels
ClusterFn = Callable[[np.ndarray, np.ndarray | None], np.ndarray]  # (X_num, X_cat) -> labels


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
    """Headline metrics for a single clustering candidate."""
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


def _subsample(
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    max_samples: int,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Random subsample of X_num (and X_cat) to at most max_samples rows."""
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


@timed
def grid_search(
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    fit_fn: FitFn,
    param_grid: dict[str, list],
    *,
    max_fit_samples: int = 50_000,
    random_state: int = 0,
    **fixed_params,
) -> dict:
    """Generic grid search over param_grid, scored by Euclidean silhouette.

    Returns `{"best": entry, "sweep": [entry, ...]}` where every entry has
    `{combo, score, n_clusters, n_noise, noise_ratio, size_balance, duration_s}`.
    Failed fits are recorded with `score=-inf` and `error=True`. Raises if no
    candidate produced a clustering.

    fixed_params are forwarded unchanged to fit_fn on every call.
    """
    sub_num, sub_cat = _subsample(X_num, X_cat, max_fit_samples, random_state)

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    best_score = float("-inf")
    best_entry: dict | None = None
    fallback_entry: dict | None = None
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
            sweep.append({
                "combo": combo,
                "score": float("-inf"),
                "n_clusters": 0,
                "n_noise": 0,
                "noise_ratio": 0.0,
                "size_balance": 0.0,
                "duration_s": time.perf_counter() - t0,
                "error": True,
            })
            continue

        duration = time.perf_counter() - t0
        s = _score_silhouette(sub_num, labels)
        entry = _measure(labels, s, combo, duration)
        sweep.append(entry)
        if fallback_entry is None:
            fallback_entry = entry
        if s > best_score:
            best_score = s
            best_entry = entry

    if best_entry is None:
        best_entry = fallback_entry

    if best_entry is None:
        raise RuntimeError(
            "grid_search: no valid clustering found across all parameter combinations."
        )

    return {"best": best_entry, "sweep": sweep}
