import logging
from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
import hdbscan
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from tqdm import tqdm

ClusterFn = Callable[[np.ndarray], np.ndarray]

logger = logging.getLogger(__name__)


def hdbscan_grid_search(
    X: np.ndarray,
    min_cluster_size: Iterable[int] = (30, 50, 100, 150),
    min_samples: Iterable[int | None] = (30, 50, 75, 100, 150),
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
    cluster_selection_epsilon: float = 0.0,
    min_clusters: int = 2,
    max_noise_ratio: float = 0.60,
    min_clustered_ratio: float = 0.20,
    penalize: bool = True,
    max_fit_samples: int = 50_000,
    pca_variance: float = 0.8,
) -> tuple[hdbscan.HDBSCAN, np.ndarray, np.ndarray, dict]:
    """Grid-search over HDBSCAN hyperparameters, scored by silhouette.

    Args:
        max_fit_samples: Fit on at most this many samples, then approximate_predict
            on the full dataset. Set to 0 to disable.
        pca_variance: Fraction of variance to retain via PCA before clustering.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)

    n_non_finite = int(np.count_nonzero(~np.isfinite(X)))
    if n_non_finite:
        logger.warning(
            "Replacing %d non-finite values with 0 before clustering.", n_non_finite
        )
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    pca = PCA(n_components=pca_variance, random_state=42)
    X = np.ascontiguousarray(pca.fit_transform(X), dtype=np.float64)

    n = X.shape[0]
    if n < 5:
        raise ValueError("Need at least 5 samples.")

    if max_fit_samples and n > max_fit_samples:
        logger.info(
            "Downsampling from %d to %d samples for HDBSCAN fit.", n, max_fit_samples
        )
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=max_fit_samples, replace=False)
        X_fit = np.ascontiguousarray(X[idx], dtype=np.float64)
    else:
        X_fit = X

    best_model, best_labels, best_proba, best_info = None, None, None, {}
    best_score = -np.inf

    for mcs in tqdm(min_cluster_size, desc="min_cluster_size"):
        for ms in tqdm(min_samples, desc="min_samples"):
            model = hdbscan.HDBSCAN(
                min_cluster_size=int(mcs),
                min_samples=None if ms is None else int(ms),
                metric=metric,
                cluster_selection_method=cluster_selection_method,
                cluster_selection_epsilon=float(cluster_selection_epsilon),
                prediction_data=True,
                algorithm="prims_kdtree",
            )
            labels = model.fit_predict(X_fit)
            proba = getattr(
                model, "probabilities_", np.zeros(X_fit.shape[0], dtype=float)
            )

            mask = labels != -1
            noise_ratio = float((~mask).mean())
            clustered_ratio = float(mask.mean())
            n_clust = int(np.unique(labels[mask]).size) if mask.any() else 0

            if (
                n_clust < min_clusters
                or noise_ratio > max_noise_ratio
                or clustered_ratio < min_clustered_ratio
            ):
                continue

            try:
                sil = float(silhouette_score(X_fit[mask], labels[mask], metric=metric))
            except Exception:
                continue

            score = sil - (0.5 * noise_ratio + 0.02 * n_clust if penalize else 0.0)
            if score > best_score:
                best_score = score
                best_model, best_labels, best_proba = model, labels, proba
                best_info = {
                    "min_cluster_size": int(mcs),
                    "min_samples": None if ms is None else int(ms),
                    "n_clusters": n_clust,
                    "noise_ratio": noise_ratio,
                    "clustered_ratio": clustered_ratio,
                    "silhouette": sil,
                    "score": score,
                }

    if best_model is None:
        raise RuntimeError(
            "No valid clustering found. Try expanding the grid or relaxing filters."
        )

    if max_fit_samples and n > max_fit_samples:
        logger.info("Predicting on full dataset (%d samples)...", n)
        best_labels, best_proba = hdbscan.approximate_predict(best_model, X)
        best_labels = np.array(best_labels)
        best_proba = np.array(best_proba)

    return best_model, best_labels, best_proba, best_info


def make_hdbscan_cluster_fn(**kwargs) -> ClusterFn:
    """Wrap :func:`hdbscan_grid_search` into a ``ClusterFn`` (X → labels)."""

    def _fn(X: np.ndarray) -> np.ndarray:
        _, labels, _, _ = hdbscan_grid_search(X, **kwargs)
        return labels

    return _fn


def assign_clusters(
    df: pd.DataFrame,
    feature_cols: list[str],
    cluster_fn: ClusterFn,
    label_col: str | None = None,
    classes: list | None = None,
    dst_col: str = "cluster",
    noise_label: int = -1,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Assign cluster labels to rows of *df* using *cluster_fn*.

    When *classes* is provided, clustering runs independently per class.
    Noise labels (== *noise_label*) are remapped to ``max_label + 1``.
    Labels are offset so they are globally unique across groups.

    Returns ``(df, centroids)`` where *centroids* maps ``str(label)`` to
    centroid ndarray.
    """
    df = df.copy()
    df[dst_col] = noise_label
    centroids: dict[str, np.ndarray] = {}
    offset = 0

    if classes is None or label_col is None:
        groups = [(df.index, df[feature_cols].values)]
    else:
        groups = [
            (
                df[df[label_col] == cls].index,
                df.loc[df[label_col] == cls, feature_cols].values,
            )
            for cls in classes
        ]

    for idx, values in groups:
        if len(values) == 0:
            continue
        labels = cluster_fn(values)

        remap = int(labels.max()) + 1
        labels = np.where(labels == noise_label, remap, labels)

        df.loc[idx, dst_col] = labels + offset

        for lbl in np.unique(labels):
            centroids[str(int(lbl) + offset)] = values[labels == lbl].mean(axis=0)

        offset += remap + 1

    return df, centroids
