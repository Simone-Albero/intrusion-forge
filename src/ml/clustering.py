from collections.abc import Iterable

import numpy as np
import hdbscan
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.neighbors import NearestNeighbors


def kmeans_grid_search(
    X: np.ndarray,
    n_clusters: Iterable[int] = range(2, 21),
    n_init: int = 10,
    max_iter: int = 300,
    random_state: int = 42,
    metric: str = "euclidean",
    penalize_k: bool = True,
) -> tuple[KMeans, np.ndarray, dict]:
    """Grid-search over k for KMeans, scored by silhouette (- 0.01*k if penalize_k).

    Returns:
        (best_model, best_labels, best_info)
    """
    X = np.asarray(X)
    if X.shape[0] < 3:
        raise ValueError("Need at least 3 samples.")

    best_model, best_labels, best_info = None, None, {}
    best_score = -np.inf

    for k in n_clusters:
        k = int(k)
        if k < 2 or k >= X.shape[0]:
            continue

        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
        )
        labels = model.fit_predict(X)
        if np.unique(labels).size < 2:
            continue

        try:
            sil = float(silhouette_score(X, labels, metric=metric))
        except Exception:
            continue

        score = sil - (0.01 * k if penalize_k else 0.0)
        if score > best_score:
            best_score = score
            best_model, best_labels = model, labels
            best_info = {"n_clusters": k, "silhouette": sil, "score": score}

    if best_model is None:
        raise RuntimeError("No valid KMeans solution found.")

    return best_model, best_labels, best_info


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
) -> tuple[hdbscan.HDBSCAN, np.ndarray, np.ndarray, dict]:
    """Grid-search over HDBSCAN hyperparameters, scored by silhouette (with optional penalties).

    Returns:
        (best_model, labels, probabilities, best_info)
    """
    X = np.asarray(X)
    n = X.shape[0]
    if n < 5:
        raise ValueError("Need at least 5 samples.")

    best_model, best_labels, best_proba, best_info = None, None, None, {}
    best_score = -np.inf

    for mcs in min_cluster_size:
        for ms in min_samples:
            model = hdbscan.HDBSCAN(
                min_cluster_size=int(mcs),
                min_samples=None if ms is None else int(ms),
                metric=metric,
                cluster_selection_method=cluster_selection_method,
                cluster_selection_epsilon=float(cluster_selection_epsilon),
                prediction_data=True,
                core_dist_n_jobs=-1,
            )
            labels = model.fit_predict(X)
            proba = getattr(model, "probabilities_", np.zeros(n, dtype=float))

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
                sil = float(silhouette_score(X[mask], labels[mask], metric=metric))
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

    return best_model, best_labels, best_proba, best_info


def hopkins_statistic(
    X: np.ndarray,
    sample_size: int | None = None,
    random_state: int = 42,
) -> float:
    """Compute the Hopkins statistic to assess clustering tendency.

    Returns a value in [0, 1]: ~0.5 → random, ~1 → clustered, ~0 → regular.
    """
    n_samples, n_features = X.shape
    sample_size = min(sample_size or 200, n_samples - 1)
    rng = np.random.RandomState(random_state)

    X_sample = X[rng.choice(n_samples, size=sample_size, replace=False)]
    nbrs = NearestNeighbors(n_neighbors=2).fit(X)

    u_distances = nbrs.kneighbors(X_sample)[0][:, 1]

    random_points = rng.uniform(
        X.min(axis=0), X.max(axis=0), size=(sample_size, n_features)
    )
    v_distances = nbrs.kneighbors(random_points)[0][:, 0]

    u_sum, v_sum = np.sum(u_distances), np.sum(v_distances)
    return 0.5 if u_sum + v_sum == 0 else float(v_sum / (u_sum + v_sum))


def compute_cluster_quality_measures(
    X: np.ndarray,
    labels: np.ndarray,
    filter_noise: bool = True,
) -> dict[str, float]:
    """Compute cluster quality measures (silhouette, Davies-Bouldin, Calinski-Harabasz, Hopkins).

    Returns a dict with keys: silhouette, davies_bouldin, calinski_harabasz, hopkins,
    n_clusters, n_noise, noise_ratio.
    """
    if filter_noise and -1 in labels:
        mask = labels != -1
        X_f, labels_f = X[mask], labels[mask]
        n_noise = int((~mask).sum())
    else:
        X_f, labels_f = X, labels
        n_noise = 0

    n_clusters = len(np.unique(labels_f))
    measures: dict[str, float] = {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": n_noise / len(labels),
    }

    try:
        measures["hopkins"] = hopkins_statistic(X)
    except Exception:
        pass

    if n_clusters >= 2 and len(X_f) > n_clusters:
        for key, fn in (
            ("silhouette", lambda: silhouette_score(X_f, labels_f)),
            ("davies_bouldin", lambda: davies_bouldin_score(X_f, labels_f)),
            ("calinski_harabasz", lambda: calinski_harabasz_score(X_f, labels_f)),
        ):
            try:
                measures[key] = float(fn())
            except Exception:
                pass

    return measures
