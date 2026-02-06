import numpy as np
from typing import Dict, Any, Iterable, Optional, Tuple, List
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.neighbors import NearestNeighbors
import hdbscan


def kmeans_grid_search(
    X: np.ndarray,
    n_clusters: Iterable[int] = range(2, 21),
    n_init: int = 10,
    max_iter: int = 300,
    random_state: int = 42,
    metric: str = "euclidean",
    penalize_k: bool = True,
) -> Tuple[KMeans, np.ndarray, Dict[str, Any]]:
    """
    Returns:
      best_model, best_labels, best_info

    Scoring:
      silhouette(X, labels) - 0.01 * k  (if penalize_k=True)
    """
    X = np.asarray(X)
    if X.shape[0] < 3:
        raise ValueError("Need at least 3 samples.")

    best_model = None
    best_labels = None
    best_score = -np.inf
    best_info: Dict[str, Any] = {}

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
            best_model = model
            best_labels = labels
            best_info = {
                "n_clusters": k,
                "silhouette": sil,
                "score": score,
            }

    if best_model is None:
        raise RuntimeError("No valid KMeans solution found.")

    return best_model, best_labels, best_info


def hdbscan_grid_search(
    X: np.ndarray,
    min_cluster_size: Iterable[int] = (30, 50, 100, 150),
    min_samples: Iterable[Optional[int]] = (None, 5, 10),
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
    cluster_selection_epsilon: float = 0.0,
    min_clusters: int = 2,
    max_noise_ratio: float = 0.60,
    min_clustered_ratio: float = 0.20,
    penalize: bool = True,
) -> Tuple[hdbscan.HDBSCAN, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Returns: (best_model, labels, probabilities, best_info)
    labels: shape (n_samples,), noise = -1
    probabilities: shape (n_samples,), in [0,1]
    """
    X = np.asarray(X)
    n = X.shape[0]
    if n < 5:
        raise ValueError("Need at least 5 samples.")

    best_model = None
    best_labels = None
    best_proba = None
    best_info: Dict[str, Any] = {}
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
            n_clusters = int(np.unique(labels[mask]).size) if mask.any() else 0

            # reject degenerate solutions
            if n_clusters < min_clusters:
                continue
            if noise_ratio > max_noise_ratio:
                continue
            if clustered_ratio < min_clustered_ratio:
                continue

            try:
                sil = float(silhouette_score(X[mask], labels[mask], metric=metric))
            except Exception:
                continue

            score = sil
            if penalize:
                score = sil - 0.5 * noise_ratio - 0.02 * n_clusters

            if score > best_score:
                best_score = score
                best_model = model
                best_labels = labels
                best_proba = proba
                best_info = {
                    "min_cluster_size": int(mcs),
                    "min_samples": None if ms is None else int(ms),
                    "n_clusters": n_clusters,
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
    sample_size: int = None,
    random_state: int = 42,
) -> float:
    """
    Compute the Hopkins statistic to assess clustering tendency.

    The Hopkins statistic tests the spatial randomness of the data.
    Values close to 0.5 indicate random data (uniform distribution).
    Values close to 0 indicate regularly spaced data.
    Values close to 1 indicate clustered data.

    Args:
        X: Input data of shape (n_samples, n_features)
        sample_size: Number of samples to use for computation.
                     Default: min(n_samples - 1, 200)
        random_state: Random state for reproducibility

    Returns:
        Hopkins statistic value in range [0, 1]
    """
    n_samples, n_features = X.shape

    if sample_size is None:
        sample_size = min(n_samples - 1, 200)

    # Ensure sample size is valid
    sample_size = min(sample_size, n_samples - 1)

    # Set random seed
    rng = np.random.RandomState(random_state)

    # Sample random points from the dataset
    sample_indices = rng.choice(n_samples, size=sample_size, replace=False)
    X_sample = X[sample_indices]

    # Fit nearest neighbors on the full dataset
    nbrs = NearestNeighbors(n_neighbors=2).fit(X)

    # Get distances to nearest neighbor for sampled real points (excluding self)
    u_distances, _ = nbrs.kneighbors(X_sample)
    u_distances = u_distances[:, 1]  # Distance to nearest neighbor (not self)

    # Generate random points within the data space
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    random_points = rng.uniform(min_vals, max_vals, size=(sample_size, n_features))

    # Get distances to nearest neighbor for random points
    v_distances, _ = nbrs.kneighbors(random_points)
    v_distances = v_distances[:, 0]  # Distance to nearest neighbor

    # Compute Hopkins statistic
    u_sum = np.sum(u_distances)
    v_sum = np.sum(v_distances)

    if u_sum + v_sum == 0:
        return 0.5  # Degenerate case

    hopkins = v_sum / (u_sum + v_sum)

    return float(hopkins)


def compute_cluster_quality_measures(
    X: np.ndarray,
    labels: np.ndarray,
    filter_noise: bool = True,
) -> Dict[str, float]:
    """
    Compute various cluster quality measures.

    Args:
        X: Input data of shape (n_samples, n_features)
        labels: Cluster labels for each sample
        filter_noise: If True, filter out noise points (label = -1) before computing metrics

    Returns:
        Dictionary containing various quality measures:
            - silhouette: Silhouette coefficient (higher is better, range [-1, 1])
            - davies_bouldin: Davies-Bouldin index (lower is better, range [0, inf))
            - calinski_harabasz: Calinski-Harabasz index (higher is better)
            - hopkins: Hopkins statistic (closer to 1 indicates clustered data, range [0, 1])
            - n_clusters: Number of clusters found
            - n_noise: Number of noise points (if any)
            - noise_ratio: Ratio of noise points to total samples
    """
    # Filter noise points if requested
    if filter_noise and -1 in labels:
        mask = labels != -1
        X_filtered = X[mask]
        labels_filtered = labels[mask]
        n_noise = np.sum(~mask)
    else:
        X_filtered = X
        labels_filtered = labels
        n_noise = 0

    n_clusters = len(np.unique(labels_filtered))

    measures = {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": n_noise / len(labels),
    }

    # Compute Hopkins statistic on the original data (before filtering)
    try:
        measures["hopkins"] = hopkins_statistic(X)
    except Exception as e:
        pass

    # Compute metrics if we have at least 2 clusters and enough samples
    if n_clusters >= 2 and len(X_filtered) > n_clusters:
        try:
            measures["silhouette"] = silhouette_score(X_filtered, labels_filtered)
        except Exception as e:
            pass

        try:
            measures["davies_bouldin"] = davies_bouldin_score(
                X_filtered, labels_filtered
            )
        except Exception as e:
            pass
        try:
            measures["calinski_harabasz"] = calinski_harabasz_score(
                X_filtered, labels_filtered
            )
        except Exception as e:
            pass

    return measures
