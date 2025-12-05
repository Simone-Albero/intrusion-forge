import numpy as np
from typing import Dict, Any, Tuple, List
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
    n_clusters_range: List[int] = None,
    init_methods: List[str] = None,
    n_init: int = 10,
    max_iter: int = 300,
    random_state: int = 42,
    metric: str = "silhouette",
) -> Tuple[KMeans, Dict[str, Any]]:
    """
    Perform grid search for K-means clustering to find optimal parameters.

    Args:
        X: Input data of shape (n_samples, n_features)
        n_clusters_range: List of number of clusters to try. Default: [2, 3, 4, 5, 6, 7, 8, 9, 10]
        init_methods: List of initialization methods. Default: ['k-means++', 'random']
        n_init: Number of time the k-means algorithm will be run with different centroid seeds
        max_iter: Maximum number of iterations
        random_state: Random state for reproducibility
        metric: Metric to optimize ('silhouette', 'calinski_harabasz', 'davies_bouldin')

    Returns:
        best_model: Best K-means model found
        best_params: Dictionary containing best parameters and scores
    """
    if n_clusters_range is None:
        n_clusters_range = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    if init_methods is None:
        init_methods = ["k-means++", "random"]

    best_score = -np.inf if metric != "davies_bouldin" else np.inf
    best_model = None
    best_params = {}
    results = []

    for n_clusters in n_clusters_range:
        for init_method in init_methods:
            # Fit K-means
            kmeans = KMeans(
                n_clusters=n_clusters,
                init=init_method,
                n_init=n_init,
                max_iter=max_iter,
                random_state=random_state,
            )
            labels = kmeans.fit_predict(X)

            # Skip if only one cluster was found
            if len(np.unique(labels)) < 2:
                continue

            # Compute scores
            sil_score = silhouette_score(X, labels)
            db_score = davies_bouldin_score(X, labels)
            ch_score = calinski_harabasz_score(X, labels)

            result = {
                "n_clusters": n_clusters,
                "init": init_method,
                "silhouette": sil_score,
                "davies_bouldin": db_score,
                "calinski_harabasz": ch_score,
                "inertia": kmeans.inertia_,
            }
            results.append(result)

            # Update best model based on metric
            if metric == "silhouette":
                current_score = sil_score
                is_better = current_score > best_score
            elif metric == "calinski_harabasz":
                current_score = ch_score
                is_better = current_score > best_score
            elif metric == "davies_bouldin":
                current_score = db_score
                is_better = current_score < best_score  # Lower is better
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if is_better:
                best_score = current_score
                best_model = kmeans
                best_params = result.copy()

    best_params["all_results"] = results
    best_params["optimization_metric"] = metric

    return best_model, best_params


def hdbscan_grid_search(
    X: np.ndarray,
    min_cluster_size_range: List[int] = None,
    min_samples_range: List[int] = None,
    cluster_selection_epsilon_range: List[float] = None,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
    optimization_metric: str = "silhouette",
) -> Tuple[hdbscan.HDBSCAN, Dict[str, Any]]:
    """
    Perform grid search for HDBSCAN clustering to find optimal parameters.

    Args:
        X: Input data of shape (n_samples, n_features)
        min_cluster_size_range: List of min_cluster_size values to try. Default: [5, 10, 15, 20, 30]
        min_samples_range: List of min_samples values to try. Default: [None, 1, 3, 5, 10]
        cluster_selection_epsilon_range: List of epsilon values. Default: [0.0, 0.1, 0.3, 0.5]
        metric: Distance metric to use
        cluster_selection_method: Method to select clusters ('eom' or 'leaf')
        optimization_metric: Metric to optimize ('silhouette', 'calinski_harabasz', 'davies_bouldin')

    Returns:
        best_model: Best HDBSCAN model found
        best_params: Dictionary containing best parameters and scores
    """
    if min_cluster_size_range is None:
        min_cluster_size_range = [20, 30, 50, 100, 150]

    if min_samples_range is None:
        min_samples_range = [None, 1, 3, 5, 10]

    if cluster_selection_epsilon_range is None:
        cluster_selection_epsilon_range = [0.0, 0.1, 0.3, 0.5]

    best_score = -np.inf if optimization_metric != "davies_bouldin" else np.inf
    best_model = None
    best_params = {}
    results = []

    for min_cluster_size in min_cluster_size_range:
        for min_samples in min_samples_range:
            for epsilon in cluster_selection_epsilon_range:
                # Fit HDBSCAN
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_epsilon=epsilon,
                    metric=metric,
                    cluster_selection_method=cluster_selection_method,
                )
                labels = clusterer.fit_predict(X)

                # Filter out noise points (label = -1) for quality metrics
                mask = labels != -1
                n_noise = np.sum(~mask)
                n_clusters = len(np.unique(labels[mask])) if np.any(mask) else 0

                # Skip if less than 2 clusters or too many noise points
                if n_clusters < 2 or n_noise > len(labels) * 0.5:
                    continue

                # Compute scores only on non-noise points
                X_filtered = X[mask]
                labels_filtered = labels[mask]

                try:
                    sil_score = silhouette_score(X_filtered, labels_filtered)
                    db_score = davies_bouldin_score(X_filtered, labels_filtered)
                    ch_score = calinski_harabasz_score(X_filtered, labels_filtered)
                except:
                    # Skip if scoring fails
                    continue

                result = {
                    "min_cluster_size": min_cluster_size,
                    "min_samples": min_samples,
                    "cluster_selection_epsilon": epsilon,
                    "n_clusters": n_clusters,
                    "n_noise": n_noise,
                    "noise_ratio": n_noise / len(labels),
                    "silhouette": sil_score,
                    "davies_bouldin": db_score,
                    "calinski_harabasz": ch_score,
                }
                results.append(result)

                # Update best model based on metric
                if optimization_metric == "silhouette":
                    current_score = sil_score
                    is_better = current_score > best_score
                elif optimization_metric == "calinski_harabasz":
                    current_score = ch_score
                    is_better = current_score > best_score
                elif optimization_metric == "davies_bouldin":
                    current_score = db_score
                    is_better = current_score < best_score  # Lower is better
                else:
                    raise ValueError(f"Unknown metric: {optimization_metric}")

                if is_better:
                    best_score = current_score
                    best_model = clusterer
                    best_params = result.copy()

    best_params["all_results"] = results
    best_params["optimization_metric"] = optimization_metric

    return best_model, best_params


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
        compute_hopkins: If True, compute Hopkins statistic for clustering tendency
        random_state: Random state for Hopkins statistic computation

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
