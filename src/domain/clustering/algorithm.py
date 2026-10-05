import hdbscan
import numpy as np
from sklearn.cluster import Birch, KMeans, SpectralClustering
from sklearn.neighbors import NearestNeighbors

from src.domain.clustering.base import subsample_features
from src.domain.clustering.factory import ClusteringFactory

_PREDICT_CHUNK = 200_000


@ClusteringFactory.register("hdbscan")
def fit_hdbscan(
    X: np.ndarray,
    *,
    min_cluster_size: int = 50,
    min_samples: int | None = None,
    cluster_selection_method: str = "leaf",
    cluster_selection_epsilon: float = 0.0,
    max_fit_samples: int,
    random_state: int,
) -> np.ndarray:
    """Fit HDBSCAN (Euclidean) and return labels (n,), keeping noise as -1."""
    n_rows = X.shape[0]

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric="euclidean",
        prediction_data=True,
    )

    if n_rows > max_fit_samples:
        X_sub = subsample_features(
            X, max_samples=max_fit_samples, random_state=random_state
        )
        clusterer.fit(X_sub)
        labels = np.concatenate(
            [
                hdbscan.approximate_predict(
                    clusterer, X[start : start + _PREDICT_CHUNK]
                )[0]
                for start in range(0, n_rows, _PREDICT_CHUNK)
            ]
        )
    else:
        clusterer.fit(X)
        labels = clusterer.labels_

    return labels


@ClusteringFactory.register("kmeans")
def fit_kmeans(
    X: np.ndarray,
    *,
    n_clusters: int = 8,
    max_fit_samples: int,
    random_state: int,
) -> np.ndarray:
    """Fit K-means on at most `max_fit_samples` rows and label every row of X."""
    n_rows = X.shape[0]
    # Bounded by the rows the model is fitted on, which the sweep scored.
    n_clusters = max(2, min(n_clusters, min(n_rows, max_fit_samples) - 1))
    X = np.ascontiguousarray(X, dtype=np.float64)
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    if n_rows > max_fit_samples:
        X_sub = subsample_features(
            X, max_samples=max_fit_samples, random_state=random_state
        )
        model.fit(X_sub)
        return model.predict(X)
    return model.fit_predict(X)


@ClusteringFactory.register("birch")
def fit_birch(
    X: np.ndarray,
    *,
    n_clusters: int = 8,
    threshold: float = 0.5,
    branching_factor: int = 50,
    max_fit_samples: int,
    random_state: int,
) -> np.ndarray:
    """Fit BIRCH with `n_clusters` (AgglomerativeClustering on CF-tree leaves)."""
    n_rows = X.shape[0]
    n_clusters = max(2, min(int(n_clusters), min(n_rows, max_fit_samples) - 1))
    X = np.ascontiguousarray(X, dtype=np.float64)
    clusterer = Birch(
        threshold=threshold,
        branching_factor=branching_factor,
        n_clusters=n_clusters,
    )
    if n_rows > max_fit_samples:
        X_sub = subsample_features(
            X, max_samples=max_fit_samples, random_state=random_state
        )
        clusterer.fit(X_sub)
        labels = clusterer.predict(X)
    else:
        clusterer.fit(X)
        labels = clusterer.labels_
    return labels


@ClusteringFactory.register("spectral")
def fit_spectral(
    X: np.ndarray,
    *,
    n_clusters: int = 8,
    affinity: str = "rbf",
    gamma: float | None = None,
    n_neighbors: int | None = None,
    max_fit_samples: int,
    random_state: int,
) -> np.ndarray:
    """Spectral clustering, with subsampling and 1-NN propagation above `max_fit_samples`."""
    if gamma is not None and affinity != "rbf":
        raise TypeError(f"fit_spectral: gamma is ignored with affinity={affinity!r}.")
    if n_neighbors is not None and affinity != "nearest_neighbors":
        raise TypeError(
            f"fit_spectral: n_neighbors is ignored with affinity={affinity!r}."
        )
    n_rows = X.shape[0]
    n_clusters = max(2, min(int(n_clusters), min(n_rows, max_fit_samples) - 1))
    X = np.ascontiguousarray(X, dtype=np.float64)

    spectral_params = dict(
        n_clusters=n_clusters,
        affinity=affinity,
        assign_labels="kmeans",
        random_state=random_state,
        eigen_solver="arpack",
    )
    if gamma is not None:
        spectral_params["gamma"] = float(gamma)
    if n_neighbors is not None:
        spectral_params["n_neighbors"] = int(n_neighbors)

    if n_rows <= max_fit_samples:
        return SpectralClustering(**spectral_params).fit_predict(X)

    X_sub = subsample_features(
        X, max_samples=max_fit_samples, random_state=random_state
    )
    sub_labels = SpectralClustering(**spectral_params).fit_predict(X_sub)
    nearest = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(X_sub)
    _, nearest_sample = nearest.kneighbors(X, n_neighbors=1, return_distance=True)
    return sub_labels[nearest_sample.ravel()]
