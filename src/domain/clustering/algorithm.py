import hdbscan
import numpy as np
from sklearn.cluster import Birch, KMeans, SpectralClustering
from sklearn.neighbors import NearestNeighbors

from src.domain.clustering import ClusteringFactory
from src.domain.clustering.base import subsample_features

_PREDICT_CHUNK = 200_000


@ClusteringFactory.register("hdbscan")
def fit_hdbscan(
    X_num: np.ndarray,
    *,
    X_cat: np.ndarray | None = None,
    min_cluster_size: int = 50,
    min_samples: int | None = None,
    cluster_selection_method: str = "leaf",
    cluster_selection_epsilon: float = 0.0,
    max_fit_samples: int = 50_000,
    random_state: int = 0,
    **fixed_params,
) -> np.ndarray:
    """Fit HDBSCAN (Euclidean) and return labels (n,), keeping noise as -1."""
    n = X_num.shape[0]

    clf = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric="euclidean",
        prediction_data=True,
    )

    if n > max_fit_samples:
        sub_num, _ = subsample_features(X_num, None, max_fit_samples, random_state)
        clf.fit(sub_num)
        labels = np.concatenate(
            [
                hdbscan.approximate_predict(clf, X_num[start : start + _PREDICT_CHUNK])[
                    0
                ]
                for start in range(0, n, _PREDICT_CHUNK)
            ]
        )
    else:
        clf.fit(X_num)
        labels = clf.labels_

    return labels


@ClusteringFactory.register("kmeans")
def fit_kmeans(
    X_num: np.ndarray,
    *,
    X_cat: np.ndarray | None = None,
    n_clusters: int = 8,
    random_state: int = 0,
    **_,
) -> np.ndarray:
    """Fit K-means on X and return labels (n,)."""
    n_clusters = max(2, min(n_clusters, X_num.shape[0] - 1))
    X_num = np.ascontiguousarray(X_num, dtype=np.float64)
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = model.fit_predict(X_num)
    return labels


@ClusteringFactory.register("birch")
def fit_birch(
    X_num: np.ndarray,
    *,
    X_cat: np.ndarray | None = None,
    n_clusters: int = 8,
    threshold: float = 0.5,
    branching_factor: int = 50,
    max_fit_samples: int = 50_000,
    random_state: int = 0,
    **_,
) -> np.ndarray:
    """Fit BIRCH with `n_clusters` (AgglomerativeClustering on CF-tree leaves)."""
    n = X_num.shape[0]
    n_clusters = max(2, min(int(n_clusters), n - 1))
    X_num = np.ascontiguousarray(X_num, dtype=np.float64)
    clf = Birch(
        threshold=threshold,
        branching_factor=branching_factor,
        n_clusters=n_clusters,
    )
    if n > max_fit_samples:
        sub_num, _sub = subsample_features(X_num, None, max_fit_samples, random_state)
        clf.fit(sub_num)
        labels = clf.predict(X_num)
    else:
        clf.fit(X_num)
        labels = clf.labels_
    return labels


@ClusteringFactory.register("spectral")
def fit_spectral(
    X_num: np.ndarray,
    *,
    X_cat: np.ndarray | None = None,
    n_clusters: int = 8,
    affinity: str = "rbf",
    gamma: float | None = None,
    n_neighbors: int = 10,
    max_fit_samples: int = 10_000,
    random_state: int = 0,
    **_,
) -> np.ndarray:
    """Spectral clustering, with subsampling and 1-NN propagation above `max_fit_samples`."""
    n = X_num.shape[0]
    n_clusters = max(2, min(int(n_clusters), n - 1))
    X_num = np.ascontiguousarray(X_num, dtype=np.float64)

    spec_kwargs = dict(
        n_clusters=n_clusters,
        affinity=affinity,
        assign_labels="kmeans",
        random_state=random_state,
        eigen_solver="arpack",
    )
    if gamma is not None and affinity == "rbf":
        spec_kwargs["gamma"] = float(gamma)
    if affinity == "nearest_neighbors":
        spec_kwargs["n_neighbors"] = int(n_neighbors)

    if n <= max_fit_samples:
        return SpectralClustering(**spec_kwargs).fit_predict(X_num)

    sub_num, _ = subsample_features(X_num, None, max_fit_samples, random_state)
    sub_labels = SpectralClustering(**spec_kwargs).fit_predict(sub_num)
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(sub_num)
    _, idx = nn.kneighbors(X_num, n_neighbors=1, return_distance=True)
    return sub_labels[idx.ravel()]
