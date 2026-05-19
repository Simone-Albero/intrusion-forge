import numpy as np
import hdbscan
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

from src.domain.clustering.base import _subsample


def fit_hdbscan(
    X_num: np.ndarray,
    *,
    X_cat: np.ndarray | None = None,
    min_cluster_size: int = 50,
    min_samples: int | None = None,
    cluster_selection_method: str = "leaf",
    cluster_selection_epsilon: float = 0.0,
    min_clusters: int = 2,
    max_noise_ratio: float = 0.60,
    min_clustered_ratio: float = 0.20,
    penalize: bool = True,
    max_fit_samples: int = 50_000,
    max_cluster_size: int | None = None,
    random_state: int = 0,
    **fixed_params,
) -> np.ndarray:
    """Fit HDBSCAN with Euclidean distance and return labels (n,).

    X_cat is accepted but ignored — clustering is always Euclidean on X_num.
    If n > max_fit_samples: subsample → fit → approximate_predict.
    No logging, no grid search.
    Returns labels of shape (n,). Invalid clusterings return all -1 when penalize=False,
    or raise ValueError when penalize=True and thresholds are violated.
    """
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
        sub_num, _ = _subsample(X_num, None, max_fit_samples, random_state)
        clf.fit(sub_num)
        labels, _ = hdbscan.approximate_predict(clf, X_num)
    else:
        clf.fit(X_num)
        labels = clf.labels_

    if max_cluster_size is not None:
        for cid in np.unique(labels[labels != -1]):
            members = np.where(labels == cid)[0]
            if len(members) > max_cluster_size:
                labels[members[max_cluster_size:]] = -1
        if len(np.unique(labels[labels != -1])) < 2:
            validity = float("-inf")

    if penalize:
        n_clustered = (labels != -1).sum()
        n_clusters = len(set(labels) - {-1})
        noise_ratio = (labels == -1).sum() / n
        clustered_ratio = n_clustered / n
        if (
            n_clusters < min_clusters
            or noise_ratio > max_noise_ratio
            or clustered_ratio < min_clustered_ratio
        ):
            raise ValueError(
                f"fit_hdbscan: invalid clustering — "
                f"clusters={n_clusters}, noise_ratio={noise_ratio:.2f}, clustered_ratio={clustered_ratio:.2f}"
            )

    return labels


def fit_kmeans(
    X_num: np.ndarray,
    *,
    X_cat: np.ndarray | None = None,
    n_clusters: int = 8,
    random_state: int = 0,
) -> np.ndarray:
    """Fit K-means on X and return labels (n,)."""

    X_num = np.ascontiguousarray(X_num, dtype=np.float64)
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = model.fit_predict(X_num)
    return labels


def fit_gmm(
    X_num: np.ndarray,
    *,
    X_cat: np.ndarray | None = None,
    n_components: int = 4,
    covariance_type: str = "full",
    random_state: int = 0,
) -> np.ndarray:
    """Fit GMM on X and return predicted cluster labels (n,)."""

    X_num = np.ascontiguousarray(X_num, dtype=np.float64)
    model = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=random_state)
    labels = model.fit_predict(X_num)
    return labels


def fit_spectral(
    X_num: np.ndarray,
    *,
    X_cat: np.ndarray | None = None,
    n_clusters: int = 8,
    affinity: str = "rbf",
    n_components: int | None = None,
    random_state: int = 0,
) -> np.ndarray:
    """Fit SpectralClustering on X and return labels (n,).

    Transductive: no predict method. ClusterFn must refit on each call.
    """

    X_num = np.ascontiguousarray(X_num, dtype=np.float64)
    model = SpectralClustering(n_clusters=n_clusters, affinity=affinity, n_components=n_components, random_state=random_state)
    labels = model.fit_predict(X_num)
    return labels


def fit_agglomerative(
    X_num: np.ndarray,
    *,
    X_cat: np.ndarray | None = None,
    n_clusters: int = 8,
    linkage: str = "ward",
) -> np.ndarray:
    """Fit AgglomerativeClustering on X and return labels (n,).

    Transductive: no predict method. ClusterFn must refit on each call.
    """
    X_num = np.ascontiguousarray(X_num, dtype=np.float64)
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X_num)
    return labels


