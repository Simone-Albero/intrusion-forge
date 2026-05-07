import numpy as np
import hdbscan

from src.data.complexity.shared import build_sparse_knn_matrix
from src.ml.clustering.base import ClusterFn, _subsample


def fit_hdbscan(
    X_num: np.ndarray,
    X_cat: np.ndarray | None = None,
    min_cluster_size: int = 50,
    min_samples: int | None = None,
    cluster_selection_method: str = "eom",
    cluster_selection_epsilon: float = 0.0,
    min_clusters: int = 2,
    max_noise_ratio: float = 0.60,
    min_clustered_ratio: float = 0.20,
    penalize: bool = True,
    max_fit_samples: int = 50_000,
    random_state: int = 0,
    **fixed_params,
) -> np.ndarray:
    """Fit HDBSCAN and return labels (n,).

    Gower path (X_cat is not None):
      k = fixed_params.get("k_cluster", 50)
      D_sparse = build_sparse_knn_matrix(X_num, X_cat, k=k)
      HDBSCAN(metric="precomputed").fit(D_sparse)
      No PCA, no approximate_predict.

    Euclidean path (X_cat is None):
      HDBSCAN(metric="euclidean").fit(X_num)
      approximate_predict used if n > max_fit_samples.

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
        metric="precomputed" if X_cat is not None else "euclidean",
        prediction_data=X_cat is None,  # only needed for approximate_predict
    )

    if X_cat is not None:
        k = fixed_params.get("k_cluster", 50)
        D_sparse = build_sparse_knn_matrix(X_num, X_cat, k=k)
        clf.fit(D_sparse)
        labels = clf.labels_
    else:
        if n > max_fit_samples:
            sub_num, _ = _subsample(X_num, None, max_fit_samples, random_state)
            clf.fit(sub_num)
            labels, _ = hdbscan.approximate_predict(clf, X_num)
        else:
            clf.fit(X_num)
            labels = clf.labels_

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


def make_hdbscan_cluster_fn(
    max_fit_samples: int = 50_000,
    random_state: int = 0,
    **best_params,
) -> ClusterFn:
    """Return a ClusterFn closing over fit_hdbscan with fixed best_params.

    No grid search inside — grid_search must be called separately to obtain best_params.
    k_cluster is forwarded inside best_params if present.
    penalize is always False in the returned ClusterFn (used for final assignment, not search).
    """
    # strip penalize if caller accidentally included it — always False in final assignment
    best_params.pop("penalize", None)

    def cluster_fn(X_num: np.ndarray, X_cat: np.ndarray | None) -> np.ndarray:
        return fit_hdbscan(
            X_num,
            X_cat,
            max_fit_samples=max_fit_samples,
            random_state=random_state,
            penalize=False,
            **best_params,
        )

    return cluster_fn
