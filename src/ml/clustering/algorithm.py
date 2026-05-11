import numpy as np
import hdbscan

from src.ml.clustering.base import _subsample


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
    return_validity: bool = False,
    **fixed_params,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Fit HDBSCAN with Euclidean distance and return labels (n,).

    X_cat is accepted but ignored — clustering is always Euclidean on X_num.
    If n > max_fit_samples: subsample → fit → approximate_predict.
    No logging, no grid search.
    Returns labels of shape (n,). Invalid clusterings return all -1 when penalize=False,
    or raise ValueError when penalize=True and thresholds are violated.
    If return_validity=True, returns (labels, dbcv_score) where dbcv_score is
    relative_validity_ (DBCV). Returns float('-inf') when DBCV is unavailable
    (subsampled fit uses approximate_predict so relative_validity_ is not reliable).
    """
    n = X_num.shape[0]

    clf = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric="euclidean",
        prediction_data=True,
        gen_min_span_tree=True,
    )

    if n > max_fit_samples:
        sub_num, _ = _subsample(X_num, None, max_fit_samples, random_state)
        clf.fit(sub_num)
        labels, _ = hdbscan.approximate_predict(clf, X_num)
        # DBCV is computed on the subsample; not reliable after approximate_predict
        validity = float("-inf")
    else:
        clf.fit(X_num)
        labels = clf.labels_
        rv = getattr(clf, "relative_validity_", None)
        validity = float(rv) if rv is not None and np.isfinite(rv) else float("-inf")

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

    if return_validity:
        return labels, validity
    return labels
