import numpy as np
from tqdm import tqdm

from src.ml.clustering.base import ClusterFn

from ...common.utils import timed


@timed
def make_clusters(
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    y_class: np.ndarray,
    classes: list,
    cluster_fn: ClusterFn,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Cluster all samples intra-class; return (labels, centroids).

    Runs cluster_fn(X_num[mask], X_cat[mask]) per class, applies monotone
    global label offsets so cluster IDs are unique across classes.
    Centroids are numerical only: mean of X_num per cluster.
    Returns:
      labels    — shape (n,), cluster ID per sample (-1 = noise)
      centroids — {cluster_id: centroid_array}
    """
    n = X_num.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    centroids: dict[int, np.ndarray] = {}
    offset = 0

    for cls in tqdm(classes, desc="Clustering classes"):
        mask = y_class == cls
        if not mask.any():
            continue

        X_num_cls = X_num[mask]
        X_cat_cls = X_cat[mask] if X_cat is not None else None

        raw_labels = cluster_fn(X_num_cls, X_cat_cls)

        # shift non-noise labels by current offset
        cluster_ids = np.unique(raw_labels[raw_labels != -1])
        shifted = np.where(raw_labels == -1, -1, raw_labels + offset)
        labels[mask] = shifted

        for cid in cluster_ids:
            global_cid = int(cid + offset)
            centroids[global_cid] = X_num_cls[raw_labels == cid].mean(axis=0)

        if len(cluster_ids) > 0:
            offset += int(cluster_ids.max()) + 1

    return labels, centroids
