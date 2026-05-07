import numpy as np
from tqdm import tqdm
import logging

from src.data.complexity.shared import build_knn_graph
from src.data.complexity.clusters import compute_cluster_geometry
from src.data.complexity.feature import compute_f_measures
from src.data.complexity.neighborhood import compute_n_measures
from src.data.complexity.network import compute_network_measures
from src.data.complexity.dimensionality import compute_t_measures

from ...common.utils import timed

__all__ = ["compute_all_complexity_measures"]
logger = logging.getLogger(__name__)


def _stratified_subsample(
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    y_class: np.ndarray,
    y_cluster: np.ndarray,
    max_samples: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Stratified subsample by cluster, targeting max_samples total."""
    unique_clusters = np.unique(y_cluster)
    per_cluster = max(1, max_samples // len(unique_clusters))
    rng = np.random.default_rng(random_state)
    idx_parts = []
    for cid in unique_clusters:
        members = np.where(y_cluster == cid)[0]
        idx_parts.append(
            rng.choice(members, size=min(len(members), per_cluster), replace=False)
        )
    sample_idx = np.concatenate(idx_parts)
    return (
        X_num[sample_idx],
        X_cat[sample_idx] if X_cat is not None else None,
        y_class[sample_idx],
        y_cluster[sample_idx],
    )


@timed
def compute_all_complexity_measures(
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    y_class: np.ndarray,
    y_cluster: np.ndarray,
    centroids: dict[str, list[float]],
    k: int = 5,
    max_samples: int | None = None,
) -> dict[str, dict[str, float | None]]:
    """Compute all complexity measures per cluster.

    Builds one k-NN graph (Gower distance, batched) and passes it to all
    measure functions. Returns {cluster_id: {measure_name: value}}.

    Inputs:
        X_num        — (n, d_num) float array, RobustScaled numericals.
        X_cat        — (n, d_cat) int array or None.
        y_class      — (n,) int array, class labels (encoded).
        y_cluster    — (n,) int array, cluster labels (-1 = noise).
        centroids    — {str(cluster_id): [float, ...]} numerical centroids.
        k            — number of neighbors for the k-NN graph.
        max_samples  — if set, subsample stratified by cluster before building
                       the kNN graph. None = use all samples.

    Returns 26 keys per cluster (see plan §5).
    """
    if max_samples is not None and len(y_cluster) > max_samples:
        n_orig = len(y_cluster)
        X_num, X_cat, y_class, y_cluster = _stratified_subsample(
            X_num, X_cat, y_class, y_cluster, max_samples
        )
        logger.info(
            "Subsampled %d → %d points (stratified by cluster)", n_orig, len(y_cluster)
        )

    logger.info("Building k-NN graph...")
    knn_idx, knn_dist = build_knn_graph(X_num, X_cat, k=k)

    with tqdm(total=5, desc="complexity families", unit="family") as pbar:
        pbar.set_description("F measures")
        f_out = compute_f_measures(X_num, y_class, y_cluster)
        pbar.update(1)

        pbar.set_description("N measures")
        n_out = compute_n_measures(y_class, y_cluster, knn_idx, knn_dist)
        pbar.update(1)

        pbar.set_description("ND measures")
        nd_out = compute_network_measures(y_class, y_cluster, knn_idx)
        pbar.update(1)

        pbar.set_description("T measures")
        t_out = compute_t_measures(X_num, X_cat, y_cluster)
        pbar.update(1)

        pbar.set_description("G measures")
        g_out = compute_cluster_geometry(X_num, X_cat, y_class, y_cluster, centroids)
        pbar.update(1)

    # union of cluster IDs across all outputs
    all_ids = set(f_out) | set(n_out) | set(nd_out) | set(t_out) | set(g_out)

    result: dict[str, dict[str, float | None]] = {}
    for cid in all_ids:
        row: dict[str, float | None] = {}
        row.update(f_out.get(cid, {}))
        row.update(n_out.get(cid, {}))
        row.update(nd_out.get(cid, {}))
        row.update(t_out.get(cid, {}))
        row.update(g_out.get(cid, {}))
        result[cid] = row

    return result
