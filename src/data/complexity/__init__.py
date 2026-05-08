import numpy as np
from tqdm import tqdm
import logging

from src.data.complexity.shared import build_knn_graph, topk_adversarial_clusters
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
    min_per_cluster: int = 50,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Stratified subsample by cluster, allocating proportionally to cluster size.

    Each cluster gets at least `min_per_cluster` samples (or all of its members
    if smaller). The remaining budget is allocated proportionally to cluster
    population. The combined size is capped at `max_samples` by trimming the
    largest contributions if needed.
    """
    rng = np.random.default_rng(random_state)
    unique_clusters, counts = np.unique(y_cluster, return_counts=True)
    n_total = int(counts.sum())

    # initial allocation: proportional with floor at min_per_cluster (capped by cluster size)
    raw_alloc = np.maximum(
        min_per_cluster, np.round(max_samples * counts / n_total).astype(int)
    )
    alloc = np.minimum(raw_alloc, counts)

    # if over budget, trim from the largest allocations until we fit
    overflow = int(alloc.sum()) - max_samples
    if overflow > 0:
        for i in np.argsort(-alloc):
            if overflow <= 0:
                break
            slack = max(0, int(alloc[i]) - max(min_per_cluster, 1))
            cut = min(slack, overflow)
            alloc[i] -= cut
            overflow -= cut

    idx_parts = []
    for cid, take in zip(unique_clusters, alloc):
        if take <= 0:
            continue
        members = np.where(y_cluster == cid)[0]
        idx_parts.append(rng.choice(members, size=int(take), replace=False))
    sample_idx = np.concatenate(idx_parts)
    return (
        X_num[sample_idx],
        X_cat[sample_idx] if X_cat is not None else None,
        y_class[sample_idx],
        y_cluster[sample_idx],
    )


def _build_population_masks(
    y_class: np.ndarray, y_cluster: np.ndarray
) -> tuple[dict[int, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    """Build boolean masks reused by all pairwise families.

    Returns:
        class_mask        : {class_label: (n,) bool}, restricted to non-noise points.
        cluster_mask      : {str(cluster_id): (n,) bool}, restricted to non-noise points.
        cluster_to_class  : {str(cluster_id): class_label} for non-noise clusters.
    """
    mask_valid = y_cluster != -1
    yc_v = y_class[mask_valid]
    yk_v = y_cluster[mask_valid]

    class_mask = {
        int(j): (y_class == int(j)) & mask_valid for j in np.unique(yc_v)
    }
    cluster_mask: dict[str, np.ndarray] = {}
    cluster_to_class: dict[str, int] = {}
    for cid in np.unique(yk_v):
        cid_str = str(int(cid))
        cluster_mask[cid_str] = y_cluster == int(cid)
        cluster_to_class[cid_str] = int(yc_v[yk_v == cid][0])
    return class_mask, cluster_mask, cluster_to_class


def _build_topk_map(
    cluster_to_class: dict[str, int],
    centroids: dict[str, list[float]],
    top_k_clusters: int,
) -> dict[str, list[str]]:
    """Build the top-K adversarial cluster map keyed by str(cluster_id).

    Only clusters present both in `cluster_to_class` and in `centroids` are
    eligible.
    """
    present_ids = [cid for cid in cluster_to_class if cid in centroids]
    if not present_ids:
        return {}
    centroid_matrix = np.stack(
        [np.asarray(centroids[cid], dtype=np.float64) for cid in present_ids]
    )
    id_to_class = {cid: cluster_to_class[cid] for cid in present_ids}
    return topk_adversarial_clusters(
        centroid_matrix, present_ids, id_to_class, top_k_clusters
    )


@timed
def compute_all_complexity_measures(
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    y_class: np.ndarray,
    y_cluster: np.ndarray,
    centroids: dict[str, list[float]],
    k: int = 30,
    top_k_clusters: int = 10,
    max_samples: int | None = None,
    min_per_cluster: int = 50,
) -> dict[str, dict[str, float | None]]:
    """Compute all complexity measures per cluster.

    Builds one k-NN graph (Gower distance, batched) and passes it to all
    measure functions. F/N/ND families are aggregated both vs adversarial
    classes and vs the top-K nearest adversarial clusters (by centroid
    Euclidean distance). Returns {cluster_id: {measure_name: value}}.

    Inputs:
        X_num            — (n, d_num) float array, RobustScaled numericals.
        X_cat            — (n, d_cat) int array or None.
        y_class          — (n,) int array, class labels (encoded).
        y_cluster        — (n,) int array, cluster labels (-1 = noise).
        centroids        — {str(cluster_id): [float, ...]} numerical centroids.
        k                — number of neighbours for the k-NN graph.
        top_k_clusters   — number of nearest adversarial clusters considered
                           in the vs-cluster aggregation of F/N/ND.
        max_samples      — if set, subsample stratified by cluster (proportional
                           with floor at `min_per_cluster`) before building the
                           k-NN graph. None = use all samples.
        min_per_cluster  — minimum samples per cluster in the subsample.
    """
    if max_samples is not None and len(y_cluster) > max_samples:
        n_orig = len(y_cluster)
        X_num, X_cat, y_class, y_cluster = _stratified_subsample(
            X_num, X_cat, y_class, y_cluster, max_samples, min_per_cluster
        )
        logger.info(
            "Subsampled %d → %d points (proportional, min %d/cluster)",
            n_orig,
            len(y_cluster),
            min_per_cluster,
        )

    logger.info("Building k-NN graph (k=%d)...", k)
    knn_idx, knn_dist = build_knn_graph(X_num, X_cat, k=k)

    class_mask, cluster_mask, cluster_to_class = _build_population_masks(
        y_class, y_cluster
    )
    top_k_map = _build_topk_map(cluster_to_class, centroids, top_k_clusters)

    with tqdm(total=5, desc="complexity families", unit="family") as pbar:
        pbar.set_description("F measures")
        f_out = compute_f_measures(
            X_num, y_class, y_cluster, cluster_to_class, top_k_map
        )
        pbar.update(1)

        pbar.set_description("N measures")
        n_out = compute_n_measures(
            y_cluster,
            knn_idx,
            knn_dist,
            X_num,
            X_cat,
            class_mask,
            cluster_mask,
            cluster_to_class,
            top_k_map,
        )
        pbar.update(1)

        pbar.set_description("ND measures")
        nd_out = compute_network_measures(
            knn_idx, class_mask, cluster_mask, cluster_to_class, top_k_map
        )
        pbar.update(1)

        pbar.set_description("T measures")
        t_out = compute_t_measures(X_num, X_cat, y_cluster)
        pbar.update(1)

        pbar.set_description("G measures")
        g_out = compute_cluster_geometry(X_num, y_class, y_cluster, centroids)
        pbar.update(1)

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
