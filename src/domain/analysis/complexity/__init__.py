import numpy as np
from tqdm import tqdm
import logging

from src.domain.analysis.complexity.shared import (
    build_knn_graph,
    topk_adversarial_clusters,
)
from src.domain.analysis.complexity.clusters import compute_cluster_geometry
from src.domain.analysis.complexity.feature import compute_f_measures
from src.domain.analysis.complexity.neighborhood import compute_n_measures
from src.domain.analysis.complexity.network import compute_network_measures
from src.domain.analysis.complexity.dimensionality import compute_t_measures

from src.core.utils import timed

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
    metric: str = "cosine",
) -> dict[str, list[str]]:
    """Build the top-K adversarial cluster map keyed by str(cluster_id).

    Selects the K nearest adversarial clusters by centroid distance under
    `metric`. Only clusters present in both `cluster_to_class` and `centroids`
    are eligible.
    """
    present_ids = [cid for cid in cluster_to_class if cid in centroids]
    if not present_ids:
        return {}
    centroid_matrix = np.stack(
        [np.asarray(centroids[cid], dtype=np.float64) for cid in present_ids]
    )
    id_to_class = {cid: cluster_to_class[cid] for cid in present_ids}
    return topk_adversarial_clusters(
        centroid_matrix, present_ids, id_to_class, top_k_clusters, metric=metric
    )


def _compute_analysis_centroids(
    X_num: np.ndarray,
    y_cluster: np.ndarray,
    metric: str,
    eps: float = 1e-8,
) -> dict[str, list[float]]:
    """Compute centroids appropriate for the configured metric.

    metric="cosine": spherical centroid (mean of L2-normalised samples, re-normalised).
    metric="euclidean": arithmetic mean.
    Returns {str(cluster_id): centroid_vector}.
    """
    result: dict[str, list[float]] = {}
    for cid in np.unique(y_cluster):
        if int(cid) == -1:
            continue
        X_c = X_num[y_cluster == int(cid)]
        if len(X_c) == 0:
            continue
        if metric == "cosine":
            norms = np.linalg.norm(X_c, axis=1, keepdims=True)
            X_c_norm = X_c / np.maximum(norms, eps)
            sph = X_c_norm.mean(axis=0)
            sph_norm = np.linalg.norm(sph)
            result[str(int(cid))] = (sph / max(sph_norm, eps)).tolist()
        else:
            result[str(int(cid))] = X_c.mean(axis=0).tolist()
    return result


@timed
def compute_all_complexity_measures(
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    y_class: np.ndarray,
    y_cluster: np.ndarray,
    centroids: dict[str, list[float]],
    *,
    k: int = 30,
    top_k_clusters: int = 10,
    max_samples: int | None = None,
    min_per_cluster: int = 50,
    metric: str = "cosine",
    noise_cluster_ids: set[int] | None = None,
    random_state: int = 42,
) -> dict[str, dict[str, float | None]]:
    """Compute all complexity measures per cluster under a single metric.

    Builds one Gower-hybrid k-NN graph and passes it to all measure families.
    All families (F, N, ND, G) use the same metric — there are no dual-metric
    outputs. Output keys are neutral (no _cosine / _euclidean suffix).

    Inputs:
        X_num            — (n, d_num) float array, RobustScaled numericals.
        X_cat            — (n, d_cat) int array or None.
        y_class          — (n,) int array, class labels (encoded).
        y_cluster        — (n,) int array, cluster labels (-1 = noise).
        centroids        — {str(cluster_id): [float, ...]} Euclidean centroids
                           (from _cluster_per_class; used only for the Euclidean
                           path — cosine path recomputes spherical centroids).
        k                — number of neighbours for the k-NN graph.
        top_k_clusters   — K nearest adversarial clusters for vs-cluster aggregation.
        max_samples      — if set, subsample stratified by cluster before building
                           the k-NN graph. None = use all samples.
        min_per_cluster  — minimum samples per cluster in the subsample.
        metric           — "cosine" or "euclidean". Controls k-NN, MST, F-family,
                           G-family centroids, pairwise distances, and silhouette.
        noise_cluster_ids — set of pseudo-cluster IDs (reassigned noise points).
                           Adds is_noise_cluster flag to the output rows.
        random_state     — seed for stratified subsampling and silhouette.

    Returns {cluster_id: {measure_name: value}}.
    """
    if max_samples is not None and len(y_cluster) > max_samples:
        n_orig = len(y_cluster)
        X_num, X_cat, y_class, y_cluster = _stratified_subsample(
            X_num, X_cat, y_class, y_cluster, max_samples, min_per_cluster,
            random_state=random_state,
        )
        logger.info(
            "Subsampled %d → %d points (proportional, min %d/cluster)",
            n_orig,
            len(y_cluster),
            min_per_cluster,
        )

    logger.info("Building Gower-%s hybrid k-NN graph (k=%d)...", metric, k)
    knn_idx, knn_dist = build_knn_graph(X_num, X_cat, k=k, metric=metric)

    class_mask, cluster_mask, cluster_to_class = _build_population_masks(
        y_class, y_cluster
    )

    # centroids appropriate for the metric (spherical if cosine, Euclidean otherwise)
    analysis_centroids = _compute_analysis_centroids(X_num, y_cluster, metric=metric)

    top_k_map = _build_topk_map(
        cluster_to_class, analysis_centroids, top_k_clusters, metric=metric
    )

    with tqdm(total=5, desc="complexity families", unit="family") as pbar:
        pbar.set_description("F measures")
        f_out = compute_f_measures(
            X_num, y_class, y_cluster, cluster_to_class, top_k_map, metric=metric
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
            metric=metric,
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
        g_out = compute_cluster_geometry(
            X_num, y_class, y_cluster, centroids,
            metric=metric, random_state=random_state,
        )
        pbar.update(1)

    noise_ids = noise_cluster_ids or set()
    all_ids = set(f_out) | set(n_out) | set(nd_out) | set(t_out) | set(g_out)

    result: dict[str, dict[str, float | None]] = {}
    for cid in all_ids:
        row: dict[str, float | None] = {}
        row.update(f_out.get(cid, {}))
        row.update(n_out.get(cid, {}))
        row.update(nd_out.get(cid, {}))
        row.update(t_out.get(cid, {}))
        row.update(g_out.get(cid, {}))
        row["is_noise_cluster"] = int(cid) in noise_ids
        result[cid] = row

    return result
