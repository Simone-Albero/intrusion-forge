import logging
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from src.core.utils import timed
from src.domain.analysis.complexity.clusters import compute_cluster_geometry
from src.domain.analysis.complexity.dimensionality import compute_t_measures
from src.domain.analysis.complexity.feature import compute_f_measures
from src.domain.analysis.complexity.neighborhood import compute_n_measures
from src.domain.analysis.complexity.network import compute_network_measures
from src.domain.analysis.complexity.shared import (
    build_approx_mst,
    build_knn_graph,
    topk_adversarial_clusters,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComplexityGraph:
    """Subsampled point cloud, its partition-independent k-NN graph, and its MST."""

    X_num: np.ndarray
    X_cat: np.ndarray | None
    y_class: np.ndarray
    y_cluster: np.ndarray
    knn_idx: np.ndarray
    knn_dist: np.ndarray
    mst_edges: np.ndarray


def _stratified_subsample(
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    y_class: np.ndarray,
    y_cluster: np.ndarray,
    *,
    max_samples: int,
    min_per_cluster: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Subsample proportionally to cluster size, with a floor of `min_per_cluster` each."""
    rng = np.random.default_rng(random_state)
    unique_clusters, counts = np.unique(y_cluster, return_counts=True)
    n_total = int(counts.sum())

    raw_alloc = np.maximum(
        min_per_cluster, np.round(max_samples * counts / n_total).astype(int)
    )
    alloc = np.minimum(raw_alloc, counts)

    overflow = int(alloc.sum()) - max_samples
    if overflow > 0:
        for i in np.argsort(-alloc):
            if overflow <= 0:
                break
            slack = max(0, int(alloc[i]) - max(min_per_cluster, 1))
            cut = min(slack, overflow)
            alloc[i] -= cut
            overflow -= cut
        if overflow > 0:
            floor_total = int(np.minimum(counts, min_per_cluster).sum())
            raise ValueError(
                f"{len(unique_clusters)} clusters need {floor_total} points at their "
                f"min_per_cluster={min_per_cluster} floor, over max_samples={max_samples} "
                "even after trimming every cluster to its floor — lower min_per_cluster, "
                "raise max_complexity_samples, or reduce the cluster count."
            )

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
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Boolean mask and class of every cluster."""
    cluster_mask: dict[str, np.ndarray] = {}
    cluster_to_class: dict[str, int] = {}
    for cid in np.unique(y_cluster):
        cid_str = str(int(cid))
        cluster_mask[cid_str] = y_cluster == int(cid)
        cluster_to_class[cid_str] = int(y_class[cluster_mask[cid_str]][0])
    return cluster_mask, cluster_to_class


def _build_topk_map(
    cluster_to_class: dict[str, int],
    centroids: dict[str, list[float]],
    *,
    top_k_clusters: int,
    metric: str,
) -> dict[str, list[str]]:
    """Map each cluster to its K nearest adversarial clusters by centroid distance."""
    cluster_ids = list(cluster_to_class)
    centroid_matrix = np.stack(
        [np.asarray(centroids[cid], dtype=np.float64) for cid in cluster_ids]
    )
    return topk_adversarial_clusters(
        centroid_matrix,
        cluster_ids,
        cluster_to_class,
        top_k=top_k_clusters,
        metric=metric,
    )


def _compute_analysis_centroids(
    X_num: np.ndarray,
    y_cluster: np.ndarray,
    *,
    metric: str,
    eps: float = 1e-8,
) -> dict[str, list[float]]:
    """Per-cluster centroids: spherical mean for cosine, arithmetic mean otherwise."""
    result: dict[str, list[float]] = {}
    for cid in np.unique(y_cluster):
        X_c = X_num[y_cluster == int(cid)]
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
def prepare_complexity_graph(
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    y_class: np.ndarray,
    y_cluster: np.ndarray,
    *,
    k: int,
    max_samples: int,
    min_per_cluster: int,
    metric: str,
    random_state: int,
) -> ComplexityGraph:
    """Build the subsample, k-NN graph and MST that both complexity passes share."""
    if len(y_cluster) > max_samples:
        n_orig = len(y_cluster)
        X_num, X_cat, y_class, y_cluster = _stratified_subsample(
            X_num,
            X_cat,
            y_class,
            y_cluster,
            max_samples=max_samples,
            min_per_cluster=min_per_cluster,
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
    logger.info("Building approximate MST over the k-NN graph...")
    mst_edges = build_approx_mst(knn_idx, knn_dist, X_num, X_cat, metric=metric)
    return ComplexityGraph(
        X_num, X_cat, y_class, y_cluster, knn_idx, knn_dist, mst_edges
    )


@timed
def compute_complexity_from_graph(
    graph: ComplexityGraph,
    y_partition: np.ndarray,
    *,
    top_k_clusters: int,
    metric: str,
    silhouette_max_samples: int,
    silhouette_min_per_cluster: int,
    random_state: int,
) -> dict[str, dict[str, float | None]]:
    """Compute every complexity-measure family for one partition of `graph`."""
    X_num, X_cat, y_class = graph.X_num, graph.X_cat, graph.y_class
    knn_idx, knn_dist = graph.knn_idx, graph.knn_dist

    cluster_mask, cluster_to_class = _build_population_masks(y_class, y_partition)
    analysis_centroids = _compute_analysis_centroids(X_num, y_partition, metric=metric)

    top_k_map = _build_topk_map(
        cluster_to_class,
        analysis_centroids,
        top_k_clusters=top_k_clusters,
        metric=metric,
    )

    with tqdm(total=5, desc="complexity families", unit="family") as pbar:
        pbar.set_description("F measures")
        f_out = compute_f_measures(X_num, y_partition, top_k_map, metric=metric)
        pbar.update(1)

        pbar.set_description("N measures")
        n_out = compute_n_measures(
            knn_idx,
            knn_dist,
            graph.mst_edges,
            cluster_mask,
            top_k_map,
        )
        pbar.update(1)

        pbar.set_description("ND measures")
        nd_out = compute_network_measures(knn_idx, cluster_mask, top_k_map)
        pbar.update(1)

        pbar.set_description("T measures")
        t_out = compute_t_measures(X_num, X_cat, y_partition)
        pbar.update(1)

        pbar.set_description("G measures")
        g_out = compute_cluster_geometry(
            X_num,
            y_partition,
            analysis_centroids,
            metric=metric,
            silhouette_max_samples=silhouette_max_samples,
            silhouette_min_per_cluster=silhouette_min_per_cluster,
            random_state=random_state,
            cluster_to_class=cluster_to_class,
        )
        pbar.update(1)

    result: dict[str, dict[str, float | None]] = {}
    for cid in sorted(cluster_mask, key=int):
        result[cid] = {
            **f_out[cid],
            **n_out[cid],
            **nd_out[cid],
            **t_out[cid],
            **g_out[cid],
        }

    return result
