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
from src.domain.analysis.complexity.queries import Queries
from src.domain.analysis.complexity.shared import (
    build_approx_mst,
    build_knn_graph,
    topk_adversarial_clusters,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Reference:
    """A uniform sample of train, its k-NN graph and its MST: what every population is
    measured against."""

    X: np.ndarray
    knn_idx: np.ndarray
    knn_dist: np.ndarray
    mst_edges: np.ndarray


@timed
def build_reference(X: np.ndarray, *, k: int, metric: str) -> Reference:
    """The k-NN graph and the approximate MST of a point cloud."""
    if X.shape[0] < 2:
        raise ValueError(f"A reference of {X.shape[0]} row(s) has no neighbours.")
    logger.info("Building the %s k-NN graph (k=%d)...", metric, k)
    knn_idx, knn_dist = build_knn_graph(X, k=k, metric=metric)
    logger.info("Building approximate MST over the k-NN graph...")
    mst_edges = build_approx_mst(knn_idx, knn_dist, X, metric=metric)
    return Reference(X, knn_idx, knn_dist, mst_edges)


def _build_topk_map(
    population_to_class: dict[str, int],
    centroids: dict[str, list[float]],
    *,
    top_k_clusters: int,
    metric: str,
) -> dict[str, list[str]]:
    """Map each population to its K nearest adversarial ones by centroid distance."""
    population_ids = list(population_to_class)
    centroid_matrix = np.stack(
        [np.asarray(centroids[pid], dtype=np.float64) for pid in population_ids]
    )
    return topk_adversarial_clusters(
        centroid_matrix,
        population_ids,
        population_to_class,
        top_k=top_k_clusters,
        metric=metric,
    )


def analysis_centroids(
    X: np.ndarray, population: np.ndarray, *, metric: str, eps: float = 1e-8
) -> dict[str, list[float]]:
    """Per-population centroids: spherical mean for cosine, arithmetic mean otherwise."""
    result: dict[str, list[float]] = {}
    for pid in np.unique(population):
        rows = X[population == int(pid)]
        if metric == "cosine":
            norms = np.linalg.norm(rows, axis=1, keepdims=True)
            sph = (rows / np.maximum(norms, eps)).mean(axis=0)
            result[str(int(pid))] = (sph / max(np.linalg.norm(sph), eps)).tolist()
        else:
            result[str(int(pid))] = rows.mean(axis=0).tolist()
    return result


@timed
def compute_population_complexity(
    reference: Reference,
    ref_population: np.ndarray,
    queries: Queries,
    *,
    population_to_class: dict[str, int],
    centroids: dict[str, list[float]],
    sizes: dict[int, int],
    top_k_clusters: int,
    metric: str,
    silhouette_max_samples: int,
    silhouette_min_per_cluster: int,
    random_state: int,
) -> dict[str, dict[str, float | None]]:
    """Every complexity-measure family for each population, keyed by population id."""
    # `ref_population` names each reference point's population; `queries` holds the rows
    # sampled from every population, `sizes` the rows each really has.
    top_k_map = _build_topk_map(
        population_to_class,
        centroids,
        top_k_clusters=top_k_clusters,
        metric=metric,
    )

    with tqdm(total=5, desc="complexity families", unit="family") as pbar:
        pbar.set_description("F measures")
        f_out = compute_f_measures(
            queries.X, queries.population, top_k_map, metric=metric
        )
        pbar.update(1)

        pbar.set_description("N measures")
        n_out = compute_n_measures(
            queries, ref_population, reference.mst_edges, top_k_map
        )
        pbar.update(1)

        pbar.set_description("ND measures")
        nd_out = compute_network_measures(
            queries, ref_population, reference.knn_idx, top_k_map
        )
        pbar.update(1)

        pbar.set_description("T measures")
        t_out = compute_t_measures(queries.X, queries.population, sizes)
        pbar.update(1)

        pbar.set_description("G measures")
        g_out = compute_cluster_geometry(
            queries.X,
            queries.population,
            centroids,
            metric=metric,
            silhouette_max_samples=silhouette_max_samples,
            silhouette_min_per_cluster=silhouette_min_per_cluster,
            random_state=random_state,
            cluster_to_class=population_to_class,
        )
        pbar.update(1)

    return {
        pid: {
            **f_out[pid],
            **n_out[pid],
            **nd_out[pid],
            **t_out[pid],
            **g_out[pid],
        }
        for pid in sorted(population_to_class, key=int)
    }
