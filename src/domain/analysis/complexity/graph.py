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
from src.domain.analysis.complexity.sample import MeasuredSample
from src.domain.analysis.complexity.shared import (
    build_approx_mst,
    build_knn_graph,
    nearest_rivals,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainGraph:
    """A uniform sample of train, its k-NN graph and its MST: what every population is
    measured against."""

    X: np.ndarray
    knn_idx: np.ndarray
    knn_dist: np.ndarray
    mst_edges: np.ndarray


@timed
def build_train_graph(X: np.ndarray, *, k: int, metric: str) -> TrainGraph:
    """The k-NN graph and the approximate MST of a point cloud."""
    if X.shape[0] < 2:
        raise ValueError(f"A graph of {X.shape[0]} row(s) has no neighbours.")
    logger.info("Building the %s k-NN graph (k=%d)...", metric, k)
    knn_idx, knn_dist = build_knn_graph(X, k=k, metric=metric)
    logger.info("Building approximate MST over the k-NN graph...")
    mst_edges = build_approx_mst(knn_idx, knn_dist, X, metric=metric)
    return TrainGraph(X, knn_idx, knn_dist, mst_edges)


@timed
def compute_population_complexity(
    train_graph: TrainGraph,
    graph_population: np.ndarray,
    sample: MeasuredSample,
    *,
    population_class: dict[int, int],
    centroids: dict[int, np.ndarray],
    sizes: dict[int, int],
    top_k_clusters: int,
    metric: str,
    silhouette_max_samples: int,
    silhouette_min_per_cluster: int,
    random_state: int,
) -> dict[int, dict[str, float | None]]:
    """Every complexity-measure family for each population, keyed by population id."""
    # `graph_population` names each graph node's population; `sample` holds the rows
    # drawn from every population, `sizes` the rows each really has.
    rivals = nearest_rivals(
        centroids, population_class, top_k=top_k_clusters, metric=metric
    )

    with tqdm(total=5, desc="complexity families", unit="family") as pbar:
        pbar.set_description("F measures")
        f_out = compute_f_measures(sample.X, sample.population, rivals, metric=metric)
        pbar.update(1)

        pbar.set_description("N measures")
        n_out = compute_n_measures(
            sample, graph_population, train_graph.mst_edges, rivals
        )
        pbar.update(1)

        pbar.set_description("ND measures")
        nd_out = compute_network_measures(
            sample, graph_population, train_graph.knn_idx, rivals
        )
        pbar.update(1)

        pbar.set_description("T measures")
        t_out = compute_t_measures(sample.X, sample.population, sizes, metric=metric)
        pbar.update(1)

        pbar.set_description("G measures")
        g_out = compute_cluster_geometry(
            sample.X,
            sample.population,
            centroids,
            metric=metric,
            silhouette_max_samples=silhouette_max_samples,
            silhouette_min_per_cluster=silhouette_min_per_cluster,
            random_state=random_state,
            population_class=population_class,
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
        for pid in sorted(population_class)
    }
