import numpy as np
from tqdm import tqdm

from src.core.utils import timed
from src.domain.analysis.complexity.sample import MeasuredSample
from src.domain.analysis.complexity.shared import aggregate_min_mean_max


def compute_clustering_coefficient(
    neighbors: np.ndarray, in_population: np.ndarray, knn_idx: np.ndarray
) -> float:
    """Mean share of a sampled row's same-population neighbour pairs that are linked."""
    coefficients = np.zeros(neighbors.shape[0], dtype=np.float64)
    for i, own_mask in enumerate(in_population):
        own_neighbors = neighbors[i, own_mask]
        if own_neighbors.size < 2:
            continue
        triangles = int(np.isin(knn_idx[own_neighbors], own_neighbors).sum())
        coefficients[i] = triangles / (own_neighbors.size * (own_neighbors.size - 1))
    return float(coefficients.mean())


@timed
def compute_network_measures(
    sample: MeasuredSample,
    graph_population: np.ndarray,
    knn_idx: np.ndarray,
    rivals: dict[int, list[int]],
) -> dict[int, dict[str, float | None]]:
    """Network-family measures per population: density, clustering coefficient, hubs.

    `knn_idx` is the graph's own k-NN: hubs are read off it, the rest off the graph
    neighbours of the population's sampled rows.
    """
    k = knn_idx.shape[1]
    in_degree = np.bincount(knn_idx.ravel(), minlength=knn_idx.shape[0])

    result: dict[int, dict[str, float | None]] = {}
    for population_id in tqdm(rivals, desc="ND measures", unit="pop", leave=False):
        neighbors = sample.neighbors[sample.population == population_id]
        in_population = (graph_population == population_id)[neighbors]
        density = [
            float((graph_population == rival)[neighbors].sum())
            / (neighbors.shape[0] * k)
            for rival in rivals[population_id]
        ]
        minimum, mean, maximum = aggregate_min_mean_max(density)
        # No graph node in the population: nothing to average.
        node_in_degrees = in_degree[graph_population == population_id]
        result[population_id] = {
            "network_density_min": minimum,
            "network_density_mean": mean,
            "network_density_max": maximum,
            "cls_coef": compute_clustering_coefficient(
                neighbors, in_population, knn_idx
            ),
            "hub": float(node_in_degrees.mean()) if node_in_degrees.size else None,
        }
    return result
