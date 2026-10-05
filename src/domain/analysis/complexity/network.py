import numpy as np
from tqdm import tqdm

from src.core.utils import timed
from src.domain.analysis.complexity.sample import MeasuredSample
from src.domain.analysis.complexity.shared import aggregate_min_mean_max


def compute_cls_coef(
    neighbors: np.ndarray, in_population: np.ndarray, knn_idx: np.ndarray
) -> float:
    """Mean share of a sampled row's same-population neighbour pairs that are linked."""
    coefs = np.zeros(neighbors.shape[0], dtype=np.float64)
    for i, intra_row in enumerate(in_population):
        intra = neighbors[i, intra_row]
        if intra.size < 2:
            continue
        triangles = int(np.isin(knn_idx[intra], intra).sum())
        coefs[i] = triangles / (intra.size * (intra.size - 1))
    return float(coefs.mean())


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
    # Mean in-degree in the reverse k-NN graph, a hubness proxy.
    in_degree = np.bincount(knn_idx.ravel(), minlength=knn_idx.shape[0])

    result: dict[int, dict[str, float | None]] = {}
    for pid in tqdm(rivals, desc="ND measures", unit="pop", leave=False):
        neighbors = sample.neighbors[sample.population == pid]
        in_population = (graph_population == pid)[neighbors]
        density = [
            float((graph_population == rival)[neighbors].sum())
            / (neighbors.shape[0] * k)
            for rival in rivals[pid]
        ]
        mn, me, mx = aggregate_min_mean_max(density)
        # No graph node in the population: nothing to average.
        members = in_degree[graph_population == pid]
        result[pid] = {
            "network_density_min": mn,
            "network_density_mean": me,
            "network_density_max": mx,
            "cls_coef": compute_cls_coef(neighbors, in_population, knn_idx),
            "hub": float(members.mean()) if members.size else None,
        }
    return result
