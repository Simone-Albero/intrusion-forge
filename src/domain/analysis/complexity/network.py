import numpy as np
from tqdm import tqdm

from src.core.utils import timed
from src.domain.analysis.complexity.queries import Queries
from src.domain.analysis.complexity.shared import aggregate_min_mean_max, make_null_row


def compute_cls_coef(
    nbs: np.ndarray, in_population: np.ndarray, knn_idx: np.ndarray
) -> float:
    """Mean fraction of a query's intra-population neighbour pairs that are connected."""
    coefs = np.zeros(nbs.shape[0], dtype=np.float64)
    for i, intra_row in enumerate(in_population):
        intra = nbs[i, intra_row]
        if intra.size < 2:
            continue
        triangles = int(np.isin(knn_idx[intra], intra).sum())
        coefs[i] = triangles / (intra.size * (intra.size - 1))
    return float(coefs.mean())


@timed
def compute_network_measures(
    queries: Queries,
    ref_population: np.ndarray,
    knn_idx: np.ndarray,
    top_k_map: dict[str, list[str]],
) -> dict[str, dict[str, float | None]]:
    """Network-family measures per population: density, clustering coefficient, hubs.

    `knn_idx` is the reference's own k-NN graph: hubs are read off it, the rest off the
    reference neighbours of the population's query rows.
    """
    k = knn_idx.shape[1]
    # Mean in-degree in the reverse k-NN graph, a hubness proxy.
    in_degree = np.bincount(knn_idx.ravel(), minlength=knn_idx.shape[0])
    null_row = make_null_row(("network_density",))

    result: dict[str, dict[str, float | None]] = {}
    for pid_str in tqdm(top_k_map, desc="ND measures", unit="pop", leave=False):
        pid = int(pid_str)
        nbs = queries.nbs[queries.population == pid]
        in_population = (ref_population == pid)[nbs]
        density = [
            float((ref_population == int(ap))[nbs].sum()) / (nbs.shape[0] * k)
            for ap in top_k_map[pid_str]
        ]
        row = dict(null_row)
        mn, me, mx = aggregate_min_mean_max(density)
        row["network_density_min"] = mn
        row["network_density_mean"] = me
        row["network_density_max"] = mx
        row["cls_coef"] = compute_cls_coef(nbs, in_population, knn_idx)
        # No reference point in the population: nothing to average.
        members = in_degree[ref_population == pid]
        row["hub"] = float(members.mean()) if members.size else None
        result[pid_str] = row
    return result
