import numpy as np
from tqdm import tqdm

from src.core.utils import timed
from src.domain.analysis.complexity.sample import MeasuredSample
from src.domain.analysis.complexity.shared import aggregate_min_mean_max

_N_KEYS = ("n1", "n2", "n3", "n4")


def _n1_vec(c_mask: np.ndarray, j_mask: np.ndarray, edges_uv: np.ndarray) -> float:
    """N1: fraction of cluster-c samples sharing an MST edge with the j population."""
    n_c = int(c_mask.sum())
    if edges_uv.shape[0] == 0:
        return 0.0
    u, v = edges_uv[:, 0], edges_uv[:, 1]
    boundary_u = u[c_mask[u] & j_mask[v]]
    boundary_v = v[c_mask[v] & j_mask[u]]
    if boundary_u.size == 0 and boundary_v.size == 0:
        return 0.0
    return float(np.unique(np.concatenate([boundary_u, boundary_v])).size / n_c)


def _n2_vec(
    neighbors: np.ndarray,
    neighbor_dist: np.ndarray,
    in_c: np.ndarray,
    in_j: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """N2: mean intra/(intra+inter) NN distance ratio over cluster-c samples."""
    valid = in_c.any(axis=1) & in_j.any(axis=1)
    if not valid.any():
        return 0.0
    rows = np.arange(neighbors.shape[0])
    intra_d = neighbor_dist[rows, in_c.argmax(axis=1)]
    inter_d = neighbor_dist[rows, in_j.argmax(axis=1)]
    ratios = intra_d[valid] / (intra_d[valid] + inter_d[valid] + eps)
    return float(ratios.mean())


def _n3_vec(
    neighbors: np.ndarray, j_mask: np.ndarray, in_c: np.ndarray, in_j: np.ndarray
) -> float:
    """N3: 1-NN error rate restricted to c ∪ j neighbours."""
    in_cj = in_c | in_j
    valid = in_cj.any(axis=1)
    if not valid.any():
        return 0.0
    rows = np.arange(neighbors.shape[0])
    nearest = neighbors[rows, in_cj.argmax(axis=1)]
    misclassified = j_mask[nearest] & valid
    return float(misclassified.sum() / valid.sum())


def _n4_vec(in_c: np.ndarray, in_j: np.ndarray) -> float:
    """N4: k-NN majority-vote error rate restricted to c ∪ j neighbours."""
    c_votes = in_c.sum(axis=1)
    j_votes = in_j.sum(axis=1)
    valid = (c_votes + j_votes) > 0
    if not valid.any():
        return 0.0
    return float(((j_votes > c_votes) & valid).sum() / valid.sum())


def _pair_metrics(
    neighbors: np.ndarray,
    neighbor_dist: np.ndarray,
    c_mask: np.ndarray,
    j_mask: np.ndarray,
    edges_uv: np.ndarray,
) -> tuple[float, float, float, float]:
    """N1-N4 of population c against one rival."""
    in_c = c_mask[neighbors]
    in_j = j_mask[neighbors]
    n1 = _n1_vec(c_mask, j_mask, edges_uv)
    n2 = _n2_vec(neighbors, neighbor_dist, in_c, in_j)
    n3 = _n3_vec(neighbors, j_mask, in_c, in_j)
    n4 = _n4_vec(in_c, in_j)
    return n1, n2, n3, n4


def _aggregate_pairs(
    neighbors: np.ndarray,
    neighbor_dist: np.ndarray,
    c_mask: np.ndarray,
    population_masks: list[np.ndarray],
    edges_uv: np.ndarray,
) -> dict[str, list[float]]:
    """N1-N4 of a population against each of its rivals."""
    out: dict[str, list[float]] = {k: [] for k in _N_KEYS}
    for j_mask in population_masks:
        n1, n2, n3, n4 = _pair_metrics(
            neighbors, neighbor_dist, c_mask, j_mask, edges_uv
        )
        # No graph node in the population: there is no MST edge to count.
        if c_mask.any():
            out["n1"].append(n1)
        out["n2"].append(n2)
        out["n3"].append(n3)
        out["n4"].append(n4)
    return out


@timed
def compute_n_measures(
    sample: MeasuredSample,
    graph_population: np.ndarray,
    mst_edges: np.ndarray,
    rivals: dict[int, list[int]],
) -> dict[int, dict[str, float | None]]:
    """N1-N4 per population against its nearest rivals, as min/mean/max.

    N1 counts graph nodes on the MST; N2-N4 read the graph neighbours of the
    population's own sampled rows.
    """
    result: dict[int, dict[str, float | None]] = {}
    for pid in tqdm(rivals, desc="N measures", unit="pop", leave=False):
        in_population = sample.population == pid
        agg = _aggregate_pairs(
            sample.neighbors[in_population],
            sample.neighbor_dist[in_population],
            graph_population == pid,
            [graph_population == rival for rival in rivals[pid]],
            mst_edges,
        )
        row: dict[str, float | None] = {}
        for nk in _N_KEYS:
            mn, me, mx = aggregate_min_mean_max(agg[nk])
            row[f"{nk}_min"] = mn
            row[f"{nk}_mean"] = me
            row[f"{nk}_max"] = mx

        result[pid] = row

    return result
