import numpy as np
from tqdm import tqdm

from src.data.complexity.shared import build_approx_mst
from ...common.utils import timed


def _n1_pair(
    c_set: set[int],
    j_set: set[int],
    mst_edges: list[tuple[int, int, float]],
) -> float:
    """N1(c, j): fraction of cluster-c samples with at least one MST edge to class j."""
    boundary = set()
    for u, v, _ in mst_edges:
        if u in c_set and v in j_set:
            boundary.add(u)
        elif v in c_set and u in j_set:
            boundary.add(v)
    return len(boundary) / len(c_set) if c_set else 0.0


def _n2_pair(
    c_indices: np.ndarray,
    j_set: set[int],
    knn_idx: np.ndarray,
    knn_dist: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """N2(c, j): mean intra/(intra + inter) NN ratio.

    For each x in cluster c:
      intra = nearest distance to a same-cluster neighbor (excl. self)
      inter = nearest distance to a class-j neighbor
    Returns mean ratio; skips points with no intra or no inter neighbor in graph.
    Higher = harder (intra ≈ inter means no separation).
    """
    c_set = set(c_indices.tolist())
    ratios = []
    for xi in c_indices:
        nbs = knn_idx[xi]
        ds = knn_dist[xi]
        intra_d = next(
            (d for nb, d in zip(nbs, ds) if int(nb) in c_set and int(nb) != xi), None
        )
        inter_d = next((d for nb, d in zip(nbs, ds) if int(nb) in j_set), None)
        if intra_d is None or inter_d is None:
            continue
        ratios.append(float(intra_d) / (float(intra_d) + float(inter_d) + eps))
    return float(np.mean(ratios)) if ratios else 0.5


def _n3_pair(
    c_indices: np.ndarray,
    c_set: set[int],
    j_set: set[int],
    knn_idx: np.ndarray,
) -> float:
    """N3(c, j): 1-NN error rate on cluster-c samples using neighbors from c union j.

    For each x in c, find the nearest neighbor in (c union j) minus {x}.
    Misclassified if that neighbor is in j.
    Higher = harder.
    """
    misclassified = 0
    valid = 0
    for xi in c_indices:
        for nb in knn_idx[xi]:
            nb = int(nb)
            if nb == xi:
                continue
            if nb in c_set:
                valid += 1
                break
            if nb in j_set:
                valid += 1
                misclassified += 1
                break
    return float(misclassified / valid) if valid > 0 else 0.0


def _n4_pair(
    c_indices: np.ndarray,
    c_set: set[int],
    j_set: set[int],
    knn_idx: np.ndarray,
) -> float:
    """N4(c, j): k-NN majority-vote error rate on cluster-c samples using neighbors from c∪j.

    For each x in c, among its k neighbors, count votes from c (excl. self) and j.
    Misclassified if j-votes > c-votes.
    Higher = harder.
    """
    misclassified = 0
    valid = 0
    for xi in c_indices:
        c_votes = sum(1 for nb in knn_idx[xi] if int(nb) in c_set and int(nb) != xi)
        j_votes = sum(1 for nb in knn_idx[xi] if int(nb) in j_set)
        if c_votes + j_votes == 0:
            continue
        valid += 1
        if j_votes > c_votes:
            misclassified += 1
    return float(misclassified / valid) if valid > 0 else 0.0


@timed
def compute_n_measures(
    y_class: np.ndarray,
    y_cluster: np.ndarray,
    knn_idx: np.ndarray,
    knn_dist: np.ndarray,
) -> dict[str, dict[str, float | None]]:
    """Compute N1-N4 per cluster vs each adversarial class, aggregated as min + mean.

    Inputs:
        y_class    — (n,) int array, class labels (full dataset, including noise).
        y_cluster  — (n,) int array, cluster labels (-1 = noise, excluded).
        knn_idx    — (n, k) int array, k-NN indices in the full dataset.
        knn_dist   — (n, k) float array, corresponding k-NN distances.

    Builds a global approximate MST once (for N1), then computes N2-N4 from the kNN graph.
    Noise points (y_cluster == -1) are excluded from cluster-level computations,
    but their indices may appear in knn_idx (as class-j neighbors).

    Output keys per cluster:
        n1_min, n1_mean, n1_max  (MST boundary fraction — higher = harder)
        n2_min, n2_mean, n2_max  (intra/inter NN ratio — higher = harder)
        n3_min, n3_mean, n3_max  (1-NN error rate — higher = harder)
        n4_min, n4_mean, n4_max  (k-NN majority error rate — higher = harder)
    """
    mst_edges = build_approx_mst(knn_idx, knn_dist)

    mask_valid = y_cluster != -1
    yc_v = y_class[mask_valid]
    yk_v = y_cluster[mask_valid]

    # global indices (in full array) for each cluster and each class
    full_idx = np.where(mask_valid)[0]
    all_classes = np.unique(yc_v)

    result: dict[str, dict[str, float | None]] = {}

    for cid in tqdm(np.unique(yk_v), desc="N measures", unit="cluster", leave=False):
        local_mask = yk_v == cid
        c_full_idx = full_idx[local_mask]
        c_set = set(c_full_idx.tolist())
        cls_c = int(yc_v[local_mask][0])
        adversarial = [j for j in all_classes if j != cls_c]

        null_row: dict[str, float | None] = {
            "n1_min": None,
            "n1_mean": None,
            "n1_max": None,
            "n2_min": None,
            "n2_mean": None,
            "n2_max": None,
            "n3_min": None,
            "n3_mean": None,
            "n3_max": None,
            "n4_min": None,
            "n4_mean": None,
            "n4_max": None,
        }

        if not adversarial or len(c_full_idx) == 0:
            result[str(cid)] = null_row
            continue

        n1_vals, n2_vals, n3_vals, n4_vals = [], [], [], []
        for j in adversarial:
            # class j: all non-noise samples of class j (across all clusters)
            j_full_idx = full_idx[yc_v == j]
            j_set = set(j_full_idx.tolist())
            if not j_set:
                continue
            n1_vals.append(_n1_pair(c_set, j_set, mst_edges))
            n2_vals.append(_n2_pair(c_full_idx, j_set, knn_idx, knn_dist))
            n3_vals.append(_n3_pair(c_full_idx, c_set, j_set, knn_idx))
            n4_vals.append(_n4_pair(c_full_idx, c_set, j_set, knn_idx))

        def _agg(vals: list[float]) -> tuple[float | None, float | None, float | None]:
            if not vals:
                return None, None, None
            return float(np.min(vals)), float(np.mean(vals)), float(np.max(vals))

        n1_min, n1_mean, n1_max = _agg(n1_vals)
        n2_min, n2_mean, n2_max = _agg(n2_vals)
        n3_min, n3_mean, n3_max = _agg(n3_vals)
        n4_min, n4_mean, n4_max = _agg(n4_vals)

        result[str(cid)] = {
            "n1_min": n1_min,
            "n1_mean": n1_mean,
            "n1_max": n1_max,
            "n2_min": n2_min,
            "n2_mean": n2_mean,
            "n2_max": n2_max,
            "n3_min": n3_min,
            "n3_mean": n3_mean,
            "n3_max": n3_max,
            "n4_min": n4_min,
            "n4_mean": n4_mean,
            "n4_max": n4_max,
        }

    return result
