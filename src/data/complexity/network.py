import numpy as np
from tqdm import tqdm

from ...common.utils import timed


def compute_cls_coef(
    y_cluster: np.ndarray,
    knn_idx: np.ndarray,
) -> dict[str, float]:
    """Clustering coefficient per cluster: fraction of neighbor pairs that are also neighbors."""
    mask_valid = y_cluster != -1
    full_idx = np.where(mask_valid)[0]
    yk_v = y_cluster[mask_valid]
    k = knn_idx.shape[1]
    result: dict[str, float] = {}
    for cid in np.unique(yk_v):
        c_idx = full_idx[yk_v == cid]
        c_set = set(c_idx.tolist())
        coefs = []
        for xi in c_idx:
            nbs = {int(nb) for nb in knn_idx[xi] if int(nb) in c_set and int(nb) != xi}
            if len(nbs) < 2:
                coefs.append(0.0)
                continue
            triangles = sum(
                1
                for nb in nbs
                for nb2 in knn_idx[nb]
                if int(nb2) in nbs and int(nb2) != nb
            )
            possible = len(nbs) * (len(nbs) - 1)
            coefs.append(triangles / possible if possible > 0 else 0.0)
        result[str(cid)] = float(np.mean(coefs)) if coefs else 0.0
    return result


def compute_hub(
    y_cluster: np.ndarray,
    knn_idx: np.ndarray,
) -> dict[str, float]:
    """Hub score per cluster: mean in-degree in the reverse kNN graph (hubness proxy)."""
    n = knn_idx.shape[0]
    in_degree = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for nb in knn_idx[i]:
            in_degree[int(nb)] += 1

    mask_valid = y_cluster != -1
    full_idx = np.where(mask_valid)[0]
    yk_v = y_cluster[mask_valid]
    result: dict[str, float] = {}
    for cid in np.unique(yk_v):
        c_idx = full_idx[yk_v == cid]
        result[str(cid)] = float(np.mean(in_degree[c_idx]))
    return result


def compute_network_density(
    y_class: np.ndarray,
    y_cluster: np.ndarray,
    knn_idx: np.ndarray,
) -> dict[str, dict[str, float | None]]:
    """Cross-class k-NN density per cluster vs each adversarial class.

    network_density(c, j) = Σ_{x ∈ cluster_c} |{nb ∈ NN(x) : nb ∈ class_j}| / (|c| × k)

    Returns {str(cid): {network_density_min: float | None, network_density_mean: float | None, network_density_max: float | None}}.
    """
    k = knn_idx.shape[1]
    mask_valid = y_cluster != -1
    full_idx = np.where(mask_valid)[0]
    yc_v = y_class[mask_valid]
    yk_v = y_cluster[mask_valid]
    all_classes = np.unique(yc_v)

    result: dict[str, dict[str, float | None]] = {}

    for cid in tqdm(np.unique(yk_v), desc="ND measures", unit="cluster", leave=False):
        local_mask = yk_v == cid
        c_full_idx = full_idx[local_mask]
        cls_c = int(yc_v[local_mask][0])
        adversarial = [j for j in all_classes if j != cls_c]

        null_row: dict[str, float | None] = {
            "network_density_min": None,
            "network_density_mean": None,
            "network_density_max": None,
        }

        if not adversarial or len(c_full_idx) == 0:
            result[str(cid)] = null_row
            continue

        densities = []
        for j in adversarial:
            j_full_idx = np.where(y_class == j)[0]
            j_set = set(j_full_idx.tolist())
            cross_edges = sum(
                1 for xi in c_full_idx for nb in knn_idx[xi] if int(nb) in j_set
            )
            densities.append(cross_edges / (len(c_full_idx) * k))

        if not densities:
            result[str(cid)] = null_row
            continue

        result[str(cid)] = {
            "network_density_min": float(np.min(densities)),
            "network_density_mean": float(np.mean(densities)),
            "network_density_max": float(np.max(densities)),
        }

    return result


@timed
def compute_network_measures(
    y_class: np.ndarray,
    y_cluster: np.ndarray,
    knn_idx: np.ndarray,
) -> dict[str, dict[str, float | None]]:
    """Compute network measures per cluster: density, clustering coefficient, hub score.

    Inputs:
        y_class    — (n,) int array, class labels (full dataset).
        y_cluster  — (n,) int array, cluster labels (-1 = noise, excluded from clusters).
        knn_idx    — (n, k) int array, k-NN indices in the full dataset.

    Noise points (y_cluster == -1) may appear as neighbors in knn_idx but are excluded
    from cluster membership.

    Output keys per cluster:
        network_density_min   min cross-class k-NN density over adversarial classes
        network_density_mean  mean cross-class k-NN density over adversarial classes
        network_density_max   max cross-class k-NN density over adversarial classes
        cls_coef              local clustering coefficient
        hub                   mean in-degree in the reverse k-NN graph (hubness proxy)
    """
    density_out = compute_network_density(y_class, y_cluster, knn_idx)
    cls_coef_out = compute_cls_coef(y_cluster, knn_idx)
    hub_out = compute_hub(y_cluster, knn_idx)

    all_ids = set(density_out) | set(cls_coef_out) | set(hub_out)
    result: dict[str, dict[str, float | None]] = {}
    for cid in all_ids:
        row: dict[str, float | None] = {**density_out.get(cid, {})}
        if cid in cls_coef_out:
            row["cls_coef"] = cls_coef_out[cid]
        if cid in hub_out:
            row["hub"] = hub_out[cid]
        result[cid] = row

    return result
