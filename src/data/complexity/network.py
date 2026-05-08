import numpy as np
from tqdm import tqdm

from src.data.complexity.shared import aggregate_min_mean_max
from ...common.utils import timed


def compute_cls_coef(
    cluster_mask: dict[str, np.ndarray],
    knn_idx: np.ndarray,
) -> dict[str, float]:
    """Local clustering coefficient per cluster.

    For each cluster c, average over its members the fraction of intra-c
    neighbour pairs that are themselves connected in the k-NN graph.
    Vectorised via boolean indexing into `knn_idx[intra_nbs]`.
    """
    result: dict[str, float] = {}
    for cid, c_mask in cluster_mask.items():
        c_idx = np.where(c_mask)[0]
        if c_idx.size == 0:
            result[cid] = 0.0
            continue
        nbs = knn_idx[c_idx]
        in_c = c_mask[nbs]
        coefs = np.zeros(c_idx.size, dtype=np.float64)
        for i, intra_row in enumerate(in_c):
            intra = nbs[i, intra_row]
            if intra.size < 2:
                continue
            triangles = int(np.isin(knn_idx[intra], intra).sum())
            coefs[i] = triangles / (intra.size * (intra.size - 1))
        result[cid] = float(coefs.mean())
    return result


def compute_hub(
    cluster_mask: dict[str, np.ndarray],
    knn_idx: np.ndarray,
) -> dict[str, float]:
    """Hub score per cluster: mean in-degree in the reverse kNN graph (hubness proxy).

    Vectorised via np.bincount on the flattened k-NN index matrix.
    """
    n = knn_idx.shape[0]
    in_degree = np.bincount(knn_idx.ravel(), minlength=n)
    return {
        cid: float(in_degree[np.where(c_mask)[0]].mean()) if c_mask.any() else 0.0
        for cid, c_mask in cluster_mask.items()
    }


def _density(nbs: np.ndarray, j_mask: np.ndarray, k: int) -> float:
    """Cross-class k-NN density for cluster c against population mask j_mask."""
    return float(j_mask[nbs].sum()) / (nbs.shape[0] * k)


def compute_network_density(
    knn_idx: np.ndarray,
    class_mask: dict[int, np.ndarray],
    cluster_mask: dict[str, np.ndarray],
    cluster_to_class: dict[str, int],
    top_k_map: dict[str, list[str]],
) -> dict[str, dict[str, float | None]]:
    """Cross-class k-NN density per cluster aggregated against (a) adversarial
    classes and (b) the top-K nearest adversarial clusters.

    density(c, j) = Σ_{x ∈ cluster_c} |{nb ∈ NN(x) : nb ∈ j}| / (|c| × k)

    Excludes noise (y_cluster == -1) from both the cluster being scored and
    from the adversarial population, mirroring the F/N families.
    """
    k = knn_idx.shape[1]
    null_row: dict[str, float | None] = {
        f"network_density_{scope}_{stat}": None
        for scope in ("class", "cluster")
        for stat in ("min", "mean", "max")
    }

    result: dict[str, dict[str, float | None]] = {}
    for cid, c_mask in tqdm(
        cluster_mask.items(), desc="ND measures", unit="cluster", leave=False
    ):
        row = dict(null_row)
        c_idx = np.where(c_mask)[0]
        if c_idx.size == 0:
            result[cid] = row
            continue

        cls_c = cluster_to_class[cid]
        nbs = knn_idx[c_idx]

        class_vals = [
            _density(nbs, m, k)
            for j, m in class_mask.items()
            if j != cls_c and m.any()
        ]
        cluster_vals = [
            _density(nbs, cluster_mask[ac], k)
            for ac in top_k_map.get(cid, [])
            if ac in cluster_mask and cluster_mask[ac].any()
        ]

        for scope, vals in (("class", class_vals), ("cluster", cluster_vals)):
            mn, me, mx = aggregate_min_mean_max(vals)
            row[f"network_density_{scope}_min"] = mn
            row[f"network_density_{scope}_mean"] = me
            row[f"network_density_{scope}_max"] = mx

        result[cid] = row

    return result


@timed
def compute_network_measures(
    knn_idx: np.ndarray,
    class_mask: dict[int, np.ndarray],
    cluster_mask: dict[str, np.ndarray],
    cluster_to_class: dict[str, int],
    top_k_map: dict[str, list[str]],
) -> dict[str, dict[str, float | None]]:
    """Network-family measures per cluster: density (vs class + vs cluster),
    clustering coefficient, hub score.

    Output keys per cluster:
        network_density_class_{min,mean,max}    cross-class kNN density vs adv. classes
        network_density_cluster_{min,mean,max}  cross-class kNN density vs top-K clusters
        cls_coef                                local clustering coefficient
        hub                                     mean in-degree (hubness proxy)
    """
    density_out = compute_network_density(
        knn_idx, class_mask, cluster_mask, cluster_to_class, top_k_map
    )
    cls_coef_out = compute_cls_coef(cluster_mask, knn_idx)
    hub_out = compute_hub(cluster_mask, knn_idx)

    result: dict[str, dict[str, float | None]] = {}
    for cid, row in density_out.items():
        merged: dict[str, float | None] = dict(row)
        if cid in cls_coef_out:
            merged["cls_coef"] = cls_coef_out[cid]
        if cid in hub_out:
            merged["hub"] = hub_out[cid]
        result[cid] = merged
    return result
