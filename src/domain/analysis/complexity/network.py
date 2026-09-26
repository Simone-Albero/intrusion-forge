import numpy as np
from tqdm import tqdm

from src.core.utils import timed
from src.domain.analysis.complexity.shared import aggregate_min_mean_max, make_null_row


def compute_cls_coef(
    cluster_mask: dict[str, np.ndarray],
    knn_idx: np.ndarray,
) -> dict[str, float]:
    """Mean fraction of a member's intra-cluster neighbour pairs that are connected."""
    result: dict[str, float] = {}
    for cid, c_mask in cluster_mask.items():
        c_idx = np.where(c_mask)[0]
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
    """Hub score per cluster: mean in-degree in the reverse kNN graph (hubness proxy)."""
    n = knn_idx.shape[0]
    in_degree = np.bincount(knn_idx.ravel(), minlength=n)
    return {
        cid: float(in_degree[np.where(c_mask)[0]].mean())
        for cid, c_mask in cluster_mask.items()
    }


def compute_network_density(
    knn_idx: np.ndarray,
    cluster_mask: dict[str, np.ndarray],
    top_k_map: dict[str, list[str]],
) -> dict[str, dict[str, float | None]]:
    """Cross-class k-NN density per cluster against its top-K adversarial clusters."""
    k = knn_idx.shape[1]
    null_row = make_null_row(("network_density",))

    result: dict[str, dict[str, float | None]] = {}
    for cid, c_mask in tqdm(
        cluster_mask.items(), desc="ND measures", unit="cluster", leave=False
    ):
        row = dict(null_row)
        nbs = knn_idx[np.where(c_mask)[0]]
        cluster_vals = [
            float(cluster_mask[ac][nbs].sum()) / (nbs.shape[0] * k)
            for ac in top_k_map[cid]
        ]

        mn, me, mx = aggregate_min_mean_max(cluster_vals)
        row["network_density_min"] = mn
        row["network_density_mean"] = me
        row["network_density_max"] = mx

        result[cid] = row

    return result


@timed
def compute_network_measures(
    knn_idx: np.ndarray,
    cluster_mask: dict[str, np.ndarray],
    top_k_map: dict[str, list[str]],
) -> dict[str, dict[str, float | None]]:
    """Network-family measures per cluster: density, clustering coefficient and hub score."""
    density_out = compute_network_density(knn_idx, cluster_mask, top_k_map)
    cls_coef_out = compute_cls_coef(cluster_mask, knn_idx)
    hub_out = compute_hub(cluster_mask, knn_idx)

    result: dict[str, dict[str, float | None]] = {}
    for cid, row in density_out.items():
        result[cid] = {**row, "cls_coef": cls_coef_out[cid], "hub": hub_out[cid]}
    return result
