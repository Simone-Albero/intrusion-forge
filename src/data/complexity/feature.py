import numpy as np
from tqdm import tqdm

from src.data.complexity.shared import aggregate_min_mean_max, make_null_row
from ...common.utils import timed


def _f1_pair(X_c: np.ndarray, X_j: np.ndarray, eps: float = 1e-8) -> float:
    """F1: 1/(1 + max_feature Fisher ratio). Higher = harder (more overlap)."""
    mu_c = X_c.mean(axis=0)
    mu_j = X_j.mean(axis=0)
    var_c = X_c.var(axis=0)
    var_j = X_j.var(axis=0)
    fisher = (mu_c - mu_j) ** 2 / (var_c + var_j + eps)
    return float(1.0 / (1.0 + np.max(fisher)))


def _f2_pair(X_c: np.ndarray, X_j: np.ndarray, eps: float = 1e-8) -> float:
    """F2: mean over features of per-feature bounding-box overlap ratio. Higher = harder."""
    min_c, max_c = X_c.min(axis=0), X_c.max(axis=0)
    min_j, max_j = X_j.min(axis=0), X_j.max(axis=0)
    overlap = np.maximum(0.0, np.minimum(max_c, max_j) - np.maximum(min_c, min_j))
    total_range = np.maximum(max_c, max_j) - np.minimum(min_c, min_j) + eps
    return float(np.mean(overlap / total_range))


def _f3_pair(X_c: np.ndarray, X_j: np.ndarray) -> float:
    """F3: min over features of the fraction of cluster-c samples in the overlap region.

    Measures how well a single feature can separate cluster c from class j.
    Higher = harder (no single feature separates well).
    """
    min_c, max_c = X_c.min(axis=0), X_c.max(axis=0)
    min_j, max_j = X_j.min(axis=0), X_j.max(axis=0)
    lo = np.maximum(min_c, min_j)
    hi = np.minimum(max_c, max_j)
    fracs = []
    for f in range(X_c.shape[1]):
        if hi[f] >= lo[f]:
            frac = float(np.mean((X_c[:, f] >= lo[f]) & (X_c[:, f] <= hi[f])))
        else:
            frac = 0.0
        fracs.append(frac)
    return float(np.min(fracs))


def _f4_pair(X_c: np.ndarray, X_j: np.ndarray) -> float:
    """F4: fraction of cluster-c samples in the overlap region on ALL features simultaneously.

    Higher = harder (many samples overlap jointly in all features).
    """
    min_c, max_c = X_c.min(axis=0), X_c.max(axis=0)
    min_j, max_j = X_j.min(axis=0), X_j.max(axis=0)
    lo = np.maximum(min_c, min_j)
    hi = np.minimum(max_c, max_j)
    # if any feature has no overlap, no sample can be in the joint region
    if np.any(hi < lo):
        return 0.0
    in_all = np.ones(len(X_c), dtype=bool)
    for f in range(X_c.shape[1]):
        in_all &= (X_c[:, f] >= lo[f]) & (X_c[:, f] <= hi[f])
    return float(np.mean(in_all))


_F_KEYS = ("f1", "f2", "f3", "f4")


def _pair_block(X_c: np.ndarray, X_others: list[np.ndarray]) -> dict[str, list[float]]:
    """Compute F1-F4 for cluster X_c against each population in X_others.

    Skips populations with fewer than 2 samples.
    """
    out: dict[str, list[float]] = {k: [] for k in _F_KEYS}
    for X_o in X_others:
        if len(X_o) < 2:
            continue
        out["f1"].append(_f1_pair(X_c, X_o))
        out["f2"].append(_f2_pair(X_c, X_o))
        out["f3"].append(_f3_pair(X_c, X_o))
        out["f4"].append(_f4_pair(X_c, X_o))
    return out


@timed
def compute_f_measures(
    X_num: np.ndarray,
    y_class: np.ndarray,
    y_cluster: np.ndarray,
    cluster_to_class: dict[str, int],
    top_k_map: dict[str, list[str]],
) -> dict[str, dict[str, float | None]]:
    """Compute F1-F4 per cluster aggregated against (a) adversarial classes and
    (b) the top-K nearest adversarial clusters, returned as min/mean/max for
    each scope.

    Inputs:
        X_num             — (n, d_num) float array, RobustScaled numericals.
        y_class           — (n,) int array, class labels.
        y_cluster         — (n,) int array, cluster labels (-1 = noise, excluded).
        cluster_to_class  — {str(cluster_id): class_label} for non-noise clusters.
        top_k_map         — {str(cluster_id): [str(adversarial_cluster_id), ...]}
                            top-K nearest adversarial clusters per cluster.

    Output keys per cluster (24 total):
        f{i}_class_{min,mean,max}    — aggregated over adversarial classes
        f{i}_cluster_{min,mean,max}  — aggregated over top-K adversarial clusters
        for i in {1, 2, 3, 4}.

    F-family is numerical-only by design (categorical information enters via
    the N/ND families through the Gower k-NN graph).
    """
    mask_valid = y_cluster != -1
    X_v = X_num[mask_valid]
    yc_v = y_class[mask_valid]
    yk_v = y_cluster[mask_valid]

    class_block: dict[int, np.ndarray] = {
        int(j): X_v[yc_v == j] for j in np.unique(yc_v)
    }
    cluster_block: dict[str, np.ndarray] = {
        str(int(cid)): X_v[yk_v == cid] for cid in np.unique(yk_v)
    }

    result: dict[str, dict[str, float | None]] = {}
    for cid_str, X_c in tqdm(
        cluster_block.items(), desc="F measures", unit="cluster", leave=False
    ):
        row = make_null_row(_F_KEYS)
        if len(X_c) < 2 or X_c.shape[1] == 0:
            result[cid_str] = row
            continue

        cls_c = cluster_to_class[cid_str]
        class_blocks = [b for j, b in class_block.items() if j != cls_c]
        cluster_blocks = [
            cluster_block[ac]
            for ac in top_k_map.get(cid_str, [])
            if ac in cluster_block
        ]

        for scope, blocks in (("class", class_blocks), ("cluster", cluster_blocks)):
            vals = _pair_block(X_c, blocks)
            for fk in _F_KEYS:
                mn, me, mx = aggregate_min_mean_max(vals[fk])
                row[f"{fk}_{scope}_min"] = mn
                row[f"{fk}_{scope}_mean"] = me
                row[f"{fk}_{scope}_max"] = mx

        result[cid_str] = row

    return result
