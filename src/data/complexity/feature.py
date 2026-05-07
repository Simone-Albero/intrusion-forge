import numpy as np
from tqdm import tqdm

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


@timed
def compute_f_measures(
    X_num: np.ndarray,
    y_class: np.ndarray,
    y_cluster: np.ndarray,
) -> dict[str, dict[str, float | None]]:
    """Compute F1-F4 per cluster vs each adversarial class, aggregated as min + mean.

    Inputs:
        X_num      — (n, d_num) float array, RobustScaled numericals.
        y_class    — (n,) int array, class labels.
        y_cluster  — (n,) int array, cluster labels (-1 = noise, excluded).

    Output keys per cluster:
        f1_min, f1_mean, f1_max  (Fisher discriminant ratio — higher = harder)
        f2_min, f2_mean, f2_max  (bounding-box overlap — higher = harder)
        f3_min, f3_mean, f3_max  (best single-feature separability — higher = harder)
        f4_min, f4_mean, f4_max  (joint-feature overlap — higher = harder)

    For each cluster c (class cls_c), pairs it against each class j ≠ cls_c.
    X_j = all non-noise samples with y_class == j (across all clusters of j).
    """
    mask_valid = y_cluster != -1
    X_v = X_num[mask_valid]
    yc_v = y_class[mask_valid]
    yk_v = y_cluster[mask_valid]

    all_classes = np.unique(yc_v)
    result: dict[str, dict[str, float | None]] = {}

    for cid in tqdm(np.unique(yk_v), desc="F measures", unit="cluster", leave=False):
        mask_c = yk_v == cid
        X_c = X_v[mask_c]
        cls_c = int(yc_v[mask_c][0])
        adversarial = [j for j in all_classes if j != cls_c]

        null_row: dict[str, float | None] = {
            "f1_min": None,
            "f1_mean": None,
            "f1_max": None,
            "f2_min": None,
            "f2_mean": None,
            "f2_max": None,
            "f3_min": None,
            "f3_mean": None,
            "f3_max": None,
            "f4_min": None,
            "f4_mean": None,
            "f4_max": None,
        }

        if not adversarial or len(X_c) < 2 or X_c.shape[1] == 0:
            result[str(cid)] = null_row
            continue

        f1_vals, f2_vals, f3_vals, f4_vals = [], [], [], []
        for j in adversarial:
            X_j = X_v[yc_v == j]
            if len(X_j) < 2:
                continue
            f1_vals.append(_f1_pair(X_c, X_j))
            f2_vals.append(_f2_pair(X_c, X_j))
            f3_vals.append(_f3_pair(X_c, X_j))
            f4_vals.append(_f4_pair(X_c, X_j))

        def _agg(vals: list[float]) -> tuple[float | None, float | None, float | None]:
            if not vals:
                return None, None, None
            return float(np.min(vals)), float(np.mean(vals)), float(np.max(vals))

        f1_min, f1_mean, f1_max = _agg(f1_vals)
        f2_min, f2_mean, f2_max = _agg(f2_vals)
        f3_min, f3_mean, f3_max = _agg(f3_vals)
        f4_min, f4_mean, f4_max = _agg(f4_vals)

        result[str(cid)] = {
            "f1_min": f1_min,
            "f1_mean": f1_mean,
            "f1_max": f1_max,
            "f2_min": f2_min,
            "f2_mean": f2_mean,
            "f2_max": f2_max,
            "f3_min": f3_min,
            "f3_mean": f3_mean,
            "f3_max": f3_max,
            "f4_min": f4_min,
            "f4_mean": f4_mean,
            "f4_max": f4_max,
        }

    return result
