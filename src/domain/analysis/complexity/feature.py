import numpy as np
from tqdm import tqdm

from src.core.utils import timed
from src.domain.analysis.complexity.shared import (
    aggregate_min_mean_max,
    make_null_row,
    scale_for_metric,
)


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
    """F3: min over features of the cluster-c fraction inside the overlap. Higher = harder."""
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
    """F4: fraction of cluster-c samples inside the overlap on every feature. Higher = harder."""
    min_c, max_c = X_c.min(axis=0), X_c.max(axis=0)
    min_j, max_j = X_j.min(axis=0), X_j.max(axis=0)
    lo = np.maximum(min_c, min_j)
    hi = np.minimum(max_c, max_j)
    if np.any(hi < lo):
        return 0.0
    in_all = np.ones(len(X_c), dtype=bool)
    for f in range(X_c.shape[1]):
        in_all &= (X_c[:, f] >= lo[f]) & (X_c[:, f] <= hi[f])
    return float(np.mean(in_all))


_F_KEYS = ("f1", "f2", "f3", "f4")


def _pair_block(X_c: np.ndarray, X_rivals: list[np.ndarray]) -> dict[str, list[float]]:
    """F1-F4 of population c against each rival."""
    out: dict[str, list[float]] = {k: [] for k in _F_KEYS}
    for X_o in X_rivals:
        if len(X_o) < 2:
            continue
        out["f1"].append(_f1_pair(X_c, X_o))
        out["f2"].append(_f2_pair(X_c, X_o))
        out["f3"].append(_f3_pair(X_c, X_o))
        out["f4"].append(_f4_pair(X_c, X_o))
    return out


@timed
def compute_f_measures(
    X: np.ndarray,
    population: np.ndarray,
    rivals: dict[int, list[int]],
    *,
    metric: str,
) -> dict[int, dict[str, float | None]]:
    """F1-F4 per population against its nearest rivals, as min/mean/max."""
    X_v = scale_for_metric(X, metric)
    rows_of: dict[int, np.ndarray] = {
        int(pid): X_v[population == pid] for pid in np.unique(population)
    }

    result: dict[int, dict[str, float | None]] = {}
    for pid, X_c in tqdm(rows_of.items(), desc="F measures", unit="pop", leave=False):
        row = make_null_row(_F_KEYS)
        if len(X_c) < 2 or X_c.shape[1] == 0:
            result[pid] = row
            continue

        vals = _pair_block(X_c, [rows_of[rival] for rival in rivals[pid]])
        for fk in _F_KEYS:
            mn, me, mx = aggregate_min_mean_max(vals[fk])
            row[f"{fk}_min"] = mn
            row[f"{fk}_mean"] = me
            row[f"{fk}_max"] = mx

        result[pid] = row

    return result
