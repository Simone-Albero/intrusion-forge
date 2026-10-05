import numpy as np
from tqdm import tqdm

from src.core.utils import timed
from src.domain.analysis.complexity.shared import (
    aggregate_min_mean_max,
    make_null_row,
    scale_for_metric,
)


def _f1_pair(X_own: np.ndarray, X_rival: np.ndarray, eps: float = 1e-8) -> float:
    """F1: 1/(1 + max_feature Fisher ratio). Higher = harder (more overlap)."""
    mean_own = X_own.mean(axis=0)
    mean_rival = X_rival.mean(axis=0)
    var_own = X_own.var(axis=0)
    var_rival = X_rival.var(axis=0)
    fisher = (mean_own - mean_rival) ** 2 / (var_own + var_rival + eps)
    return float(1.0 / (1.0 + np.max(fisher)))


def _f2_pair(X_own: np.ndarray, X_rival: np.ndarray, eps: float = 1e-8) -> float:
    """F2: mean over features of per-feature bounding-box overlap ratio. Higher = harder."""
    min_own, max_own = X_own.min(axis=0), X_own.max(axis=0)
    min_rival, max_rival = X_rival.min(axis=0), X_rival.max(axis=0)
    overlap = np.maximum(
        0.0, np.minimum(max_own, max_rival) - np.maximum(min_own, min_rival)
    )
    total_range = np.maximum(max_own, max_rival) - np.minimum(min_own, min_rival) + eps
    return float(np.mean(overlap / total_range))


def _f3_pair(X_own: np.ndarray, X_rival: np.ndarray) -> float:
    """F3: min over features of the own share inside the overlap. Higher = harder."""
    min_own, max_own = X_own.min(axis=0), X_own.max(axis=0)
    min_rival, max_rival = X_rival.min(axis=0), X_rival.max(axis=0)
    lo = np.maximum(min_own, min_rival)
    hi = np.minimum(max_own, max_rival)
    fractions = []
    for feature in range(X_own.shape[1]):
        if hi[feature] >= lo[feature]:
            fraction = float(
                np.mean(
                    (X_own[:, feature] >= lo[feature])
                    & (X_own[:, feature] <= hi[feature])
                )
            )
        else:
            fraction = 0.0
        fractions.append(fraction)
    return float(np.min(fractions))


def _f4_pair(X_own: np.ndarray, X_rival: np.ndarray) -> float:
    """F4: own share inside the overlap on every feature. Higher = harder."""
    min_own, max_own = X_own.min(axis=0), X_own.max(axis=0)
    min_rival, max_rival = X_rival.min(axis=0), X_rival.max(axis=0)
    lo = np.maximum(min_own, min_rival)
    hi = np.minimum(max_own, max_rival)
    if np.any(hi < lo):
        return 0.0
    in_all = np.ones(len(X_own), dtype=bool)
    for feature in range(X_own.shape[1]):
        in_all &= (X_own[:, feature] >= lo[feature]) & (
            X_own[:, feature] <= hi[feature]
        )
    return float(np.mean(in_all))


_F_KEYS = ("f1", "f2", "f3", "f4")


def _pair_measures(
    X_own: np.ndarray, X_rivals: list[np.ndarray]
) -> dict[str, list[float]]:
    """F1-F4 of a population against each of its rivals."""
    values_by_key: dict[str, list[float]] = {k: [] for k in _F_KEYS}
    for X_rival in X_rivals:
        if len(X_rival) < 2:
            continue
        values_by_key["f1"].append(_f1_pair(X_own, X_rival))
        values_by_key["f2"].append(_f2_pair(X_own, X_rival))
        values_by_key["f3"].append(_f3_pair(X_own, X_rival))
        values_by_key["f4"].append(_f4_pair(X_own, X_rival))
    return values_by_key


@timed
def compute_f_measures(
    X: np.ndarray,
    population: np.ndarray,
    rivals: dict[int, list[int]],
    *,
    metric: str,
) -> dict[int, dict[str, float | None]]:
    """F1-F4 per population against its nearest rivals, as min/mean/max."""
    X_scaled = scale_for_metric(X, metric)
    rows_by_population: dict[int, np.ndarray] = {
        int(population_id): X_scaled[population == population_id]
        for population_id in np.unique(population)
    }

    result: dict[int, dict[str, float | None]] = {}
    for population_id, X_own in tqdm(
        rows_by_population.items(), desc="F measures", unit="pop", leave=False
    ):
        row = make_null_row(_F_KEYS)
        if len(X_own) < 2 or X_own.shape[1] == 0:
            result[population_id] = row
            continue

        pair_values = _pair_measures(
            X_own, [rows_by_population[rival] for rival in rivals[population_id]]
        )
        for key in _F_KEYS:
            minimum, mean, maximum = aggregate_min_mean_max(pair_values[key])
            row[f"{key}_min"] = minimum
            row[f"{key}_mean"] = mean
            row[f"{key}_max"] = maximum

        result[population_id] = row

    return result
