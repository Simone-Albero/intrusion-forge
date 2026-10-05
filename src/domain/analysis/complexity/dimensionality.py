import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from src.core.utils import timed
from src.domain.analysis.complexity.shared import scale_for_metric


def _t3_t4(rows: np.ndarray, n_rows: int) -> tuple[float | None, float | None]:
    """T3 and T4 from one PCA fit, None when degenerate."""
    n_sampled, n_features = rows.shape
    if n_sampled < 2 or n_features < 1:
        return None, None
    if n_features == 1:
        return 1.0 / n_rows, 1.0
    max_components = min(n_sampled, n_features)
    cumulative_variance = np.cumsum(
        PCA(n_components=max_components).fit(rows).explained_variance_ratio_
    )
    n_components_95 = min(
        int(np.searchsorted(cumulative_variance, 0.95)) + 1, max_components
    )
    return n_components_95 / n_rows, n_components_95 / n_features


@timed
def compute_t_measures(
    X: np.ndarray, population: np.ndarray, sizes: dict[int, int], *, metric: str
) -> dict[int, dict[str, float | None]]:
    """Per-population dimensionality measures T2, T3 and T4."""
    # The components come from the rows of `X`, a sample of each population; the ratios
    # are over `sizes`, the rows the population really holds.
    X = scale_for_metric(X, metric)
    result: dict[int, dict[str, float | None]] = {}
    n_features = X.shape[1]
    for population_id in tqdm(
        np.unique(population), desc="T measures", unit="pop", leave=False
    ):
        n_rows = sizes[int(population_id)]
        t3, t4 = _t3_t4(X[population == population_id], n_rows)
        result[int(population_id)] = {"t2": n_features / n_rows, "t3": t3, "t4": t4}
    return result
