import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from src.core.utils import timed
from src.domain.analysis.complexity.shared import scale_for_metric


def _t3_t4(rows: np.ndarray, n_rows: int) -> tuple[float | None, float | None]:
    """T3 and T4 from one PCA fit: (n_pca_95/n, n_pca_95/d), None when degenerate."""
    n, d = rows.shape
    if n < 2 or d < 1:
        return None, None
    if d == 1:
        return 1.0 / n_rows, 1.0
    max_components = min(n, d)
    cumvar = np.cumsum(
        PCA(n_components=max_components).fit(rows).explained_variance_ratio_
    )
    n_pca_95 = min(int(np.searchsorted(cumvar, 0.95)) + 1, max_components)
    return n_pca_95 / n_rows, n_pca_95 / d


@timed
def compute_t_measures(
    X: np.ndarray, population: np.ndarray, sizes: dict[int, int], *, metric: str
) -> dict[int, dict[str, float | None]]:
    """Per-population dimensionality measures T2, T3 and T4."""
    # The components come from the rows of `X`, a sample of each population; the ratios
    # are over `sizes`, the rows the population really holds.
    X = scale_for_metric(X, metric)
    result: dict[int, dict[str, float | None]] = {}
    d = X.shape[1]
    for pid in tqdm(np.unique(population), desc="T measures", unit="pop", leave=False):
        n_rows = sizes[int(pid)]
        t3_val, t4_val = _t3_t4(X[population == pid], n_rows)
        result[int(pid)] = {"t2": d / n_rows, "t3": t3_val, "t4": t4_val}
    return result
