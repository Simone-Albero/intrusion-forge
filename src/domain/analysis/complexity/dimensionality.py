import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from src.core.utils import timed


def t2(n: int, d_num: int, d_cat: int) -> float:
    """Feature-to-sample ratio: (d_num + d_cat) / n."""
    return (d_num + d_cat) / n if n > 0 else 0.0


def _t3_t4(X_num: np.ndarray) -> tuple[float, float]:
    """Compute T3 and T4 from a single PCA fit.

    Returns (n_pca_95 / d_num, n_pca_95 / n).
    """
    n, d_num = X_num.shape
    if d_num < 2 or n < 2:
        return 1.0, (d_num / n if n > 0 else 0.0)
    max_components = min(n, d_num)
    cumvar = np.cumsum(
        PCA(n_components=max_components).fit(X_num).explained_variance_ratio_
    )
    n_pca_95 = min(int(np.searchsorted(cumvar, 0.95)) + 1, max_components)
    return n_pca_95 / d_num, n_pca_95 / n


def t3(X_num: np.ndarray) -> float:
    """PCA intrinsic dimensionality ratio: n_pca_95 / d_num."""
    return _t3_t4(X_num)[0]


def t4(X_num: np.ndarray) -> float:
    """PCA components-to-sample ratio: n_pca_95 / n."""
    return _t3_t4(X_num)[1]


@timed
def compute_t_measures(
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    y_cluster: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute dimensionality complexity measures T2, T3, T4 per cluster.

    T2 = (d_num + d_cat) / n   — feature count relative to cluster size.
    T3 = n_pca_95 / d_num      — fraction of PCA components needed for 95% variance.
    T4 = n_pca_95 / n          — PCA components-to-cluster-size ratio.

    Inputs:
        X_num     — (n, d_num) float array, RobustScaled numericals.
        X_cat     — (n, d_cat) int array or None.
        y_cluster — (n,) int array, cluster labels.

    Returns {str(cluster_id): {t2: float, t3: float, t4: float}}.
    """
    result: dict[str, dict[str, float]] = {}
    cluster_ids = [int(cid) for cid in np.unique(y_cluster) if int(cid) != -1]
    d_cat = X_cat.shape[1] if X_cat is not None else 0
    for cid in tqdm(cluster_ids, desc="T measures", unit="cluster", leave=False):
        Xn = X_num[y_cluster == cid]
        n, d_num = Xn.shape
        t3_val, t4_val = _t3_t4(Xn)
        result[str(cid)] = {"t2": t2(n, d_num, d_cat), "t3": t3_val, "t4": t4_val}
    return result
