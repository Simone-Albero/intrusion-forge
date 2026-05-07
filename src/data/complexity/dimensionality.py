import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from ...common.utils import timed


def t2(n: int, d_num: int, d_cat: int) -> float:
    """Feature-to-sample ratio: (d_num + d_cat) / n."""
    return (d_num + d_cat) / n if n > 0 else 0.0


def t3(X_num: np.ndarray) -> float:
    """PCA intrinsic dimensionality ratio: n_pca_95 / d_num."""
    n, d_num = X_num.shape
    if d_num < 2 or n < 2:
        return 1.0
    max_components = min(n, d_num)
    pca = PCA(n_components=max_components)
    cumvar = np.cumsum(pca.fit(X_num).explained_variance_ratio_)
    n_pca_95 = min(int(np.searchsorted(cumvar, 0.95)) + 1, max_components)
    return n_pca_95 / d_num


def t4(X_num: np.ndarray) -> float:
    """PCA components-to-sample ratio: n_pca_95 / n."""
    n, d_num = X_num.shape
    if d_num < 2 or n < 2:
        return d_num / n if n > 0 else 0.0
    max_components = min(n, d_num)
    pca = PCA(n_components=max_components)
    cumvar = np.cumsum(pca.fit(X_num).explained_variance_ratio_)
    n_pca_95 = min(int(np.searchsorted(cumvar, 0.95)) + 1, max_components)
    return n_pca_95 / n


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
    for cid in tqdm(
        np.unique(y_cluster), desc="T measures", unit="cluster", leave=False
    ):
        mask = y_cluster == cid
        Xn = X_num[mask]
        Xc = X_cat[mask] if X_cat is not None else None
        n, d_num = Xn.shape
        d_cat = Xc.shape[1] if Xc is not None else 0
        result[str(cid)] = {
            "t2": t2(n, d_num, d_cat),
            "t3": t3(Xn),
            "t4": t4(Xn),
        }
    return result
