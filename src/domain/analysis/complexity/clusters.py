import numpy as np
from sklearn.metrics import pairwise_distances, silhouette_samples

from src.core.utils import timed


def _approx_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
    max_samples: int = 10_000,
    min_per_cluster: int = 50,
    random_state: int = 42,
) -> np.ndarray | None:
    """Approximate silhouette scores with stratified subsampling.

    Non-sampled points receive NaN. Returns None if < 2 unique labels.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return None

    n = len(X)
    if n <= max_samples:
        idx = np.arange(n)
    else:
        rng = np.random.default_rng(random_state)
        idx_parts: list[np.ndarray] = []
        for lbl in unique_labels:
            members = np.where(labels == lbl)[0]
            take = min(len(members), max(min_per_cluster, 1))
            idx_parts.append(rng.choice(members, size=take, replace=False))
        guaranteed = np.concatenate(idx_parts)
        remaining = max_samples - len(guaranteed)
        if remaining > 0:
            pool = np.setdiff1d(np.arange(n), guaranteed)
            extra = rng.choice(pool, size=min(remaining, len(pool)), replace=False)
            idx = np.concatenate([guaranteed, extra])
        else:
            idx = guaranteed

    try:
        scores = silhouette_samples(X[idx], labels[idx], metric=metric)
    except ValueError:
        return None

    full = np.full(n, np.nan)
    full[idx] = scores
    return full


def _dispersion(
    samples: np.ndarray, centroid: np.ndarray, metric: str
) -> tuple[float | None, float | None]:
    """(max_dispersion, p95_dispersion) for given metric."""
    if len(samples) == 0:
        return None, None
    dists = pairwise_distances(samples, centroid.reshape(1, -1), metric=metric).ravel()
    return float(np.max(dists)), float(np.percentile(dists, 95))


def _nearest_foreign(
    pw_row: np.ndarray, present_ids: list[str], id_to_class: dict[str, int], cls_c: int | None
) -> float | None:
    """Min centroid distance to a cluster of a different class."""
    foreign_mask = np.array(
        [id_to_class.get(c) != cls_c for c in present_ids], dtype=bool
    )
    foreign_dists = pw_row[foreign_mask]
    if foreign_mask.any() and np.isfinite(foreign_dists).any():
        return float(np.min(foreign_dists))
    return None


def _nearest_sibling(
    pw_row: np.ndarray, present_ids: list[str], id_to_class: dict[str, int], cls_c: int | None, cid: str
) -> float | None:
    """Min centroid distance to a cluster of the same class."""
    sibling_mask = np.array(
        [id_to_class.get(c) == cls_c and c != cid for c in present_ids], dtype=bool
    )
    sibling_dists = pw_row[sibling_mask]
    if sibling_mask.any() and np.isfinite(sibling_dists).any():
        return float(np.min(sibling_dists))
    return None


def _spherical_centroid(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Fréchet mean under cosine distance: mean of L2-normalised samples, re-normalised."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / np.maximum(norms, eps)
    sph = X_norm.mean(axis=0)
    sph_norm = np.linalg.norm(sph)
    return sph / max(sph_norm, eps)


@timed
def compute_cluster_geometry(
    X_num: np.ndarray,
    y_class: np.ndarray,
    y_cluster: np.ndarray,
    centroids: dict[str, list[float]],
    *,
    metric: str = "cosine",
    random_state: int = 42,
) -> dict[str, dict[str, float | None]]:
    """Compute geometry measures per cluster using a single configured metric.

    Inputs:
        X_num       — (n, d_num) float array, RobustScaled numericals.
        y_class     — (n,) int array, class labels.
        y_cluster   — (n,) int array, cluster labels (-1 = noise, excluded).
        centroids   — {str(cluster_id): [float, ...]} Euclidean centroids.
        metric      — "cosine" or "euclidean". Controls centroid type, pairwise
                      distances, and silhouette computation.
        random_state — seed for silhouette subsampling.

    Output keys per cluster (neutral, no metric suffix):
        max_dispersion, p95_dispersion
        dist_to_nearest_foreign_cluster, min_sibling_centroid_dist
        p5_silhouette, frac_at_risk          (cluster-label silhouette)
        p5_silhouette_class, frac_at_risk_class  (class-label silhouette)

    Noise points (y_cluster == -1) are excluded from all computations.
    """
    mask_valid = y_cluster != -1
    X_v = X_num[mask_valid]
    yc_v = y_class[mask_valid]
    yk_v = y_cluster[mask_valid]

    present_ids = [str(cid) for cid in np.unique(yk_v) if str(cid) in centroids]
    if not present_ids:
        return {}

    id_to_class: dict[str, int] = {}
    for cid in present_ids:
        mask_cid = yk_v == int(cid)
        if mask_cid.any():
            id_to_class[cid] = int(yc_v[mask_cid][0])

    # build centroid matrix appropriate for the metric
    if metric == "cosine":
        centroid_matrix = np.stack(
            [_spherical_centroid(X_v[yk_v == int(cid)]) for cid in present_ids]
        )
    else:
        centroid_matrix = np.stack(
            [np.asarray(centroids[cid], dtype=np.float64) for cid in present_ids]
        )

    id_to_idx = {cid: i for i, cid in enumerate(present_ids)}

    pw = pairwise_distances(centroid_matrix, metric=metric)
    np.fill_diagonal(pw, np.inf)

    sil_cluster = _approx_silhouette(X_v, yk_v, metric=metric, random_state=random_state)
    sil_class = _approx_silhouette(X_v, yc_v, metric=metric, random_state=random_state)

    result: dict[str, dict[str, float | None]] = {}

    for cid in present_ids:
        idx_c = id_to_idx[cid]
        cls_c = id_to_class.get(cid)
        mask_cid = yk_v == int(cid)
        samples = X_v[mask_cid]
        centroid = centroid_matrix[idx_c]

        max_disp, p95_disp = _dispersion(samples, centroid, metric)
        dist_foreign = _nearest_foreign(pw[idx_c], present_ids, id_to_class, cls_c)
        min_sib = _nearest_sibling(pw[idx_c], present_ids, id_to_class, cls_c, cid)

        def _p5_frac(sil_values: np.ndarray | None) -> tuple[float | None, float | None]:
            if sil_values is None:
                return None, None
            sil_c = sil_values[mask_cid]
            sil_finite = sil_c[np.isfinite(sil_c)]
            if len(sil_finite) == 0:
                return None, None
            return float(np.percentile(sil_finite, 5)), float(np.mean(sil_finite < 0))

        p5_sil, frac_at_risk = _p5_frac(sil_cluster)
        p5_sil_class, frac_at_risk_class = _p5_frac(sil_class)

        result[cid] = {
            "max_dispersion": max_disp,
            "p95_dispersion": p95_disp,
            "dist_to_nearest_foreign_cluster": dist_foreign,
            "min_sibling_centroid_dist": min_sib,
            "p5_silhouette": p5_sil,
            "frac_at_risk": frac_at_risk,
            "p5_silhouette_class": p5_sil_class,
            "frac_at_risk_class": frac_at_risk_class,
        }

    return result
