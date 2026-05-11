import numpy as np
from sklearn.metrics import pairwise_distances, silhouette_samples

from ...common.utils import timed


def _approx_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
    max_samples: int = 10_000,
    min_per_cluster: int = 50,
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
        rng = np.random.default_rng(42)
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


@timed
def compute_cluster_geometry(
    X_num: np.ndarray,
    y_class: np.ndarray,
    y_cluster: np.ndarray,
    centroids: dict[str, list[float]],
) -> dict[str, dict[str, float | None]]:
    """Compute geometry measures per cluster.

    Inputs:
        X_num      — (n, d_num) float array, RobustScaled numericals.
        y_class    — (n,) int array, class labels.
        y_cluster  — (n,) int array, cluster labels (-1 = noise, excluded).
        centroids  — {str(cluster_id): [float, ...]} numerical centroids.

    Output keys per cluster (Euclidean + cosine variants always computed):
        max_dispersion / max_dispersion_cosine
        p95_dispersion / p95_dispersion_cosine
        dist_to_nearest_foreign_cluster / dist_to_nearest_foreign_cluster_cosine
        p5_silhouette / p5_silhouette_cosine
        frac_at_risk
        min_sibling_centroid_dist / min_sibling_centroid_dist_cosine

    Noise points (y_cluster == -1) are excluded from all computations.
    """
    mask_valid = y_cluster != -1
    X_v = X_num[mask_valid]
    yc_v = y_class[mask_valid]
    yk_v = y_cluster[mask_valid]

    present_ids = [str(cid) for cid in np.unique(yk_v) if str(cid) in centroids]
    if not present_ids:
        return {}

    centroid_matrix = np.stack(
        [np.asarray(centroids[cid], dtype=np.float64) for cid in present_ids]
    )
    id_to_idx = {cid: i for i, cid in enumerate(present_ids)}

    id_to_class: dict[str, int] = {}
    for cid in present_ids:
        mask_cid = yk_v == int(cid)
        if mask_cid.any():
            id_to_class[cid] = int(yc_v[mask_cid][0])

    # pairwise centroid distances — Euclidean and cosine
    pw_euc = pairwise_distances(centroid_matrix, metric="euclidean")
    pw_cos = pairwise_distances(centroid_matrix, metric="cosine")
    np.fill_diagonal(pw_euc, np.inf)
    np.fill_diagonal(pw_cos, np.inf)

    # silhouette on non-noise points (both metrics)
    sil_euc = _approx_silhouette(X_v, yk_v, metric="euclidean")
    sil_cos = _approx_silhouette(X_v, yk_v, metric="cosine")

    result: dict[str, dict[str, float | None]] = {}

    for cid in present_ids:
        idx_c = id_to_idx[cid]
        cls_c = id_to_class.get(cid)

        mask_cid = yk_v == int(cid)
        samples = X_v[mask_cid]
        centroid = centroid_matrix[idx_c]

        max_disp_e, p95_disp_e = _dispersion(samples, centroid, "euclidean")
        max_disp_c, p95_disp_c = _dispersion(samples, centroid, "cosine")

        dist_foreign_e = _nearest_foreign(pw_euc[idx_c], present_ids, id_to_class, cls_c)
        dist_foreign_c = _nearest_foreign(pw_cos[idx_c], present_ids, id_to_class, cls_c)

        min_sib_e = _nearest_sibling(pw_euc[idx_c], present_ids, id_to_class, cls_c, cid)
        min_sib_c = _nearest_sibling(pw_cos[idx_c], present_ids, id_to_class, cls_c, cid)

        def _p5_frac(sil_values: np.ndarray | None) -> tuple[float | None, float | None]:
            if sil_values is None:
                return None, None
            sil_c = sil_values[mask_cid]
            sil_finite = sil_c[np.isfinite(sil_c)]
            if len(sil_finite) == 0:
                return None, None
            return float(np.percentile(sil_finite, 5)), float(np.mean(sil_finite < 0))

        p5_sil_e, frac_at_risk = _p5_frac(sil_euc)
        p5_sil_c, _ = _p5_frac(sil_cos)

        result[cid] = {
            "max_dispersion": max_disp_e,
            "p95_dispersion": p95_disp_e,
            "dist_to_nearest_foreign_cluster": dist_foreign_e,
            "p5_silhouette": p5_sil_e,
            "frac_at_risk": frac_at_risk,
            "min_sibling_centroid_dist": min_sib_e,
            "max_dispersion_cosine": max_disp_c,
            "p95_dispersion_cosine": p95_disp_c,
            "dist_to_nearest_foreign_cluster_cosine": dist_foreign_c,
            "p5_silhouette_cosine": p5_sil_c,
            "min_sibling_centroid_dist_cosine": min_sib_c,
        }

    return result
