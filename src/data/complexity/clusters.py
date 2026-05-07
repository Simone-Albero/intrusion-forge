import numpy as np
from sklearn.metrics import pairwise_distances, silhouette_samples

from ...common.utils import timed


def _approx_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    max_samples: int = 10_000,
    min_per_cluster: int = 50,
) -> np.ndarray | None:
    """Approximate silhouette scores (Euclidean).

    Stratified sample of up to max_samples points ensuring min_per_cluster per
    cluster. Non-sampled points receive NaN. Returns None if < 2 unique labels.
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
        scores = silhouette_samples(X[idx], labels[idx])
    except ValueError:
        return None

    full = np.full(n, np.nan)
    full[idx] = scores
    return full


@timed
def compute_cluster_geometry(
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    y_class: np.ndarray,
    y_cluster: np.ndarray,
    centroids: dict[str, list[float]],
) -> dict[str, dict[str, float | None]]:
    """Compute 5 geometry measures per cluster.

    Inputs:
        X_num      — (n, d_num) float array, RobustScaled numericals.
        y_class    — (n,) int array, class labels.
        y_cluster  — (n,) int array, cluster labels (-1 = noise, excluded).
        centroids  — {str(cluster_id): [float, ...]} numerical centroids.

    Output keys per cluster:
        max_dispersion                  max Euclidean distance sample → centroid
        dist_to_nearest_foreign_cluster min centroid-to-centroid dist (cross-class)
        p5_silhouette                   5th pct of Euclidean silhouette scores
        frac_at_risk                    fraction of points with silhouette < 0
        min_sibling_centroid_dist       min centroid-to-centroid dist (same class)

    Noise points (y_cluster == -1) are excluded from all computations.
    Silhouette is computed on non-noise points with cluster labels as groups.
    """
    # --- filter noise ---
    mask_valid = y_cluster != -1
    X_v = X_num[mask_valid]
    yc_v = y_class[mask_valid]
    yk_v = y_cluster[mask_valid]

    # --- centroid matrix (only clusters present in data and centroids dict) ---
    present_ids = [str(cid) for cid in np.unique(yk_v) if str(cid) in centroids]
    if not present_ids:
        return {}

    centroid_matrix = np.stack(
        [np.asarray(centroids[cid], dtype=np.float64) for cid in present_ids]
    )  # shape (C, d)
    id_to_idx = {cid: i for i, cid in enumerate(present_ids)}

    # class ownership per centroid
    id_to_class: dict[str, int] = {}
    for cid in present_ids:
        cid_int = int(cid)
        mask_cid = yk_v == cid_int
        if mask_cid.any():
            id_to_class[cid] = int(yc_v[mask_cid][0])

    # pairwise centroid distances (C x C)
    pw = pairwise_distances(centroid_matrix, metric="euclidean")  # (C, C)
    np.fill_diagonal(pw, np.inf)

    # --- silhouette on non-noise points using cluster labels ---
    sil_values = _approx_silhouette(X_v, yk_v)

    result: dict[str, dict[str, float | None]] = {}

    for cid in present_ids:
        cid_int = int(cid)
        idx_c = id_to_idx[cid]
        cls_c = id_to_class.get(cid)

        mask_cid = yk_v == cid_int
        samples = X_v[mask_cid]
        centroid = centroid_matrix[idx_c]

        # max_dispersion
        dists = pairwise_distances(
            samples, centroid.reshape(1, -1), metric="euclidean"
        ).ravel()
        max_dispersion = float(np.max(dists)) if len(dists) > 0 else None

        # dist_to_nearest_foreign_cluster
        row = pw[idx_c]
        foreign_mask = np.array(
            [id_to_class.get(cid2) != cls_c for cid2 in present_ids], dtype=bool
        )
        foreign_dists = row[foreign_mask]
        dist_to_nearest_foreign = (
            float(np.min(foreign_dists))
            if foreign_mask.any() and np.isfinite(foreign_dists).any()
            else None
        )

        # min_sibling_centroid_dist (same class, different cluster)
        sibling_mask = np.array(
            [id_to_class.get(cid2) == cls_c and cid2 != cid for cid2 in present_ids],
            dtype=bool,
        )
        sibling_dists = row[sibling_mask]
        min_sibling = (
            float(np.min(sibling_dists))
            if sibling_mask.any() and np.isfinite(sibling_dists).any()
            else None
        )

        # p5_silhouette + frac_at_risk
        if sil_values is not None:
            sil_c = sil_values[mask_cid]
            sil_finite = sil_c[np.isfinite(sil_c)]
            p5_sil = (
                float(np.percentile(sil_finite, 5)) if len(sil_finite) > 0 else None
            )
            frac_at_risk = (
                float(np.mean(sil_finite < 0)) if len(sil_finite) > 0 else None
            )
        else:
            p5_sil = None
            frac_at_risk = None

        result[cid] = {
            "max_dispersion": max_dispersion,
            "dist_to_nearest_foreign_cluster": dist_to_nearest_foreign,
            "p5_silhouette": p5_sil,
            "frac_at_risk": frac_at_risk,
            "min_sibling_centroid_dist": min_sibling,
        }

    return result
