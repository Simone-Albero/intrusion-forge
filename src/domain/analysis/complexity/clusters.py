import numpy as np
from sklearn.metrics import pairwise_distances, silhouette_samples

from src.core.utils import timed


def _approx_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    metric: str,
    max_samples: int,
    min_per_cluster: int,
    random_state: int,
) -> np.ndarray | None:
    """Silhouette scores on a stratified subsample, NaN elsewhere, None below two labels."""
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return None

    n = len(X)
    if n <= max_samples:
        idx = np.arange(n)
    else:
        rng = np.random.default_rng(random_state)
        # The floor holds even past the cap: a silhouette tail read off a handful of
        # rows per cluster is noise, so many clusters grow the subsample instead.
        idx_parts: list[np.ndarray] = []
        for lbl in unique_labels:
            members = np.where(labels == lbl)[0]
            take = min(len(members), min_per_cluster)
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
    samples: np.ndarray, centroid: np.ndarray, *, metric: str
) -> tuple[float, float]:
    """Max and 95th-percentile distance of a cluster's samples from its centroid."""
    dists = pairwise_distances(samples, centroid.reshape(1, -1), metric=metric).ravel()
    return float(np.max(dists)), float(np.percentile(dists, 95))


def _nearest_rival(
    pw_row: np.ndarray, own_class: int, classes: np.ndarray
) -> float | None:
    """Distance to the closest centroid of a different class."""
    rival = pw_row[classes != own_class]
    finite = rival[np.isfinite(rival)]
    if finite.size == 0:
        return None
    return float(np.min(finite))


@timed
def compute_cluster_geometry(
    X: np.ndarray,
    population: np.ndarray,
    centroids: dict[int, np.ndarray],
    *,
    metric: str,
    silhouette_max_samples: int,
    silhouette_min_per_cluster: int,
    random_state: int,
    population_class: dict[int, int],
) -> dict[int, dict[str, float | None]]:
    """Per-population geometry: dispersion, rival separation and silhouette tail."""
    pids = [int(pid) for pid in np.unique(population)]
    centroid_matrix = np.stack(
        [np.asarray(centroids[pid], dtype=np.float64) for pid in pids]
    )
    classes = np.array([population_class[pid] for pid in pids])

    pw = pairwise_distances(centroid_matrix, metric=metric)
    np.fill_diagonal(pw, np.inf)

    sil = _approx_silhouette(
        X,
        population,
        metric=metric,
        max_samples=silhouette_max_samples,
        min_per_cluster=silhouette_min_per_cluster,
        random_state=random_state,
    )

    result: dict[int, dict[str, float | None]] = {}

    for i, pid in enumerate(pids):
        in_population = population == pid
        max_disp, p95_disp = _dispersion(
            X[in_population], centroid_matrix[i], metric=metric
        )
        dist_rival = _nearest_rival(pw[i], classes[i], classes)

        if sil is None:
            p5_sil, frac_at_risk = None, None
        else:
            sil_c = sil[in_population]
            sil_finite = sil_c[np.isfinite(sil_c)]
            if len(sil_finite) == 0:
                p5_sil, frac_at_risk = None, None
            else:
                p5_sil = float(np.percentile(sil_finite, 5))
                frac_at_risk = float(np.mean(sil_finite < 0))

        result[pid] = {
            "max_dispersion": max_disp,
            "p95_dispersion": p95_disp,
            "dist_to_nearest_rival": dist_rival,
            "p5_silhouette": p5_sil,
            "frac_at_risk": frac_at_risk,
        }

    return result
