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
        rows = np.arange(n)
    else:
        rng = np.random.default_rng(random_state)
        # The floor holds even past the cap: a silhouette tail read off a handful of
        # rows per cluster is noise, so many clusters grow the subsample instead.
        picked_parts: list[np.ndarray] = []
        for label in unique_labels:
            members = np.where(labels == label)[0]
            take = min(len(members), min_per_cluster)
            picked_parts.append(rng.choice(members, size=take, replace=False))
        floor_rows = np.concatenate(picked_parts)
        remaining = max_samples - len(floor_rows)
        if remaining > 0:
            pool = np.setdiff1d(np.arange(n), floor_rows)
            extra = rng.choice(pool, size=min(remaining, len(pool)), replace=False)
            rows = np.concatenate([floor_rows, extra])
        else:
            rows = floor_rows

    try:
        scores = silhouette_samples(X[rows], labels[rows], metric=metric)
    except ValueError:
        return None

    scores_by_row = np.full(n, np.nan)
    scores_by_row[rows] = scores
    return scores_by_row


def _dispersion(
    samples: np.ndarray, centroid: np.ndarray, *, metric: str
) -> tuple[float, float]:
    """Max and 95th-percentile distance of a cluster's samples from its centroid."""
    distances = pairwise_distances(
        samples, centroid.reshape(1, -1), metric=metric
    ).ravel()
    return float(np.max(distances)), float(np.percentile(distances, 95))


def _nearest_rival(
    distances_to_centroids: np.ndarray, own_class: int, classes: np.ndarray
) -> float | None:
    """Distance to the closest centroid of a different class."""
    rival = distances_to_centroids[classes != own_class]
    finite = rival[np.isfinite(rival)]
    if finite.size == 0:
        return None
    return float(np.min(finite))


@timed
def compute_geometry_measures(
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
    population_ids = [int(population_id) for population_id in np.unique(population)]
    centroid_matrix = np.stack(
        [
            np.asarray(centroids[population_id], dtype=np.float64)
            for population_id in population_ids
        ]
    )
    classes = np.array(
        [population_class[population_id] for population_id in population_ids]
    )

    centroid_distances = pairwise_distances(centroid_matrix, metric=metric)
    np.fill_diagonal(centroid_distances, np.inf)

    silhouettes = _approx_silhouette(
        X,
        population,
        metric=metric,
        max_samples=silhouette_max_samples,
        min_per_cluster=silhouette_min_per_cluster,
        random_state=random_state,
    )

    result: dict[int, dict[str, float | None]] = {}

    for i, population_id in enumerate(population_ids):
        in_population = population == population_id
        max_dispersion, p95_dispersion = _dispersion(
            X[in_population], centroid_matrix[i], metric=metric
        )
        dist_to_nearest_rival = _nearest_rival(
            centroid_distances[i], classes[i], classes
        )

        if silhouettes is None:
            p5_silhouette, frac_at_risk = None, None
        else:
            own_silhouettes = silhouettes[in_population]
            finite_silhouettes = own_silhouettes[np.isfinite(own_silhouettes)]
            if len(finite_silhouettes) == 0:
                p5_silhouette, frac_at_risk = None, None
            else:
                p5_silhouette = float(np.percentile(finite_silhouettes, 5))
                frac_at_risk = float(np.mean(finite_silhouettes < 0))

        result[population_id] = {
            "max_dispersion": max_dispersion,
            "p95_dispersion": p95_dispersion,
            "dist_to_nearest_rival": dist_to_nearest_rival,
            "p5_silhouette": p5_silhouette,
            "frac_at_risk": frac_at_risk,
        }

    return result
