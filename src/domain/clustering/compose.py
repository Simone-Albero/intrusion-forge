from collections.abc import Callable

import numpy as np

from src.domain.clustering import ClusteringFactory
from src.domain.clustering.base import ClusterFn, grid_search, make_hybrid_silhouette_fn

Reporter = Callable[[str, dict], None]  # (algo_name, grid_search_result) -> None


def _split_grid_fixed(params: dict) -> tuple[dict, dict]:
    """Split params into ({list → grid}, {scalar → fixed})."""
    grid = {k: v for k, v in params.items() if isinstance(v, list)}
    fixed = {k: v for k, v in params.items() if not isinstance(v, list)}
    return grid, fixed


# algorithms clustering on mixed features: scored with the Gower-hybrid silhouette
_HYBRID_SCORED_ALGOS = ("kprototypes",)

# algorithms parameterised by an explicit cluster count: their `n_clusters` grid
# is derived per class from the target average cluster size (see _n_clusters_grid)
# instead of a static, dataset-blind list.
_N_CLUSTERS_ALGOS = ("kmeans", "spectral", "birch", "kprototypes")


def _n_clusters_grid(
    n_class: int, target_size: int, k_cap: int, levels: int = 7
) -> list[int]:
    """Data-relative `n_clusters` candidates: a geometric band of average cluster
    sizes from `target_size` (finest) up by powers of two, each clamped to
    [2, k_cap] and de-duplicated.
    """
    sizes = (target_size * (2**i) for i in range(levels))
    ks = {min(k_cap, max(2, round(n_class / s))) for s in sizes}
    return sorted(ks)


def resolution_aware_floor(n_class: int, target_size: int, floor_cap: int) -> int:
    """Absorption floor for a class's finest clustering candidate, capped at `floor_cap`.

    Ties the floor to the same finest-candidate average size `_n_clusters_grid`
    targets, so a class too small (or a `target_size` too aggressive) for that
    candidate to clear `floor_cap` gets a proportionally lower floor instead of
    having its finest partition absorbed wholesale.
    """
    finest_k = max(2, round(n_class / target_size))
    finest_avg_size = n_class / finest_k
    return min(floor_cap, max(5, round(0.5 * finest_avg_size)))


def _make_single_cluster_fn(
    name: str,
    params: dict,
    max_fit_samples: int,
    random_state: int,
    reporter: Reporter | None = None,
    metric: str = "cosine",
    max_clusters: int | None = None,
    min_clusters: int | None = None,
    grid_target_cluster_size: int | None = None,
    resolution_weight: float = 0.1,
) -> ClusterFn:
    """Build a ClusterFn for a single registered algorithm."""
    fit_fn = ClusteringFactory.get(name)
    # An algorithm with no explicit params (e.g. kmeans, whose n_clusters grid is
    # derived per class) parses from YAML as None.
    grid, fixed = _split_grid_fixed(params or {})
    silhouette_fn = (
        make_hybrid_silhouette_fn(metric=metric, random_state=random_state)
        if name in _HYBRID_SCORED_ALGOS
        else None
    )

    def _fn(X_num: np.ndarray, X_cat: np.ndarray | None = None) -> np.ndarray:
        common = {
            "max_fit_samples": max_fit_samples,
            "random_state": random_state,
            **fixed,
        }
        algo_grid = dict(grid)
        if name in _N_CLUSTERS_ALGOS and grid_target_cluster_size:
            # Per-class n_clusters band, sized so the finest candidate targets
            # ~grid_target_cluster_size points/cluster. Capped so each cluster
            # keeps ≥25 subsample points (a meaningful silhouette) and, when
            # `max_clusters` is set, so the total cluster count stays within the
            # complexity budget. Replaces any static n_clusters list from config.
            k_cap = max(2, max_fit_samples // 25)
            if max_clusters is not None:
                k_cap = min(k_cap, max_clusters)
            algo_grid["n_clusters"] = _n_clusters_grid(
                X_num.shape[0], grid_target_cluster_size, k_cap
            )
        if algo_grid:
            # min_clusters only applies to n_clusters-parameterised algorithms: there,
            # cluster count and average size are the same knob, so the guard stays
            # coherent with the (separate) absorption floor. For hdbscan, count and
            # size are decoupled (min_cluster_size/min_samples) — a raw-count guard
            # would happily admit many tiny clusters that the absorption floor then
            # wipes out, making things worse rather than better.
            effective_min_clusters = min_clusters if name in _N_CLUSTERS_ALGOS else None
            result = grid_search(
                X_num,
                X_cat,
                fit_fn,
                algo_grid,
                resolution_weight=resolution_weight,
                min_clusters=effective_min_clusters,
                silhouette_fn=silhouette_fn,
                **common,
            )
            if reporter is not None:
                reporter(name, result)
            return fit_fn(X_num, X_cat=X_cat, **result["best"]["combo"], **common)
        return fit_fn(X_num, X_cat=X_cat, **common)

    return _fn


def build_cluster_fn(
    algorithms: dict[str, dict],
    max_fit_samples: int,
    random_state: int,
    reporter: Reporter | None = None,
    metric: str = "cosine",
    max_clusters: int | None = None,
    min_clusters: int | None = None,
    grid_target_cluster_size: int | None = None,
    resolution_weight: float = 0.1,
) -> ClusterFn:
    """Build a ClusterFn from a single {algorithm_name: params} entry.

    `reporter` is invoked once with the algorithm's grid-search result. Exactly
    one algorithm is supported; small genuine clusters are handled post-hoc by
    the resolution-aware absorption floor in the pipeline, not by a pre-grid
    prune — `min_clusters` only guards against the *selected* partition having
    too few clusters overall, regardless of individual cluster size.
    """
    if len(algorithms) != 1:
        raise ValueError(
            f"build_cluster_fn expects exactly one algorithm, got {len(algorithms)}: "
            f"{list(algorithms)}"
        )
    (name, params), = algorithms.items()
    return _make_single_cluster_fn(
        name,
        params,
        max_fit_samples,
        random_state,
        reporter=reporter,
        metric=metric,
        max_clusters=max_clusters,
        min_clusters=min_clusters,
        grid_target_cluster_size=grid_target_cluster_size,
        resolution_weight=resolution_weight,
    )
