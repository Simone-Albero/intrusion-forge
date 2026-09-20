from collections.abc import Callable

import numpy as np

from src.domain.clustering.base import ClusterFn, grid_search
from src.domain.clustering.factory import ClusteringFactory

Reporter = Callable[[str, dict], None]


def _split_grid_fixed(params: dict) -> tuple[dict, dict]:
    """Split params into ({list → grid}, {scalar → fixed})."""
    grid = {k: v for k, v in params.items() if isinstance(v, list)}
    fixed = {k: v for k, v in params.items() if not isinstance(v, list)}
    return grid, fixed


_N_CLUSTERS_ALGOS = ("kmeans", "spectral", "birch")


def _n_clusters_grid(
    n_class: int, target_size: int, k_cap: int, levels: int = 7
) -> list[int]:
    """Data-relative `n_clusters` candidates: a geometric band from `target_size` up."""
    sizes = (target_size * (2**i) for i in range(levels))
    ks = {min(k_cap, max(2, round(n_class / s))) for s in sizes}
    return sorted(ks)


def resolution_aware_floor(n_class: int, target_size: int, floor_cap: int) -> int:
    """Absorption floor tied to the finest candidate's average size, capped at `floor_cap`."""
    finest_k = max(2, round(n_class / target_size))
    finest_avg_size = n_class / finest_k
    return min(floor_cap, max(5, round(0.5 * finest_avg_size)))


def _make_single_cluster_fn(
    name: str,
    params: dict,
    max_fit_samples: int,
    random_state: int,
    reporter: Reporter | None = None,
    max_clusters: int | None = None,
    min_clusters: int | None = None,
    grid_target_cluster_size: int | None = None,
    resolution_weight: float = 0.1,
) -> ClusterFn:
    """Build a ClusterFn for a single registered algorithm."""
    fit_fn = ClusteringFactory.get(name)
    grid, fixed = _split_grid_fixed(params or {})

    def _fn(X_num: np.ndarray, X_cat: np.ndarray | None = None) -> np.ndarray:
        common = {
            "max_fit_samples": max_fit_samples,
            "random_state": random_state,
            **fixed,
        }
        algo_grid = dict(grid)
        if name in _N_CLUSTERS_ALGOS and grid_target_cluster_size:
            k_cap = max(2, max_fit_samples // 25)
            if max_clusters is not None:
                k_cap = min(k_cap, max_clusters)
            algo_grid["n_clusters"] = _n_clusters_grid(
                X_num.shape[0], grid_target_cluster_size, k_cap
            )
        if algo_grid:
            effective_min_clusters = min_clusters if name in _N_CLUSTERS_ALGOS else None
            result = grid_search(
                X_num,
                X_cat,
                fit_fn,
                algo_grid,
                resolution_weight=resolution_weight,
                min_clusters=effective_min_clusters,
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
    max_clusters: int | None = None,
    min_clusters: int | None = None,
    grid_target_cluster_size: int | None = None,
    resolution_weight: float = 0.1,
) -> ClusterFn:
    """Build a ClusterFn from a single {algorithm_name: params} entry."""
    if len(algorithms) != 1:
        raise ValueError(
            f"build_cluster_fn expects exactly one algorithm, got {len(algorithms)}: "
            f"{list(algorithms)}"
        )
    ((name, params),) = algorithms.items()
    return _make_single_cluster_fn(
        name,
        params,
        max_fit_samples,
        random_state,
        reporter=reporter,
        max_clusters=max_clusters,
        min_clusters=min_clusters,
        grid_target_cluster_size=grid_target_cluster_size,
        resolution_weight=resolution_weight,
    )
