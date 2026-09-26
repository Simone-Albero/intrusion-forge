from collections.abc import Callable
from inspect import signature

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


def build_cluster_fn(
    algorithms: dict[str, dict],
    *,
    max_fit_samples: int,
    random_state: int,
    reporter: Reporter,
    max_clusters: int | None,
    min_clusters: int | None,
    grid_target_cluster_size: int | None,
    resolution_weight: float,
    metric: str,
) -> ClusterFn:
    """Build a ClusterFn from a single {algorithm_name: params} config entry."""
    if len(algorithms) != 1:
        raise ValueError(
            f"build_cluster_fn expects exactly one algorithm, got {len(algorithms)}: "
            f"{list(algorithms)}"
        )
    ((name, params),) = algorithms.items()
    fit_fn = ClusteringFactory.get(name)
    grid, fixed = _split_grid_fixed(params or {})
    derives_n_clusters = name in _N_CLUSTERS_ALGOS and bool(grid_target_cluster_size)

    # Checked here, before any fit, so the error names the algorithm and the key.
    configured = grid.keys() | fixed.keys()
    unknown = sorted(configured - signature(fit_fn).parameters.keys())
    if unknown:
        raise TypeError(f"Clustering algorithm {name!r} takes no parameter {unknown}.")
    if min_clusters is not None and not ("n_clusters" in grid or derives_n_clusters):
        raise TypeError(
            f"Clustering algorithm {name!r} takes no min_clusters here: it floors the "
            "n_clusters of a sweep, and this configuration sweeps none."
        )
    supplied = {"max_fit_samples", "random_state"} | (
        {"n_clusters"} if derives_n_clusters else set()
    )
    clashing = sorted(configured & supplied)
    if clashing:
        raise ValueError(
            f"Clustering algorithm {name!r}: {clashing} are set by the pipeline, "
            "not in the algorithm's params (a fixed n_clusters needs "
            "grid_target_cluster_size: null)."
        )

    def _fn(X_num: np.ndarray) -> np.ndarray:
        common = {
            "max_fit_samples": max_fit_samples,
            "random_state": random_state,
            **fixed,
        }
        algo_grid = dict(grid)
        if derives_n_clusters:
            k_cap = max(2, max_fit_samples // 25)
            if max_clusters is not None:
                k_cap = min(k_cap, max_clusters)
            algo_grid["n_clusters"] = _n_clusters_grid(
                X_num.shape[0], grid_target_cluster_size, k_cap
            )
        if algo_grid:
            report, best_labels = grid_search(
                X_num,
                fit_fn,
                algo_grid,
                resolution_weight=resolution_weight,
                min_clusters=min_clusters,
                score_metric=metric,
                **common,
            )
            reporter(name, report)
            if best_labels is not None:
                return best_labels
            return fit_fn(X_num, **report["best"]["combo"], **common)
        return fit_fn(X_num, **common)

    return _fn
