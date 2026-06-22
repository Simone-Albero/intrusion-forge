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


def _make_single_cluster_fn(
    name: str,
    params: dict,
    max_fit_samples: int,
    random_state: int,
    reporter: Reporter | None = None,
    metric: str = "cosine",
) -> ClusterFn:
    """Build a ClusterFn for a single registered algorithm."""
    fit_fn = ClusteringFactory.get(name)
    grid, fixed = _split_grid_fixed(params)
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
        if grid:
            result = grid_search(
                X_num,
                X_cat,
                fit_fn,
                grid,
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
) -> ClusterFn:
    """Build a ClusterFn from a single {algorithm_name: params} entry.

    `reporter` is invoked once with the algorithm's grid-search result. Exactly
    one algorithm is supported; degenerate K values are handled post-hoc by the
    small-cluster absorption in the pipeline, not by a pre-grid prune.
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
    )
