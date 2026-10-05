from collections.abc import Callable
from inspect import signature

import numpy as np

from src.domain.clustering.base import ClusterFn, grid_search, merge_small_clusters
from src.domain.clustering.factory import ClusteringFactory
from src.domain.clustering.hardness import KdnNodes

Reporter = Callable[[str, dict], None]


def _split_grid_fixed(params: dict) -> tuple[dict, dict]:
    """Split params into ({list → grid}, {scalar → fixed})."""
    grid = {k: v for k, v in params.items() if isinstance(v, list)}
    fixed = {k: v for k, v in params.items() if not isinstance(v, list)}
    return grid, fixed


_N_CLUSTERS_ALGOS = ("kmeans", "spectral", "birch")


def _n_clusters_grid(
    n_class: int, target_size: int, max_n_clusters: int, levels: int = 7
) -> list[int]:
    """Data-relative `n_clusters` candidates: a geometric band from `target_size` up."""
    sizes = (target_size * (2**level) for level in range(levels))
    candidates = {min(max_n_clusters, max(2, round(n_class / size))) for size in sizes}
    return sorted(candidates)


def build_cluster_fn(
    algorithms: dict[str, dict],
    *,
    max_fit_samples: int,
    random_state: int,
    reporter: Reporter,
    max_clusters: int,
    min_cluster_floor: int,
    hardness_k: int,
    eval_rows_per_train_row: float,
    nodes: KdnNodes,
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
    derives_n_clusters = name in _N_CLUSTERS_ALGOS

    # Checked here, before any fit, so the error names the algorithm and the key.
    configured = grid.keys() | fixed.keys()
    unknown = sorted(configured - signature(fit_fn).parameters.keys())
    if unknown:
        raise TypeError(f"Clustering algorithm {name!r} takes no parameter {unknown}.")
    supplied = {"max_fit_samples", "random_state"} | (
        {"n_clusters"} if derives_n_clusters else set()
    )
    clashing = sorted(configured & supplied)
    if clashing:
        raise ValueError(
            f"Clustering algorithm {name!r}: {clashing} are set by the pipeline, "
            "not in the algorithm's params."
        )

    def _fn(
        X: np.ndarray, *, row_ids: np.ndarray, label: object
    ) -> tuple[np.ndarray, int, int]:
        fixed_fit_params = {
            "max_fit_samples": max_fit_samples,
            "random_state": random_state,
            **fixed,
        }
        full_grid = dict(grid)
        if derives_n_clusters:
            max_n_clusters = max(
                2, min(max_fit_samples // min_cluster_floor, max_clusters)
            )
            full_grid["n_clusters"] = _n_clusters_grid(
                X.shape[0], min_cluster_floor, max_n_clusters
            )
        report, best_labels = grid_search(
            X,
            fit_fn,
            full_grid,
            row_ids=row_ids,
            label=label,
            nodes=nodes,
            hardness_k=hardness_k,
            eval_rows_per_train_row=eval_rows_per_train_row,
            min_cluster_floor=min_cluster_floor,
            max_clusters=max_clusters,
            merge_metric=metric,
            **fixed_fit_params,
        )
        reporter(name, report)
        if best_labels is not None:
            best = report["best"]
            return best_labels, best["n_merged_clusters"], best["n_merged"]
        refit_labels = fit_fn(X, **report["best"]["combo"], **fixed_fit_params)
        # Refit on the full class: the sweep's merge counts described the scored
        # subsample, at a floor scaled down to match it, so they don't describe this.
        final_labels, n_merged_clusters, n_merged = merge_small_clusters(
            X, refit_labels, min_size=min_cluster_floor, metric=metric
        )
        return final_labels, n_merged_clusters, n_merged

    return _fn
