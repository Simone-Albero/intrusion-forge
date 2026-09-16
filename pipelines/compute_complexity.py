import logging
import sys
from pathlib import Path

import numpy as np

from pipelines import paths_from_cfg
from src.core.config import load_config
from src.core.io import load_df
from src.core.log import (
    JSONSubscriber,
    LogBundle,
    LogDispatcher,
    setup_logger,
)
from src.core.utils import flush_timing, load_from_json, skip_if_exists, timed
from src.domain.analysis.complexity import (
    ComplexityGraph,
    compute_complexity_from_graph,
    prepare_complexity_graph,
)

setup_logger(log_file="resources/logs.txt")
logger = logging.getLogger(__name__)


def cluster_class_map(y_cluster: np.ndarray, y_class: np.ndarray) -> dict[str, int]:
    """Full cluster_id → class_id map from unfiltered train labels (noise included)."""
    return {
        str(c): int(y_class[y_cluster == c][0]) for c in np.unique(y_cluster) if c != -1
    }


@timed
def compute_cluster_complexity(
    graph: ComplexityGraph,
    noise_cluster_ids: list[int],
    cluster_to_class: dict[str, int],
    *,
    top_k_clusters: int,
    metric: str,
    random_state: int,
) -> dict:
    """Compute per-cluster complexity measures and attach the class of each cluster."""
    logger.info("Computing cluster-level complexity measures ...")
    complexity = compute_complexity_from_graph(
        graph,
        graph.y_cluster,
        top_k_clusters=top_k_clusters,
        metric=metric,
        noise_cluster_ids=set(noise_cluster_ids),
        random_state=random_state,
    )

    return {
        str(cid): {**measures, "cluster_class": cluster_to_class.get(str(cid))}
        for cid, measures in complexity.items()
    }


@timed
def compute_class_complexity(
    graph: ComplexityGraph,
    *,
    top_k_clusters: int,
    metric: str,
    random_state: int,
) -> dict:
    """Compute per-class complexity measures, treating each class as a partition."""
    logger.info("Computing class-level complexity measures ...")
    return compute_complexity_from_graph(
        graph,
        graph.y_class,
        top_k_clusters=top_k_clusters,
        metric=metric,
        noise_cluster_ids=None,
        random_state=random_state,
    )


def main() -> None:
    """Entry point for the dataset-level complexity stage."""
    cfg = load_config(
        config_path=Path(__file__).parent.parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )
    paths = paths_from_cfg(cfg)

    cluster_marker = paths.shared / "complexity.json"
    class_marker = paths.shared / "class_complexity.json"
    run_cluster = not skip_if_exists(cluster_marker, cfg.complexity.force, "complexity")
    run_class = not skip_if_exists(
        class_marker, cfg.complexity.force, "class_complexity"
    )
    if not (run_cluster or run_class):
        return

    num_cols = list(cfg.data.num_cols) if cfg.data.num_cols else []
    cat_cols = list(cfg.data.cat_cols) if cfg.data.cat_cols else []
    ext = cfg.data.extension

    train_df = load_df(str(paths.processed_data / f"train.{ext}"))

    X_num = (
        train_df[num_cols].to_numpy(dtype=np.float64)
        if num_cols
        else np.empty((len(train_df), 0))
    )
    X_cat = train_df[cat_cols].to_numpy() if cat_cols else None
    y_class = train_df[f"encoded_{cfg.data.label_col}"].to_numpy(dtype=np.int64)

    bus = LogDispatcher()
    bus.subscribe(JSONSubscriber(paths.shared))

    graph = None
    noise_cluster_ids: list[int] = []
    if run_cluster or run_class:
        clusters_meta = load_from_json(paths.shared / "metadata/clusters_meta.json")
        noise_cluster_ids = clusters_meta.get("noise_cluster_ids", [])

        y_cluster = train_df["cluster"].to_numpy(dtype=np.int64)
        if noise_cluster_ids:
            genuine = ~np.isin(y_cluster, noise_cluster_ids)
            X_num_g = X_num[genuine]
            X_cat_g = X_cat[genuine] if X_cat is not None else None
            y_class_g = y_class[genuine]
            y_cluster_g = y_cluster[genuine]
        else:
            X_num_g, X_cat_g, y_class_g, y_cluster_g = X_num, X_cat, y_class, y_cluster

        graph = prepare_complexity_graph(
            X_num_g,
            X_cat_g,
            y_class_g,
            y_cluster_g,
            k=cfg.complexity.k,
            max_samples=cfg.complexity.max_complexity_samples,
            min_per_cluster=cfg.complexity.min_subsample_per_cluster,
            metric=cfg.complexity.distance,
            random_state=cfg.seed,
        )

    if run_cluster:
        cluster_to_class = cluster_class_map(y_cluster, y_class)
        cluster_complexity = compute_cluster_complexity(
            graph,
            noise_cluster_ids,
            cluster_to_class,
            top_k_clusters=cfg.complexity.top_k_clusters,
            metric=cfg.complexity.distance,
            random_state=cfg.seed,
        )
        bus.publish(LogBundle.from_dict({"json/complexity": cluster_complexity}))
        logger.info("Cluster complexity published to %s.", cluster_marker)

    if run_class:
        class_complexity = compute_class_complexity(
            graph,
            top_k_clusters=cfg.complexity.top_k_clusters,
            metric=cfg.complexity.distance,
            random_state=cfg.seed,
        )
        bus.publish(LogBundle.from_dict({"json/class_complexity": class_complexity}))
        logger.info("Class complexity published to %s.", class_marker)

    flush_timing(paths.shared / "timing.json")


if __name__ == "__main__":
    main()
