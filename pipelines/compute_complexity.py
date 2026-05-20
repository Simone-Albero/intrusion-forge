import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config import load_config, save_config
from src.core.io import load_df
from src.core.log import (
    JSONSubscriber,
    LogBundle,
    LogDispatcher,
    setup_logger,
)
from src.core.paths import OutputPaths
from src.core.utils import flush_timing, load_from_json, skip_if_exists, timed

from src.domain.analysis.complexity import compute_all_complexity_measures

setup_logger(log_file="resources/logs.txt")
logger = logging.getLogger(__name__)


@timed
def compute_complexity(
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    y_class: np.ndarray,
    y_cluster: np.ndarray,
    centroids: dict,
    noise_cluster_ids: list[int],
    k: int,
    top_k_clusters: int,
    min_subsample_per_cluster: int,
    max_complexity_samples: int | None,
    metric: str,
    random_state: int,
) -> dict:
    """Compute per-cluster complexity measures + cluster→class mapping.

    Output schema: {cluster_id: {<measure>: ..., "cluster_class": <int>}}.
    """
    logger.info("Computing complexity measures ...")
    complexity = compute_all_complexity_measures(
        X_num,
        X_cat,
        y_class,
        y_cluster,
        centroids,
        k=k,
        top_k_clusters=top_k_clusters,
        max_samples=max_complexity_samples,
        min_per_cluster=min_subsample_per_cluster,
        metric=metric,
        noise_cluster_ids=set(noise_cluster_ids),
        random_state=random_state,
    )

    cluster_to_class: dict[str, int] = {}
    for cid in np.unique(y_cluster):
        if cid == -1:
            continue
        mask = y_cluster == cid
        cluster_to_class[str(cid)] = int(y_class[mask][0])

    return {
        str(cid): {**measures, "cluster_class": cluster_to_class.get(str(cid))}
        for cid, measures in complexity.items()
    }


def main():
    """Main entry point for complexity computation (dataset-level, shared)."""
    cfg = load_config(
        config_path=Path(__file__).parent.parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )
    paths = OutputPaths(
        processed_data=Path(cfg.path.processed_data),
        shared=Path(cfg.path.shared),
        configs=Path(cfg.path.configs),
        outputs=Path(cfg.path.outputs),
        pickle=Path(cfg.path.pickle),
        models=Path(cfg.path.models),
        figures=Path(cfg.path.figures),
    )

    marker = paths.shared / "complexity.json"
    if skip_if_exists(marker, cfg.complexity.force, "complexity"):
        return

    save_config(cfg, paths.shared / "configs/config_composed_complexity.json")

    num_cols = list(cfg.data.num_cols) if cfg.data.num_cols else []
    cat_cols = list(cfg.data.cat_cols) if cfg.data.cat_cols else []
    ext = cfg.data.extension

    train_df = load_df(str(paths.processed_data / f"train.{ext}"))
    val_df = load_df(str(paths.processed_data / f"val.{ext}"))
    test_df = load_df(str(paths.processed_data / f"test.{ext}"))
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)

    X_num = (
        combined[num_cols].to_numpy(dtype=np.float64)
        if num_cols
        else np.empty((len(combined), 0))
    )
    X_cat = combined[cat_cols].to_numpy() if cat_cols else None
    y_class = combined[f"encoded_{cfg.data.label_col}"].to_numpy(dtype=np.int64)
    y_cluster = combined["cluster"].to_numpy(dtype=np.int64)

    clusters_meta = load_from_json(paths.shared / "clusters_meta.json")
    centroids = clusters_meta.get("centroids", {})
    noise_cluster_ids = clusters_meta.get("noise_cluster_ids", [])

    complexity = compute_complexity(
        X_num=X_num,
        X_cat=X_cat,
        y_class=y_class,
        y_cluster=y_cluster,
        centroids=centroids,
        noise_cluster_ids=noise_cluster_ids,
        k=cfg.complexity.k,
        top_k_clusters=cfg.complexity.top_k_clusters,
        min_subsample_per_cluster=cfg.complexity.min_subsample_per_cluster,
        max_complexity_samples=cfg.complexity.max_complexity_samples,
        metric=cfg.complexity.distance,
        random_state=cfg.seed,
    )

    bus = LogDispatcher()
    bus.subscribe(JSONSubscriber(paths.shared))
    bus.publish(LogBundle.from_dict({"json/complexity": complexity}))
    logger.info("Complexity published to %s.", marker)

    flush_timing(paths.shared / "timing_complexity.json")


if __name__ == "__main__":
    main()
