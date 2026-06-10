import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config import load_config
from src.core.io import load_df, save_df
from src.core.log import (
    JSONSubscriber,
    LogBundle,
    LogDispatcher,
    setup_logger,
)
from src.core.paths import OutputPaths
from src.core.utils import flush_timing, load_from_json, skip_if_exists, timed

from src.domain.analysis.complexity import compute_all_complexity_measures
from src.domain.data.preprocessing import (
    attach_cluster_features,
    cluster_feature_columns,
    scale_columns_on_train,
)

setup_logger(log_file="resources/logs.txt")
logger = logging.getLogger(__name__)


def _compute_class_centroids(
    X_num: np.ndarray,
    y_class: np.ndarray,
    metric: str,
    eps: float = 1e-8,
) -> dict[str, list[float]]:
    """Per-class centroid appropriate for the configured metric.

    cosine    → spherical centroid (mean of L2-normalised samples, re-normalised).
    euclidean → arithmetic mean.
    """
    result: dict[str, list[float]] = {}
    for cid in np.unique(y_class):
        if int(cid) == -1:
            continue
        X_c = X_num[y_class == int(cid)]
        if len(X_c) == 0:
            continue
        if metric == "cosine":
            norms = np.linalg.norm(X_c, axis=1, keepdims=True)
            X_c_norm = X_c / np.maximum(norms, eps)
            sph = X_c_norm.mean(axis=0)
            sph_norm = np.linalg.norm(sph)
            result[str(int(cid))] = (sph / max(sph_norm, eps)).tolist()
        else:
            result[str(int(cid))] = X_c.mean(axis=0).tolist()
    return result


@timed
def compute_cluster_complexity(
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
    logger.info("Computing cluster-level complexity measures ...")
    complexity = compute_all_complexity_measures(
        X_num,
        X_cat,
        y_class=y_class,
        y_cluster=y_cluster,
        centroids=centroids,
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


@timed
def compute_class_complexity(
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    y_class: np.ndarray,
    centroids: dict,
    k: int,
    top_k_clusters: int,
    min_subsample_per_cluster: int,
    max_complexity_samples: int | None,
    metric: str,
    random_state: int,
) -> dict:
    """Compute per-class complexity measures.

    Treats each class as a partition: passes `y_class` as both class labels and
    partition labels to `compute_all_complexity_measures`. Output schema is
    identical to the cluster-level one (neutral keys, no `cluster_class`).
    """
    logger.info("Computing class-level complexity measures ...")
    return compute_all_complexity_measures(
        X_num,
        X_cat,
        y_class=y_class,
        y_cluster=y_class,
        centroids=centroids,
        k=k,
        top_k_clusters=top_k_clusters,
        max_samples=max_complexity_samples,
        min_per_cluster=min_subsample_per_cluster,
        metric=metric,
        noise_cluster_ids=None,
        random_state=random_state,
    )


def main():
    """Main entry point for complexity computation (dataset-level, shared).

    Runs both per-cluster and per-class analyses under the same `cfg.complexity`
    config, producing `shared/complexity.json` and `shared/class_complexity.json`.
    Each stage is independently skippable via its own marker file.
    """
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

    cluster_marker = paths.shared / "complexity.json"
    class_marker = paths.shared / "class_complexity.json"
    meta_marker = paths.shared / "complexity_meta.json"
    run_cluster = not skip_if_exists(cluster_marker, cfg.complexity.force, "complexity")
    run_class = not skip_if_exists(class_marker, cfg.complexity.force, "class_complexity")
    run_extend = not skip_if_exists(meta_marker, cfg.complexity.force, "complexity_extend")
    if not (run_cluster or run_class or run_extend):
        return

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

    bus = LogDispatcher()
    bus.subscribe(JSONSubscriber(paths.shared))

    if run_cluster:
        y_cluster = combined["cluster"].to_numpy(dtype=np.int64)
        clusters_meta = load_from_json(paths.shared / "metadata/clusters_meta.json")
        cluster_centroids = clusters_meta.get("centroids", {})
        noise_cluster_ids = clusters_meta.get("noise_cluster_ids", [])

        cluster_complexity = compute_cluster_complexity(
            X_num=X_num,
            X_cat=X_cat,
            y_class=y_class,
            y_cluster=y_cluster,
            centroids=cluster_centroids,
            noise_cluster_ids=noise_cluster_ids,
            k=cfg.complexity.k,
            top_k_clusters=cfg.complexity.top_k_clusters,
            min_subsample_per_cluster=cfg.complexity.min_subsample_per_cluster,
            max_complexity_samples=cfg.complexity.max_complexity_samples,
            metric=cfg.complexity.distance,
            random_state=cfg.seed,
        )
        bus.publish(LogBundle.from_dict({"json/complexity": cluster_complexity}))
        logger.info("Cluster complexity published to %s.", cluster_marker)

    if run_class:
        class_centroids = _compute_class_centroids(
            X_num, y_class, metric=cfg.complexity.distance
        )

        class_complexity = compute_class_complexity(
            X_num=X_num,
            X_cat=X_cat,
            y_class=y_class,
            centroids=class_centroids,
            k=cfg.complexity.k,
            top_k_clusters=cfg.complexity.top_k_clusters,
            min_subsample_per_cluster=cfg.complexity.min_subsample_per_cluster,
            max_complexity_samples=cfg.complexity.max_complexity_samples,
            metric=cfg.complexity.distance,
            random_state=cfg.seed,
        )
        bus.publish(LogBundle.from_dict({"json/class_complexity": class_complexity}))
        logger.info("Class complexity published to %s.", class_marker)

    if run_extend:
        cluster_features = (
            cluster_complexity if run_cluster else load_from_json(cluster_marker)
        )
        complexity_cols = cluster_feature_columns(cluster_features)
        if not complexity_cols:
            logger.warning("No complexity columns to attach; skipping dataset extension.")
        else:
            splits = {"train": train_df, "val": val_df, "test": test_df}
            extended: dict[str, pd.DataFrame] = {}
            for name, split_df in splits.items():
                merged = attach_cluster_features(split_df, cluster_features)
                before = len(merged)
                merged = merged.dropna(subset=complexity_cols, how="all")
                if before - len(merged):
                    logger.info(
                        "Extend %s: dropped %d/%d rows with no cluster match",
                        name,
                        before - len(merged),
                        before,
                    )
                extended[name] = merged
            extended = scale_columns_on_train(extended, complexity_cols)
            for name, split_df in extended.items():
                save_df(split_df, paths.processed_data / f"{name}_extended.{ext}")
            bus.publish(
                LogBundle.from_dict({"json/complexity_meta": {"columns": complexity_cols}})
            )
            logger.info(
                "Extended splits saved as *_extended.%s (%d complexity columns); meta -> %s",
                ext,
                len(complexity_cols),
                meta_marker,
            )

    flush_timing(paths.shared / "timing.json")


if __name__ == "__main__":
    main()
