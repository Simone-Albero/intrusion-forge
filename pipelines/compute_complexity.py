import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pipelines import paths_from_cfg
from src.core.config import load_config, save_config, to_container
from src.core.io import load_df
from src.core.log import (
    JSONSubscriber,
    LogBundle,
    LogDispatcher,
    setup_logger,
)
from src.core.utils import flush_timing, load_from_json, skip_if_unchanged, timed
from src.domain.analysis.complexity import (
    ComplexityGraph,
    compute_complexity_from_graph,
    prepare_complexity_graph,
)

setup_logger()
logger = logging.getLogger(__name__)


def cluster_class_map(y_cluster: np.ndarray, y_class: np.ndarray) -> dict[str, int]:
    """Full cluster_id → class_id map from unfiltered train labels (noise included)."""
    return {str(c): int(y_class[y_cluster == c][0]) for c in np.unique(y_cluster)}


@timed
def compute_cluster_complexity(
    graph: ComplexityGraph,
    *,
    noise_cluster_ids: list[int],
    cluster_to_class: dict[str, int],
    top_k_clusters: int,
    metric: str,
    silhouette_max_samples: int,
    silhouette_min_per_cluster: int,
    random_state: int,
) -> list[dict]:
    """Compute per-cluster complexity measures and attach the class of each cluster."""
    logger.info("Computing cluster-level complexity measures ...")
    complexity = compute_complexity_from_graph(
        graph,
        graph.y_cluster,
        top_k_clusters=top_k_clusters,
        metric=metric,
        noise_cluster_ids=set(noise_cluster_ids),
        silhouette_max_samples=silhouette_max_samples,
        silhouette_min_per_cluster=silhouette_min_per_cluster,
        random_state=random_state,
    )

    return [
        {
            "cluster_id": int(cid),
            **measures,
            "cluster_class": cluster_to_class[str(cid)],
        }
        for cid, measures in complexity.items()
    ]


@timed
def compute_class_complexity(
    graph: ComplexityGraph,
    *,
    top_k_clusters: int,
    metric: str,
    silhouette_max_samples: int,
    silhouette_min_per_cluster: int,
    random_state: int,
) -> list[dict]:
    """Compute per-class complexity measures, treating each class as a partition."""
    logger.info("Computing class-level complexity measures ...")
    complexity = compute_complexity_from_graph(
        graph,
        graph.y_class,
        top_k_clusters=top_k_clusters,
        metric=metric,
        noise_cluster_ids=None,
        silhouette_max_samples=silhouette_max_samples,
        silhouette_min_per_cluster=silhouette_min_per_cluster,
        random_state=random_state,
    )
    return [{"class_id": int(cid), **measures} for cid, measures in complexity.items()]


def _fingerprint(
    cfg, train_df: pd.DataFrame, *, columns: list[str], noise_cluster_ids: list[int]
) -> dict:
    """The config complexity runs under, plus a digest of the rows and regions."""
    # Hashed as loaded, `cluster` included: the measures describe the regions, which
    # prepare's record identifies only by their inputs.
    digest = hashlib.blake2b(digest_size=16)
    digest.update(
        pd.util.hash_pandas_object(train_df[columns], index=False).to_numpy().tobytes()
    )
    digest.update(json.dumps(sorted(noise_cluster_ids)).encode())
    return {
        # Bumped when the code changes what a config computes: old records never match.
        "schema": 1,
        # The data keys complexity reads; the rest reach it only through the digest.
        "data": {
            "num_cols": list(cfg.data.num_cols),
            "cat_cols": list(cfg.data.cat_cols),
            "label_col": cfg.data.label_col,
        },
        "complexity": to_container(cfg.complexity),
        "seed": cfg.seed,
        "data_digest": digest.hexdigest(),
    }


def main() -> None:
    """Entry point for the dataset-level complexity stage."""
    cfg = load_config(
        config_path=Path(__file__).parent.parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )
    paths = paths_from_cfg(cfg)
    prepared_path = paths.shared / "prepare_fingerprint.json"
    if not prepared_path.exists():
        raise FileNotFoundError(f"Missing {prepared_path}: run `make prepare` first.")
    prepared_distance = load_from_json(prepared_path)["clustering"]["distance"]
    if cfg.complexity.distance != prepared_distance:
        raise ValueError(
            f"complexity.distance {cfg.complexity.distance!r} must match the "
            f"{prepared_distance!r} the regions were built with: re-run `make prepare`."
        )

    num_cols = list(cfg.data.num_cols)
    cat_cols = list(cfg.data.cat_cols)
    label_col = f"encoded_{cfg.data.label_col}"
    train_df = load_df(str(paths.processed_data / f"train.{cfg.data.extension}"))
    clusters_meta = load_from_json(paths.shared / "metadata/clusters_meta.json")
    noise_cluster_ids = clusters_meta["noise_cluster_ids"]

    cluster_output = paths.shared / "complexity.json"
    class_output = paths.shared / "class_complexity.json"
    record = paths.shared / "complexity_fingerprint.json"
    fingerprint = _fingerprint(
        cfg,
        train_df,
        columns=num_cols + cat_cols + [label_col, "cluster"],
        noise_cluster_ids=noise_cluster_ids,
    )
    # One record for both outputs: they share the graph, which costs most of the stage.
    if skip_if_unchanged(
        [cluster_output, class_output],
        record,
        fingerprint,
        force=cfg.force,
        stage_name="complexity",
    ):
        return
    # Dropped first: an interrupted recompute leaves new outputs under an old record.
    record.unlink(missing_ok=True)

    X_num = (
        train_df[num_cols].to_numpy(dtype=np.float64)
        if num_cols
        else np.empty((len(train_df), 0))
    )
    X_cat = train_df[cat_cols].to_numpy() if cat_cols else None
    y_class = train_df[label_col].to_numpy(dtype=np.int64)

    bus = LogDispatcher()
    bus.subscribe(JSONSubscriber(paths.shared))

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

    cluster_to_class = cluster_class_map(y_cluster, y_class)
    cluster_complexity = compute_cluster_complexity(
        graph,
        noise_cluster_ids=noise_cluster_ids,
        cluster_to_class=cluster_to_class,
        top_k_clusters=cfg.complexity.top_k_clusters,
        metric=cfg.complexity.distance,
        silhouette_max_samples=cfg.complexity.silhouette_max_samples,
        silhouette_min_per_cluster=cfg.complexity.silhouette_min_per_cluster,
        random_state=cfg.seed,
    )
    bus.publish(LogBundle.from_dict({"json/complexity": cluster_complexity}))
    logger.info("Cluster complexity published to %s.", cluster_output)

    class_complexity = compute_class_complexity(
        graph,
        top_k_clusters=cfg.complexity.top_k_clusters,
        metric=cfg.complexity.distance,
        silhouette_max_samples=cfg.complexity.silhouette_max_samples,
        silhouette_min_per_cluster=cfg.complexity.silhouette_min_per_cluster,
        random_state=cfg.seed,
    )
    bus.publish(LogBundle.from_dict({"json/class_complexity": class_complexity}))
    logger.info("Class complexity published to %s.", class_output)

    bus.publish(LogBundle.from_dict({"json/complexity_fingerprint": fingerprint}))
    flush_timing(paths.shared / "timing.json")
    save_config(cfg, paths.shared / "config_composed_complexity.json")


if __name__ == "__main__":
    main()
