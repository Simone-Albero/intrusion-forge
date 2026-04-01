import logging
import sys
from pathlib import Path

import pandas as pd

from src.common.config import load_config
from src.common.logging import setup_logger
from src.common.utils import load_from_json, save_to_json

from src.data.analyze import (
    compute_clusters_metadata,
    compute_df_metadata,
    get_df_info,
)
from src.data.io import load_df, save_df
from sklearn.preprocessing import RobustScaler

from src.data.preprocessing import (
    LogTransformer,
    TopNHashEncoder,
    build_preprocessor,
    drop_nans,
    encode_labels,
    ml_split,
    query_filter,
    rare_category_filter,
    random_undersample_df,
)
from src.ml.clustering import assign_clusters, make_hdbscan_cluster_fn

setup_logger()
logger = logging.getLogger(__name__)


def preprocess_df(
    df,
    num_cols,
    cat_cols,
    label_col,
    filter_query,
    min_cat_count,
    train_frac,
    val_frac,
    test_frac,
    random_state,
    top_n,
    hash_buckets,
):
    """Preprocess dataframe: filter, encode, scale, and split."""
    df = drop_nans(df, num_cols + cat_cols + [label_col])
    df = query_filter(df, filter_query)
    df = rare_category_filter(df, [label_col], min_count=min_cat_count)

    train_df, val_df, test_df = ml_split(
        df,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        random_state=random_state,
        label_col=label_col,
    )
    train_df = random_undersample_df(train_df, label_col, random_state)

    preprocessor = build_preprocessor(
        num_cols=num_cols,
        cat_cols=cat_cols,
        num_steps=[
            ("log_transformer", LogTransformer()),
            ("scaler", RobustScaler()),
        ],
        cat_steps=[
            ("top_n_encoder", TopNHashEncoder(top_n=top_n, hash_buckets=hash_buckets)),
        ],
    )
    logger.info("Preprocessor: %s", preprocessor)
    preprocessor.fit(train_df)
    train_df, val_df, test_df = (
        preprocessor.transform(split) for split in [train_df, val_df, test_df]
    )

    return train_df, val_df, test_df


def run_clustering(
    train_df,
    val_df,
    test_df,
    feature_cols,
    label_col,
    clustering_type,
    cluster_classes,
    ignore_clusters,
    seed,
    cluster_col="cluster",
):
    """Apply the requested clustering strategy and return updated splits with centroids."""
    cluster_fn = make_hdbscan_cluster_fn()

    if cluster_classes is not None and len(cluster_classes) == 0:
        cluster_classes = train_df[label_col].unique().tolist()
        logger.info(
            "No cluster_classes specified, using all classes: %s", cluster_classes
        )

    if clustering_type == "over_all":
        train_df["_split"] = "train"
        val_df["_split"] = "val"
        test_df["_split"] = "test"

        combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
        combined, centroids = assign_clusters(
            combined,
            feature_cols,
            cluster_fn,
            label_col,
            cluster_classes,
            cluster_col,
        )

        train_df = (
            combined[combined["_split"] == "train"]
            .drop("_split", axis=1)
            .reset_index(drop=True)
        )
        val_df = (
            combined[combined["_split"] == "val"]
            .drop("_split", axis=1)
            .reset_index(drop=True)
        )
        test_df = (
            combined[combined["_split"] == "test"]
            .drop("_split", axis=1)
            .reset_index(drop=True)
        )

    elif clustering_type == "over_splits":
        centroids = {}
        offset = 0
        updated = []
        for split_df in [train_df, val_df, test_df]:
            split_df, split_c = assign_clusters(
                split_df,
                feature_cols,
                cluster_fn,
                label_col,
                cluster_classes,
                cluster_col,
            )
            if offset:
                split_df[cluster_col] += offset
                split_c = {str(int(k) + offset): v for k, v in split_c.items()}
            offset = int(split_df[cluster_col].max()) + 1
            centroids.update(split_c)
            updated.append(split_df)
        train_df, val_df, test_df = updated

    else:
        raise ValueError(f"Unknown clustering_type: '{clustering_type}'")

    if ignore_clusters:
        for df in [val_df, test_df]:
            df.drop(
                df[df[cluster_col].isin(ignore_clusters)].index,
                inplace=True,
            )
            df.reset_index(drop=True, inplace=True)
        train_df = train_df[~train_df[cluster_col].isin(ignore_clusters)].reset_index(
            drop=True
        )
        train_df = random_undersample_df(train_df, label_col, seed)

    return train_df, val_df, test_df, centroids


def recompute_clusters_metadata(cfg):
    """Reload processed splits and recompute clusters_meta.json.

    Useful to refresh derived sections without rerunning the full pipeline.
    Centroids are reconstructed from the existing clusters_meta.json.
    """
    num_cols = list(cfg.data.num_cols) if cfg.data.num_cols else []
    cat_cols = list(cfg.data.cat_cols) if cfg.data.cat_cols else []
    processed_data_path = Path(cfg.path.processed_data)
    json_logs_path = Path(cfg.path.json_logs)

    train_df, val_df, test_df = (
        load_df(str(processed_data_path / f"{split}.{cfg.data.extension}"))
        for split in ("train", "val", "test")
    )

    existing = load_from_json(json_logs_path / "data/clusters_meta.json")
    centroids = {
        cid: stats["centroid"]
        for cid, stats in existing["cluster_stats"].items()
        if stats.get("centroid") is not None
    }

    clusters_metadata = compute_clusters_metadata(
        train_df,
        val_df,
        test_df,
        cfg.data.label_col,
        cluster_col="cluster",
        centroids=centroids,
        feature_cols=num_cols + cat_cols,
        metric=cfg.distance_metric,
    )
    save_to_json(clusters_metadata, json_logs_path / "data/clusters_meta.json")
    logger.info("clusters_meta.json recomputed and saved.")


def prepare(cfg):
    """Prepare data given a configuration object."""
    num_cols = list(cfg.data.num_cols) if cfg.data.num_cols else []
    cat_cols = list(cfg.data.cat_cols) if cfg.data.cat_cols else []
    label_col = cfg.data.label_col

    raw_data_path = Path(cfg.path.raw_data)
    processed_data_path = Path(cfg.path.processed_data)
    json_logs_path = Path(cfg.path.json_logs)

    logger.info("Loading and preprocessing data...")
    df = load_df(str(raw_data_path))

    df_info = get_df_info(df, label_col=label_col)
    save_to_json(df_info, json_logs_path / "data/df_info.json")

    train_df, val_df, test_df = preprocess_df(
        df,
        num_cols,
        cat_cols,
        label_col,
        cfg.data.filter_query,
        cfg.data.min_cat_count,
        cfg.data.train_frac,
        cfg.data.val_frac,
        cfg.data.test_frac,
        cfg.seed,
        cfg.data.top_n,
        cfg.data.hash_buckets,
    )

    train_df, val_df, test_df = (
        df.reset_index(drop=True) for df in [train_df, val_df, test_df]
    )

    centroids = None
    if cfg.clustering_type is not None:
        train_df, val_df, test_df, centroids = run_clustering(
            train_df,
            val_df,
            test_df,
            feature_cols=num_cols + cat_cols,
            label_col=label_col,
            clustering_type=cfg.clustering_type,
            cluster_classes=cfg.cluster_classes,
            ignore_clusters=cfg.ignore_clusters,
            seed=cfg.seed,
        )

    train_df, val_df, test_df, label_mapping = encode_labels(
        train_df, val_df, test_df, label_col, dst_label_col=f"encoded_{label_col}"
    )

    logger.info("Saving processed data...")
    for split_name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        save_df(split_df, processed_data_path / f"{split_name}.{cfg.data.extension}")

    logger.info("Computing and saving metadata...")
    metadata = compute_df_metadata(
        {"train": train_df, "val": val_df, "test": test_df},
        label_col,
        num_cols,
        cat_cols,
        cfg.data.benign_tag,
        label_mapping,
    )
    save_to_json(metadata, json_logs_path / "data/df_meta.json")

    if cfg.clustering_type is not None and centroids is not None:
        clusters_metadata = compute_clusters_metadata(
            train_df,
            val_df,
            test_df,
            label_col,
            cluster_col="cluster",
            centroids=centroids,
            feature_cols=num_cols + cat_cols,
            metric=cfg.distance_metric,
        )
        save_to_json(clusters_metadata, json_logs_path / "data/clusters_meta.json")

    return train_df, val_df, test_df, metadata


def main():
    """Main entry point for data preparation."""
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )
    # prepare(cfg)
    recompute_clusters_metadata(cfg)


if __name__ == "__main__":
    main()
