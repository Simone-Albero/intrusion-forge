import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.common.config import load_config
from src.common.logging import setup_logger
from src.common.utils import save_to_json

from src.data.io import load_df, save_df
from src.data.preprocessing import (
    LogTransformer,
    TopNHashEncoder,
    drop_nans,
    encode_labels,
    ml_split,
    query_filter,
    rare_category_filter,
    random_undersample_df,
)

from src.ml.clustering import hdbscan_grid_search

setup_logger()
logger = logging.getLogger(__name__)


def get_df_info(df):
    """Return basic information about the dataframe."""
    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "memory_usage": df.memory_usage(deep=True).sum(),
        "feature_info": {},
    }

    for col in df.columns:
        info["feature_info"][col] = {
            "dtype": df[col].dtype,
            "unique_count": df[col].nunique(),
        }

    return info


def compute_splits_metadata(
    train_df,
    val_df,
    test_df,
    label_col,
    num_cols,
    cat_cols,
    benign_tag,
    label_mapping=None,
):
    """Compute and return metadata dictionary for the dataset."""
    class_counts = train_df[label_col].value_counts().sort_index()
    class_weights = len(train_df) / (len(class_counts) * class_counts)
    class_weights = np.log1p(class_weights) / np.log1p(class_weights).max()
    class_weights_list = class_weights.tolist()

    metadata = {
        "label_mapping": label_mapping or {},
        "dataset_sizes": {
            split: len(df_split)
            for split, df_split in zip(
                ["train", "val", "test"], [train_df, val_df, test_df]
            )
        },
        "samples_per_class": {
            split: df_split[label_col].value_counts().to_dict()
            for split, df_split in zip(
                ["train", "val", "test"], [train_df, val_df, test_df]
            )
        },
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "benign_tag": benign_tag,
        "num_classes": train_df[label_col].nunique(),
        "class_weights": class_weights_list,
    }

    return metadata


def build_preprocessor(num_cols, cat_cols, top_n, hash_buckets):
    """Build a ColumnTransformer for numerical and categorical features."""
    set_config(transform_output="pandas")
    transformers = []

    if num_cols:
        num_pipeline = Pipeline(
            [
                # ("quantile_clipper", QuantileClipper()),
                ("log_transformer", LogTransformer()),
                ("scaler", RobustScaler()),
            ]
        )
        logger.info(f"Numerical pipeline steps: {num_pipeline.steps}")
        transformers.append(("num", num_pipeline, num_cols))

    if cat_cols:
        cat_pipeline = Pipeline(
            [
                (
                    "top_n_encoder",
                    TopNHashEncoder(top_n=top_n, hash_buckets=hash_buckets),
                ),
            ]
        )
        logger.info(f"Categorical pipeline steps: {cat_pipeline.steps}")
        transformers.append(("cat", cat_pipeline, cat_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


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

    preprocessor = build_preprocessor(num_cols, cat_cols, top_n, hash_buckets)
    preprocessor.fit(train_df)
    train_df, val_df, test_df = (
        preprocessor.transform(split) for split in [train_df, val_df, test_df]
    )

    train_df, val_df, test_df, label_mapping = encode_labels(
        train_df,
        val_df,
        test_df,
        label_col,
        dst_label_col=f"encoded_{label_col}",
    )

    return train_df, val_df, test_df, label_mapping


def compute_clusters(
    df, feature_cols, label_col, classes=None, thresholds=None, dst_col="cluster"
):
    """Compute HDBSCAN clusters for specified classes and thresholds."""
    df[dst_col] = -1
    offset = 0
    if classes is None:
        logger.info("Computing clusters for all data...")
        _, labels, proba, info = hdbscan_grid_search(df[feature_cols].values)
        core_mask = (labels != -1) & (proba >= thresholds[0] if thresholds else 0.0)
        df[dst_col] = labels
        return df, info

    if len(classes) > len(thresholds):
        thresholds = thresholds + [0.0] * (len(classes) - len(thresholds))

    infos = {}
    for cls, threshold in zip(classes, thresholds):
        logger.info(
            f"Computing clusters for class '{cls}' with threshold {threshold}..."
        )
        cls_mask = df[label_col] == cls
        _, labels, proba, info = hdbscan_grid_search(
            df.loc[cls_mask, feature_cols].values
        )
        core_mask = (labels != -1) & (proba >= threshold)

        cls_indices = df[cls_mask & core_mask].index
        df.loc[cls_indices, dst_col] = labels[core_mask] + offset
        offset += labels.max() + 1
        infos[cls] = info
    return df, infos


def clusters_over_splits(
    train_df, val_df, test_df, feature_cols, label_col, dst_col, classes, thresholds
):
    train_df, _ = compute_clusters(
        train_df, feature_cols, label_col, classes, thresholds, dst_col=dst_col
    )
    val_df, _ = compute_clusters(
        val_df, feature_cols, label_col, classes, thresholds, dst_col=dst_col
    )
    test_df, _ = compute_clusters(
        test_df, feature_cols, label_col, classes, thresholds, dst_col=dst_col
    )

    return train_df, val_df, test_df


def clusters_over_all(
    train_df, val_df, test_df, feature_cols, label_col, dst_col, classes, thresholds
):
    train_df["_split"] = "train"
    val_df["_split"] = "val"
    test_df["_split"] = "test"

    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined_df, _ = compute_clusters(
        combined_df, feature_cols, label_col, classes, thresholds, dst_col=dst_col
    )

    train_df = (
        combined_df[combined_df["_split"] == "train"]
        .drop("_split", axis=1)
        .reset_index(drop=True)
    )

    val_df = (
        combined_df[combined_df["_split"] == "val"]
        .drop("_split", axis=1)
        .reset_index(drop=True)
    )
    test_df = (
        combined_df[combined_df["_split"] == "test"]
        .drop("_split", axis=1)
        .reset_index(drop=True)
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
    cluster_thresholds,
    ignore_clusters,
    seed,
    cluster_col="cluster",
):
    """Apply the requested clustering strategy and return updated splits with metadata."""
    cluster_fn = {
        "over_splits": clusters_over_splits,
        "over_all": clusters_over_all,
    }.get(clustering_type)

    if cluster_fn is None:
        raise ValueError(f"Unknown clustering_type: '{clustering_type}'")

    train_df, val_df, test_df = cluster_fn(
        train_df,
        val_df,
        test_df,
        feature_cols=feature_cols,
        label_col=label_col,
        dst_col=cluster_col,
        classes=cluster_classes,
        thresholds=cluster_thresholds,
    )

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

    clusters_metadata = {
        "num_clusters": int(train_df[cluster_col].nunique()),
        "cluster_counts": train_df[cluster_col].value_counts().to_dict(),
    }
    for cls in cluster_classes:
        cls_mask = train_df[label_col] == cls
        clusters_metadata[f"clusters_in_class_{cls}"] = (
            train_df.loc[cls_mask, cluster_col].value_counts().to_dict()
        )

    return train_df, val_df, test_df, clusters_metadata


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

    df_info = get_df_info(df)
    save_to_json(df_info, json_logs_path / "raw_df_info.json")

    train_df, val_df, test_df, label_mapping = preprocess_df(
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

    if cfg.clustering_type is not None:
        train_df, val_df, test_df, clusters_metadata = run_clustering(
            train_df,
            val_df,
            test_df,
            feature_cols=num_cols + cat_cols,
            label_col=label_col,
            clustering_type=cfg.clustering_type,
            cluster_classes=cfg.cluster_classes,
            cluster_thresholds=cfg.cluster_thresholds,
            ignore_clusters=cfg.ignore_clusters,
            seed=cfg.seed,
        )
        save_to_json(clusters_metadata, json_logs_path / "clusters.json")

    train_df, val_df, test_df, label_mapping = encode_labels(
        train_df, val_df, test_df, label_col, cfg.data.benign_tag
    )

    logger.info("Saving processed data...")
    for split_name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        save_df(split_df, processed_data_path / f"{split_name}.{cfg.data.extension}")

    logger.info("Computing and saving metadata...")
    metadata = compute_splits_metadata(
        train_df,
        val_df,
        test_df,
        label_col,
        num_cols,
        cat_cols,
        cfg.data.benign_tag,
        label_mapping,
    )
    save_to_json(metadata, json_logs_path / "df_metadata.json")

    return train_df, val_df, test_df, metadata


def main():
    """Main entry point for data preparation."""
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )
    prepare(cfg)


if __name__ == "__main__":
    main()
