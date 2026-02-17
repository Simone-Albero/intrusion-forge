import logging
import sys
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

# Suppress sklearn deprecation warnings from hdbscan
# warnings.filterwarnings(
#     "ignore", message=".*force_all_finite.*", category=FutureWarning
# )

from src.common.config import load_config
from src.common.logging import setup_logger
from src.common.utils import save_to_json
from src.data.io import load_df, save_df, load_data_splits
from src.data.preprocessing import (
    QuantileClipper,
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


def df_info(df):
    """Print basic information about the dataframe."""
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info("DataFrame info:")
    logger.info(df.info())
    logger.info("DataFrame description:")
    logger.info(df.describe(include="all"))

    for col in df.columns:
        logger.info(f"Column '{col}' info:")
        num_unique = df[col].nunique()
        logger.info(f"Unique values: {num_unique}")
        logger.info(f"Dtype: {df[col].dtype}")
        logger.info(f"Top 5 frequent values:\n{df[col].value_counts().head()}")


def compute_df_metadata(
    train_df,
    val_df,
    test_df,
    label_col,
    num_cols,
    cat_cols,
    add_is_unk,
    add_log_freq,
    benign_tag,
    label_mapping=None,
):
    """Compute and return metadata dictionary for the dataset."""
    # Compute class weights
    class_counts = train_df[label_col].value_counts().sort_index()
    class_weights = len(train_df) / (len(class_counts) * class_counts)
    class_weights = np.log1p(class_weights) / np.log1p(class_weights).max()
    class_weights_list = class_weights.tolist()

    metadata = {
        "label_mapping": label_mapping if label_mapping else {},
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
        "numerical_columns": num_cols
        + ([f"{col}__is_unk" for col in cat_cols] if add_is_unk else [])
        + ([f"{col}__log_freq" for col in cat_cols] if add_log_freq else []),
        "categorical_columns": cat_cols,
        "benign_tag": benign_tag,
        "num_classes": train_df[label_col].nunique(),
        "class_weights": class_weights_list,
    }

    return metadata


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
    add_log_freq,
    add_is_unk,
    benign_tag,
):
    """Preprocess dataframe: filter, encode, scale, and split."""
    # Filter and clean data
    df = drop_nans(df, num_cols + cat_cols + [label_col])
    df = query_filter(df, filter_query)
    df = rare_category_filter(df, [label_col], min_count=min_cat_count)

    # Split data
    train_df, val_df, test_df = ml_split(
        df,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        random_state=random_state,
        label_col=label_col,
    )

    train_df = random_undersample_df(train_df, label_col, random_state)

    set_config(transform_output="pandas")

    # Build preprocessing pipelines
    num_pipeline = None
    if num_cols and len(num_cols) > 0:
        num_pipeline = Pipeline(
            [
                ("quantile_clipper", QuantileClipper()),
                ("log_transformer", LogTransformer()),
                ("scaler", RobustScaler()),
            ]
        )

    cat_pipeline = None
    if cat_cols and len(cat_cols) > 0:
        cat_pipeline = Pipeline(
            [
                (
                    "top_n_encoder",
                    TopNHashEncoder(
                        top_n=top_n,
                        hash_buckets=hash_buckets,
                        add_log_freq=add_log_freq,
                        add_is_unk=add_is_unk,
                    ),
                ),
            ]
        )

    # Build transformers list, excluding None pipelines
    transformers = []
    if num_pipeline is not None:
        logger.info(f"Numerical pipeline steps: {num_pipeline.steps}")
        transformers.append(("num", num_pipeline, num_cols))
    if cat_pipeline is not None:
        logger.info(f"Categorical pipeline steps: {cat_pipeline.steps}")
        transformers.append(("cat", cat_pipeline, cat_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    preprocessor.fit(train_df)
    train_df = preprocessor.transform(train_df)
    val_df = preprocessor.transform(val_df)
    test_df = preprocessor.transform(test_df)

    # Encode labels
    train_df, val_df, test_df, label_mapping = encode_labels(
        train_df, val_df, test_df, label_col, benign_tag
    )

    return train_df, val_df, test_df, label_mapping


def compute_clusters(df, feature_cols, label_col, classes, thresholds):
    df["cluster"] = -1
    offset = 0
    for cls, threshold in zip(classes, thresholds):
        logger.info(
            f"Computing clusters for class '{cls}' with threshold {threshold}..."
        )
        cls_mask = df[label_col] == cls
        model, labels, proba, info = hdbscan_grid_search(
            df.loc[cls_mask, feature_cols].values
        )
        core_mask = (labels != -1) & (proba >= threshold)

        # Get indices of the class subset, then filter by core_mask
        cls_indices = df[cls_mask].index
        core_indices = cls_indices[core_mask]
        df.loc[core_indices, "cluster"] = labels[core_mask] + offset
        offset += labels.max() + 1
        logger.info(f"Class '{cls}': {info}")

    return df


def clusters_over_splits(
    train_df, val_df, test_df, feature_cols, label_col, classes, thresholds
):
    train_df = compute_clusters(train_df, feature_cols, label_col, classes, thresholds)
    val_df = compute_clusters(val_df, feature_cols, label_col, classes, thresholds)
    test_df = compute_clusters(test_df, feature_cols, label_col, classes, thresholds)

    return train_df, val_df, test_df


def clusters_over_all(
    train_df, val_df, test_df, feature_cols, label_col, classes, thresholds
):
    train_df["_split"] = "train"
    val_df["_split"] = "val"
    test_df["_split"] = "test"

    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined_df = compute_clusters(
        combined_df, feature_cols, label_col, classes, thresholds
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


def main():
    """Main entry point for data preparation."""
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )

    num_cols = list(cfg.data.num_cols) if cfg.data.num_cols else []
    cat_cols = list(cfg.data.cat_cols) if cfg.data.cat_cols else []
    label_col = cfg.data.label_col

    raw_data_path = Path(cfg.path.raw_data)
    processed_data_path = Path(cfg.path.processed_data)
    json_logs_path = Path(cfg.path.json_logs)

    # Load and preprocess raw data
    logger.info("Loading and preprocessing data...")
    df = load_df(str(raw_data_path))
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
        cfg.data.add_log_freq,
        cfg.data.add_is_unk,
        cfg.data.benign_tag,
    )

    # Reset indices to ensure alignment with positional indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    if cfg.clustering_type is not None and cfg.clustering_type == "over_splits":
        train_df, val_df, test_df = clusters_over_splits(
            train_df,
            val_df,
            test_df,
            feature_cols=num_cols + cat_cols,
            label_col=label_col,
            classes=cfg.cluster_classes,
            thresholds=cfg.cluster_thresholds,
        )
    elif cfg.clustering_type is not None and cfg.clustering_type == "over_all":
        train_df, val_df, test_df = clusters_over_all(
            train_df,
            val_df,
            test_df,
            feature_cols=num_cols + cat_cols,
            label_col=label_col,
            classes=cfg.cluster_classes,
            thresholds=cfg.cluster_thresholds,
        )

    if "cluster" in train_df.columns:
        splits = {"train": train_df, "val": val_df, "test": test_df}

        cluster_info = {}
        for split, df in splits.items():
            cluster_info[split] = df["cluster"].value_counts().to_dict()

            for cls in cfg.cluster_classes:
                cls_mask = df[label_col] == cls
                cluster_info[f"class_{cls}_{split}"] = (
                    df.loc[cls_mask, "cluster"].value_counts().to_dict()
                )

        logger.info(f"Cluster info: {cluster_info}")
        save_to_json(
            cluster_info,
            json_logs_path / "metadata" / f"clusters_info.json",
        )

    if cfg.ignore_clusters:
        ignore_clusters = set(cfg.ignore_clusters)
        train_df = train_df[~train_df["cluster"].isin(ignore_clusters)].reset_index(
            drop=True
        )
        val_df = val_df[~val_df["cluster"].isin(ignore_clusters)].reset_index(drop=True)
        test_df = test_df[~test_df["cluster"].isin(ignore_clusters)].reset_index(
            drop=True
        )

    train_df, val_df, test_df, label_mapping = encode_labels(
        train_df, val_df, test_df, label_col, cfg.data.benign_tag
    )

    # Save processed data
    logger.info("Saving processed data...")
    save_df(
        train_df,
        processed_data_path / f"{cfg.data.file_name}_train.{cfg.data.extension}",
    )
    save_df(
        val_df,
        processed_data_path / f"{cfg.data.file_name}_val.{cfg.data.extension}",
    )
    save_df(
        test_df,
        processed_data_path / f"{cfg.data.file_name}_test.{cfg.data.extension}",
    )

    # Compute and save metadata
    logger.info("Computing and saving metadata...")
    metadata = compute_df_metadata(
        train_df,
        val_df,
        test_df,
        label_col,
        num_cols,
        cat_cols,
        cfg.data.add_is_unk,
        cfg.data.add_log_freq,
        cfg.data.benign_tag,
        label_mapping,
    )
    save_to_json(
        metadata,
        json_logs_path / "metadata" / f"df.json",
    )


if __name__ == "__main__":
    main()
