import json
import logging
import sys
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler

from src.common.config import load_config
from src.common.logging import setup_logger
from src.common.utils import save_to_json, save_to_pickle
from src.data.io import load_df, save_df, load_data_splits
from src.data.preprocessing import (
    QuantileClipper,
    LogTransformer,
    TopNHashEncoder,
    drop_nans,
    ml_split,
    query_filter,
    rare_category_filter,
)
from src.ml.clustering import kmeans_grid_search

setup_logger()
logger = logging.getLogger(__name__)


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

    set_config(transform_output="pandas")

    # Build preprocessing pipelines
    num_pipeline = Pipeline(
        [
            ("quantile_clipper", QuantileClipper()),
            ("log_transformer", LogTransformer()),
            ("scaler", RobustScaler()),
        ]
    )

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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    preprocessor.fit(train_df)
    train_df = preprocessor.transform(train_df)
    val_df = preprocessor.transform(val_df)
    test_df = preprocessor.transform(test_df)

    # Encode labels
    label_encoder = LabelEncoder()
    multi_label_col = f"multi_{label_col}"
    for df_split in [train_df, val_df, test_df]:
        if df_split is train_df:
            df_split[multi_label_col] = label_encoder.fit_transform(df_split[label_col])
        else:
            df_split[multi_label_col] = label_encoder.transform(df_split[label_col])

        if benign_tag:
            df_split[f"bin_{label_col}"] = (df_split[label_col] != benign_tag).astype(
                int
            )

    label_mapping = {int(i): str(name) for i, name in enumerate(label_encoder.classes_)}
    logger.info(f"Label mapping: {label_mapping}")

    return train_df, val_df, test_df, label_mapping


def expand_class_labels(
    train_df,
    val_df,
    test_df,
    target_class,
    feature_cols,
    label_col,
    train_idx=None,
    val_idx=None,
    test_idx=None,
):
    """Expand target class into subclasses using clustering on false negatives."""
    train_mask = train_df.index.isin(train_idx) & (train_df[label_col] == target_class)
    val_mask = val_df.index.isin(val_idx) & (val_df[label_col] == target_class)
    test_mask = test_df.index.isin(test_idx) & (test_df[label_col] == target_class)

    clusterer, _ = kmeans_grid_search(
        train_df.loc[train_mask, feature_cols], n_clusters_range=range(2, 10)
    )
    train_clusters = clusterer.predict(train_df.loc[train_mask, feature_cols])
    val_clusters = clusterer.predict(val_df.loc[val_mask, feature_cols])
    test_clusters = clusterer.predict(test_df.loc[test_mask, feature_cols])

    train_df, cluster_map = _assign_cluster_labels(
        train_df, train_mask, train_clusters, label_col
    )
    val_df, _ = _assign_cluster_labels(
        val_df,
        val_mask,
        val_clusters,
        label_col,
        cluster_to_label_map=cluster_map,
    )
    test_df, _ = _assign_cluster_labels(
        test_df,
        test_mask,
        test_clusters,
        label_col,
        cluster_to_label_map=cluster_map,
    )

    return train_df, val_df, test_df


def _assign_cluster_labels(
    df, mask, clusters, label_col, min_samples=100, cluster_to_label_map=None
):
    """Assign new class labels based on cluster assignments."""
    indices = df[mask].index

    if cluster_to_label_map is None:
        # Training: create mapping
        cluster_to_label_map = {}
        new_label = df[label_col].max() + 1

        for cluster_id in np.unique(clusters):
            cluster_mask = clusters == cluster_id
            if cluster_mask.sum() < min_samples:
                continue

            cluster_to_label_map[cluster_id] = new_label
            df.loc[indices[cluster_mask], label_col] = new_label
            new_label += 1
    else:
        # Test: use existing mapping
        for cluster_id in np.unique(clusters):
            if cluster_id in cluster_to_label_map:
                cluster_mask = clusters == cluster_id
                df.loc[indices[cluster_mask], label_col] = cluster_to_label_map[
                    cluster_id
                ]

    return df, cluster_to_label_map


def main():
    """Main entry point for data preparation."""
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )

    num_cols = list(cfg.data.num_cols)
    cat_cols = list(cfg.data.cat_cols)
    label_col = cfg.data.label_col
    raw_data_path = Path(cfg.path.raw_data) / f"{cfg.data.file_name}.csv"
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

    # Optional: expand classes based on false negatives
    # train_df, val_df, test_df = load_data_splits(
    #     processed_data_path, cfg.data.file_name, cfg.data.extension
    # )
    # train_df, val_df, test_df = expand_class_labels(
    #     train_df,
    #     val_df,
    #     test_df,
    #     target_class=3,
    #     feature_cols=num_cols + cat_cols,
    #     label_col=label_col,
    # )

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
        json_logs_path / "df_metadata.json",
    )


if __name__ == "__main__":
    main()
