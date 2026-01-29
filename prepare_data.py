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
from src.common.utils import save_to_json, save_to_pickle, load_from_pickle
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
from src.ml.clustering import kmeans_grid_search

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
    train_df, val_df, test_df, label_mapping = encode_labels(
        train_df, val_df, test_df, label_col, benign_tag
    )

    return train_df, val_df, test_df, label_mapping


def kmeans_to_new_labels(
    train_df,
    val_df,
    test_df,
    target_class,
    feature_cols,
    label_col,
    train_idx=None,
    val_idx=None,
    test_idx=None,
    min_samples=100,
):
    """Expand target class into subclasses using clustering."""

    train_mask = (
        train_df.index.isin(train_idx)
        if train_idx is not None
        else pd.Series(True, index=train_df.index)
    ) & (train_df[label_col] == target_class)
    val_mask = (
        val_df.index.isin(val_idx)
        if val_idx is not None
        else pd.Series(True, index=val_df.index)
    ) & (val_df[label_col] == target_class)
    test_mask = (
        test_df.index.isin(test_idx)
        if test_idx is not None
        else pd.Series(True, index=test_df.index)
    ) & (test_df[label_col] == target_class)

    clusterer, _ = kmeans_grid_search(
        train_df.loc[train_mask, feature_cols], n_clusters_range=range(2, 10)
    )
    train_clusters = clusterer.predict(train_df.loc[train_mask, feature_cols])
    val_clusters = clusterer.predict(val_df.loc[val_mask, feature_cols])
    test_clusters = clusterer.predict(test_df.loc[test_mask, feature_cols])

    cluster_counts = pd.Series(train_clusters).value_counts()
    valid_clusters = cluster_counts[cluster_counts >= min_samples].index.tolist()

    cluster_to_label = {
        cluster_id: f"{target_class}_{i}" for i, cluster_id in enumerate(valid_clusters)
    }

    train_df = _update_cluster_labels(
        train_df, train_mask, train_clusters, cluster_to_label, label_col
    )
    val_df = _update_cluster_labels(
        val_df, val_mask, val_clusters, cluster_to_label, label_col
    )
    test_df = _update_cluster_labels(
        test_df, test_mask, test_clusters, cluster_to_label, label_col
    )

    logger.info(
        f"Expanded class {target_class} into {len(valid_clusters)} subclasses: {list(cluster_to_label.values())}"
    )

    return train_df, val_df, test_df


def fn_to_new_labels(
    train_df,
    val_df,
    test_df,
    target_class,
    feature_cols,
    label_col,
    label_mapping,
    cfg,
):
    class_label = label_mapping.get(target_class, str(target_class))

    for split_name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        class_indices = load_from_pickle(
            Path(cfg.path.pickles) / f"class_indices/{split_name}.pkl"
        )
        fn_idx_1 = class_indices[target_class][0]
        fn_idx_2 = class_indices[target_class][4]

        split_df.loc[fn_idx_1, label_col] = f"{class_label}_1"
        split_df.loc[fn_idx_2, label_col] = f"{class_label}_2"

    train_df, val_df, test_df, label_mapping = encode_labels(
        train_df, val_df, test_df, label_col, cfg.data.benign_tag
    )

    return train_df, val_df, test_df, label_mapping


def _update_cluster_labels(df, mask, clusters, cluster_to_label, label_col):
    """Update labels based on cluster assignments."""
    indices = df[mask].index

    for cluster_id, new_label in cluster_to_label.items():
        cluster_mask = clusters == cluster_id
        df.loc[indices[cluster_mask], label_col] = new_label

    return df


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
    raw_data_path = (
        Path(cfg.path.raw_data) / cfg.data.name / f"{cfg.data.file_name}.csv"
    )
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

    # train_df.loc[
    #     train_df[label_col].str.contains("DOS", case=False, na=False),
    #     label_col,
    # ] = "Class_A"
    # val_df.loc[
    #     val_df[label_col].str.contains("DOS", case=False, na=False),
    #     label_col,
    # ] = "Class_A"
    # test_df.loc[
    #     test_df[label_col].str.contains("DOS", case=False, na=False),
    #     label_col,
    # ] = "Class_A"

    # train_df, val_df, test_df, label_mapping = encode_labels(
    #     train_df, val_df, test_df, label_col, cfg.data.benign_tag
    # )

    # Save processed data
    logger.info("Saving processed data...")
    save_df(
        train_df,
        processed_data_path
        / cfg.data.name
        / f"{cfg.data.file_name}_train.{cfg.data.extension}",
    )
    save_df(
        val_df,
        processed_data_path
        / cfg.data.name
        / f"{cfg.data.file_name}_val.{cfg.data.extension}",
    )
    save_df(
        test_df,
        processed_data_path
        / cfg.data.name
        / f"{cfg.data.file_name}_test.{cfg.data.extension}",
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
        json_logs_path / "metadata" / f"df_{cfg.run_id}.json",
    )


if __name__ == "__main__":
    main()
