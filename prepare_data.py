import json
import logging
import sys
from pathlib import Path
from typing import Any, Tuple
import pickle

import pandas as pd
import numpy as np
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.decomposition import PCA

from src.common.config import load_config
from src.common.logging import setup_logger
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
from src.ml.clustering import kmeans_grid_search, hdbscan_grid_search

setup_logger()
logger = logging.getLogger(__name__)


def preprocess_df(
    df: pd.DataFrame,
    cfg: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocess dataframe for machine learning tasks.
    Applies filtering, encoding, scaling, and train/val/test splitting.
    """
    # Extract parameters from cfg
    num_cols = list(cfg.data.num_cols)
    cat_cols = list(cfg.data.cat_cols)
    label_col = cfg.data.label_col
    benign_tag = cfg.data.benign_tag

    query = cfg.data.filter_query
    min_cat_count = cfg.data.min_cat_count
    top_n = cfg.data.top_n
    hash_buckets = cfg.data.hash_buckets
    add_log_freq = cfg.data.add_log_freq
    add_is_unk = cfg.data.add_is_unk

    train_frac = cfg.data.train_frac
    val_frac = cfg.data.val_frac
    test_frac = cfg.data.test_frac
    random_state = cfg.seed

    # Global preprocessing steps
    df = drop_nans(df, num_cols + cat_cols + [label_col])
    df = query_filter(df, query)
    df = rare_category_filter(df, [label_col], min_count=min_cat_count)

    train_df, val_df, test_df = ml_split(
        df,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        random_state=random_state,
        label_col=label_col,
    )

    # Make sklearn transformers/ColumnTransformer return pandas DataFrames
    set_config(transform_output="pandas")

    num_pipeline = Pipeline(
        steps=[
            ("quantile_clipper", QuantileClipper()),
            ("log_transformer", LogTransformer()),
            ("scaler", RobustScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
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

    # Encode label column
    label_encoder = LabelEncoder()
    multi_label_col = f"multi_{label_col}"
    train_df[multi_label_col] = label_encoder.fit_transform(train_df[label_col])
    val_df[multi_label_col] = label_encoder.transform(val_df[label_col])
    test_df[multi_label_col] = label_encoder.transform(test_df[label_col])

    if benign_tag is not None:
        bin_label_col = f"bin_{label_col}"
        train_df[bin_label_col] = (train_df[label_col] != benign_tag).astype(int)
        val_df[bin_label_col] = (val_df[label_col] != benign_tag).astype(int)
        test_df[bin_label_col] = (test_df[label_col] != benign_tag).astype(int)

    label_mapping = {
        int(i): str(class_name) for i, class_name in enumerate(label_encoder.classes_)
    }
    logger.info(f"Label mapping: {label_mapping}")

    class_counts = train_df[multi_label_col].value_counts().sort_index()
    num_classes = len(class_counts)
    total_samples = len(train_df)
    class_weights = total_samples / (num_classes * class_counts)
    log_weights = np.log1p(class_weights)  # log(1 + x)
    normalized_weights = log_weights / log_weights.max()
    class_weights_list = normalized_weights.tolist()

    # Prepare and save metadata
    metadata = {
        "label_mapping": label_mapping,
        "dataset_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "total": len(train_df) + len(val_df) + len(test_df),
        },
        "samples_per_class": {
            "train": train_df[label_col].value_counts().to_dict(),
            "val": val_df[label_col].value_counts().to_dict(),
            "test": test_df[label_col].value_counts().to_dict(),
        },
        "numerical_columns": num_cols
        + [f"{col}__is_unk" for col in cat_cols if add_is_unk]
        + [f"{col}__log_freq" for col in cat_cols if add_log_freq],
        "categorical_columns": cat_cols,
        "multi_label_column": multi_label_col,
        "bin_label_column": f"bin_{label_col}" if benign_tag is not None else None,
        "benign_tag": benign_tag,
        "num_classes": len(train_df[multi_label_col].unique()),
        "class_weights": class_weights_list,
    }

    metadata_path = Path(cfg.path.json_logs) / "df_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Dataset metadata saved to {metadata_path}")

    logger.info("Saving processed data...")
    processed_data_dir = Path(cfg.path.processed_data)
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        output_path = (
            processed_data_dir
            / f"{cfg.data.file_name}_{split_name}.{cfg.data.extension}"
        )
        save_df(split_df, str(output_path))
        logger.info(f"Saved {split_name} data: {len(split_df)} samples")

    return train_df, val_df, test_df


def expand_class_labels(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Any,
    target_class: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Expand a target class into multiple subclasses using clustering on false negatives.

    Returns:
        Updated train, val, and test dataframes with new class labels
    """
    feature_cols = list(cfg.data.num_cols)
    label_col = "multi_" + cfg.data.label_col

    # Load false negatives
    fn_train = _load_false_negatives(cfg, "train", target_class)
    fn_test = _load_false_negatives(cfg, "test", target_class)

    # Get masks for false negatives of target class
    train_fn_mask = train_df.index.isin(fn_train) & (
        train_df[label_col] == target_class
    )
    test_fn_mask = test_df.index.isin(fn_test) & (test_df[label_col] == target_class)

    # Dimensionality reduction with PCA
    # pca = PCA(n_components=0.95)
    # train_reduced = pca.fit_transform(train_df.loc[train_fn_mask, feature_cols])
    # test_reduced = pca.transform(test_df.loc[test_fn_mask, feature_cols])

    print(f"{train_df[label_col].nunique()} classes before expansion.")

    # Clustering
    clusterer, _ = kmeans_grid_search(
        train_df.loc[train_fn_mask, feature_cols], n_clusters_range=range(2, 10)
    )
    train_clusters = clusterer.predict(train_df.loc[train_fn_mask, feature_cols])
    new_cluster_count = len(np.unique(train_clusters))
    logger.info(f"Expanded class {target_class} into {new_cluster_count} clusters")
    test_clusters = clusterer.predict(test_df.loc[test_fn_mask, feature_cols])

    train_df, cluster_map = _assign_cluster_labels(
        train_df, train_fn_mask, train_clusters, label_col
    )
    test_df, _ = _assign_cluster_labels(
        test_df,
        test_fn_mask,
        test_clusters,
        label_col,
        cluster_to_label_map=cluster_map,
    )

    metadata_path = Path(cfg.path.json_logs) / "df_metadata.json"
    with open(metadata_path, "rb") as f:
        metadata = json.load(f)

    num_new_classes = train_df[label_col].nunique()
    metadata["num_classes"] = num_new_classes
    print(f"{num_new_classes} classes after expansion.")

    class_counts = train_df[label_col].value_counts().sort_index()
    num_classes = len(class_counts)
    total_samples = len(train_df)
    class_weights = total_samples / (num_classes * class_counts)
    log_weights = np.log1p(class_weights)  # log(1 + x)
    normalized_weights = log_weights / log_weights.max()
    class_weights_list = normalized_weights.tolist()
    metadata["class_weights"] = class_weights_list

    class_name = metadata["label_mapping"][str(target_class)]

    actual_new_classes = num_new_classes - len(metadata["label_mapping"])
    for i in range(actual_new_classes):
        new_class_id = class_counts.index.max() - actual_new_classes + 1 + i
        metadata["label_mapping"][str(new_class_id)] = f"{class_name}_subclass_{i+1}"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    for split_name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        output_path = (
            Path(cfg.path.processed_data)
            / f"{cfg.data.file_name}_{split_name}.{cfg.data.extension}"
        )
        save_df(split_df, str(output_path))
        logger.info(f"Saved {split_name} data: {len(split_df)} samples")

    return train_df, val_df, test_df


def _load_false_negatives(cfg: Any, split: str, target_class: int) -> list:
    """Load false negative indices for a specific class and split."""
    pickle_path = Path(cfg.path.pickles) / f"false_negatives_{split}.pkl"
    with open(pickle_path, "rb") as f:
        fn_dict = pickle.load(f)
    return fn_dict.get(target_class, [])


def _assign_cluster_labels(
    df: pd.DataFrame,
    mask: pd.Series,
    clusters: np.ndarray,
    label_col: str,
    min_samples_per_cluster: int = 100,
    cluster_to_label_map: dict = None,
) -> Tuple[pd.DataFrame, dict]:
    """Assign new class labels based on cluster assignments.

    Returns:
        Tuple of (updated dataframe, cluster_to_label mapping)
    """
    new_label = df[label_col].max() + 1
    indices = df[mask].index

    if cluster_to_label_map is None:
        # Training phase: create the mapping
        cluster_to_label_map = {}
        for cluster_id in np.unique(clusters):
            cluster_mask = clusters == cluster_id
            cluster_size = cluster_mask.sum()

            if cluster_size < min_samples_per_cluster:
                logger.info(
                    f"Skipping cluster {cluster_id} with only {cluster_size} samples"
                )
                continue

            cluster_to_label_map[cluster_id] = new_label
            df.loc[indices[cluster_mask], label_col] = new_label
            logger.info(
                f"Assigned label {new_label} to cluster {cluster_id} ({cluster_size} samples)"
            )
            new_label += 1
    else:
        # Test phase: use existing mapping
        for cluster_id in np.unique(clusters):
            if cluster_id not in cluster_to_label_map:
                cluster_size = (clusters == cluster_id).sum()
                logger.info(
                    f"Skipping unmapped cluster {cluster_id} ({cluster_size} samples in test)"
                )
                continue

            cluster_mask = clusters == cluster_id
            assigned_label = cluster_to_label_map[cluster_id]
            df.loc[indices[cluster_mask], label_col] = assigned_label
            logger.info(
                f"Assigned label {assigned_label} to cluster {cluster_id} ({cluster_mask.sum()} samples)"
            )

    return df, cluster_to_label_map


def main() -> None:
    """Main entry point for data preparation."""
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )

    # Load raw data
    logger.info("Loading raw data...")
    raw_data_path = Path(cfg.path.raw_data) / f"{cfg.data.file_name}.csv"
    df = load_df(str(raw_data_path))

    # Preprocess data
    logger.info("Preprocessing data...")
    train_df, val_df, test_df = preprocess_df(df, cfg)

    # Expand class labels if specified
    train_df, val_df, test_df = load_data_splits(
        Path(cfg.path.processed_data), cfg.data.file_name, cfg.data.extension
    )
    expand_class_labels(train_df, val_df, test_df, cfg, target_class=3)


if __name__ == "__main__":
    main()
