import json
import logging
import sys
from pathlib import Path
from typing import Any, Tuple

import pandas as pd
import numpy as np
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.decomposition import PCA

from src.common.config import load_config
from src.common.logging import setup_logger
from src.data.io import load_df, save_df
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
    target_class: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_cols = list(cfg.data.num_cols) + list(cfg.data.cat_cols)
    label_col = "multi_" + cfg.data.label_col

    train_mask = train_df[label_col] == target_class
    val_mask = val_df[label_col] == target_class
    test_mask = test_df[label_col] == target_class

    pca = PCA(n_components=0.95)
    pca.fit(train_df.loc[train_mask, feature_cols])
    train_reduced = pca.transform(train_df.loc[train_mask, feature_cols])
    val_reduced = pca.transform(val_df.loc[val_mask, feature_cols])
    test_reduced = pca.transform(test_df.loc[test_mask, feature_cols])

    model, _ = kmeans_grid_search(train_reduced, n_clusters_range=list(range(1, 10, 1)))
    train_clusters = model.predict(train_reduced)
    val_clusters = model.predict(val_reduced)
    test_clusters = model.predict(test_reduced)

    # Shift all labels greater than target_class by number of clusters - 1
    num_clusters = len(np.unique(train_clusters))
    shift_amount = num_clusters - 1

    train_df.loc[train_df[label_col] > target_class, label_col] += shift_amount
    val_df.loc[val_df[label_col] > target_class, label_col] += shift_amount
    test_df.loc[test_df[label_col] > target_class, label_col] += shift_amount

    # Replace target_class with cluster labels (starting from target_class)
    train_df.loc[train_mask, label_col] = target_class + train_clusters
    val_df.loc[val_mask, label_col] = target_class + val_clusters
    test_df.loc[test_mask, label_col] = target_class + test_clusters

    logger.info(f"Expanded class {target_class} into {num_clusters} subclasses")

    return train_df, val_df, test_df


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


if __name__ == "__main__":
    main()
