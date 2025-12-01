from typing import Tuple
from pathlib import Path
import logging
import sys
import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import set_config

from src.data.preprocessing import (
    drop_nans,
    query_filter,
    rare_category_filter,
    subsample_df,
    ml_split,
    QuantileClipper,
    TopNCategoryEncoder,
)
from src.data.io import load_df, save_df
from src.common.config import load_config
from src.common.logging import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


def preprocess_df(
    df: pd.DataFrame,
    cfg,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Extract parameters from cfg
    num_cols = cfg.data.num_cols
    cat_cols = cfg.data.cat_cols
    label_col = cfg.data.label_col
    benign_tag = cfg.data.benign_tag
    query = cfg.data.filter_query
    min_cat_count = cfg.data.min_cat_count
    top_n_categories = cfg.data.top_n_categories
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

    # Convert to standard Python lists (in case they're OmegaConf ListConfig)
    num_cols = list(num_cols)
    cat_cols = list(cat_cols)

    num_pipeline = Pipeline(
        steps=[
            ("quantile_clipper", QuantileClipper()),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("top_n_encoder", TopNCategoryEncoder(top_n=top_n_categories)),
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
    train_df["multi_" + label_col] = label_encoder.fit_transform(train_df[label_col])
    val_df["multi_" + label_col] = label_encoder.transform(val_df[label_col])
    test_df["multi_" + label_col] = label_encoder.transform(test_df[label_col])

    if benign_tag is not None:
        train_df["bin_" + label_col] = (train_df[label_col] != benign_tag).astype(int)
        val_df["bin_" + label_col] = (val_df[label_col] != benign_tag).astype(int)
        test_df["bin_" + label_col] = (test_df[label_col] != benign_tag).astype(int)

    mapping = {
        int(i): str(class_name) for i, class_name in enumerate(label_encoder.classes_)
    }
    logger.info(f"Label mapping: {mapping}")

    mapping_path = Path(cfg.path.processed_data) / "label_mapping.json"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    logger.info(f"Label mapping saved to {mapping_path}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )

    logging.info("Loading raw data...")
    df = load_df(cfg.path.raw_data + "/" + cfg.data.file_name + ".csv")

    logging.info("Preprocessing data...")
    train_df, val_df, test_df = preprocess_df(df, cfg)

    logger.info("Saving processed data...")
    Path(cfg.path.processed_data).mkdir(parents=True, exist_ok=True)

    save_df(
        train_df,
        cfg.path.processed_data
        + "/"
        + cfg.data.file_name
        + "_train."
        + cfg.data.extension,
    )
    save_df(
        val_df,
        cfg.path.processed_data
        + "/"
        + cfg.data.file_name
        + "_val."
        + cfg.data.extension,
    )
    save_df(
        test_df,
        cfg.path.processed_data
        + "/"
        + cfg.data.file_name
        + "_test."
        + cfg.data.extension,
    )
