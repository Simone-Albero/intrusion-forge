import logging
import sys
from pathlib import Path

from src.common.config import load_config
from src.common.logging import setup_logger
from src.common.utils import save_to_json

from src.data.io import load_listed_dfs
from src.data.analyze import compute_class_separability

setup_logger()
logger = logging.getLogger(__name__)


def analyze(cfg):
    num_cols = list(cfg.data.num_cols) if cfg.data.num_cols else []
    cat_cols = list(cfg.data.cat_cols) if cfg.data.cat_cols else []
    feature_cols = num_cols + cat_cols

    logger.info("Loading data ...")
    train_df, val_df, test_df = load_listed_dfs(
        Path(cfg.path.processed_data),
        [
            f"train.{cfg.data.extension}",
            f"val.{cfg.data.extension}",
            f"test.{cfg.data.extension}",
        ],
    )

    separability_results = {}
    label_cols = (
        [cfg.data.label_col]
        if "cluster" not in train_df.columns
        else [cfg.data.label_col, "cluster"]
    )
    for split_name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        for label_col in label_cols:
            logger.info(f"Computing separability on {split_name} set ...")
            X = split_df[feature_cols].values
            y = split_df[label_col].values
            separability_result = compute_class_separability(X, y)
            save_to_json(
                separability_result,
                Path(cfg.path.json_logs)
                / f"data/separability/{label_col}_{split_name}.json",
            )
            if label_col not in separability_results:
                separability_results[label_col] = {}
            separability_results[label_col][split_name] = separability_result

    return separability_results


def main():
    """Main entry point for data analysis."""
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )
    analyze(cfg)


if __name__ == "__main__":
    main()
