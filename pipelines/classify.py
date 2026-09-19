import logging
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pipelines import paths_from_cfg
from pipelines.classify_evaluation import publish_evaluation
from pipelines.classify_training import (
    ClassifyContext,
    build_splits,
    build_trainer,
    train_splits,
)
from src.core.config import load_config, save_config
from src.core.io import load_listed_dfs
from src.core.log import (
    FilesystemFigureSubscriber,
    JSONSubscriber,
    LogDispatcher,
    PickleSubscriber,
    setup_logger,
)
from src.core.utils import flush_timing, load_from_json, timed
from src.domain.data.preprocessing import random_undersample_df, subsample_df
from src.domain.plot.base import set_figure_format
from src.domain.plot.style import apply_plot_style

setup_logger(log_file="resources/logs.txt")
apply_plot_style()
logger = logging.getLogger(__name__)


def _seed_everything(seed: int) -> None:
    """Seed the random, numpy and torch generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class DataConfig:
    """Shared data parameters across stages."""

    processed_data_path: Path
    extension: str
    label_col: str
    n_samples: int | None
    balance: str = "undersample"


def _load_data(
    data: DataConfig, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test splits, balancing and subsampling the training set only."""
    train_df, val_df, test_df = load_listed_dfs(
        data.processed_data_path,
        [
            f"train.{data.extension}",
            f"val.{data.extension}",
            f"test.{data.extension}",
        ],
    )
    if data.balance == "undersample":
        train_df = random_undersample_df(
            train_df, data.label_col, random_state=random_state
        )
    if data.n_samples is not None:
        train_df = subsample_df(
            train_df,
            data.n_samples,
            random_state=random_state,
            label_col=data.label_col,
        )
    return train_df, val_df, test_df


@timed
def classify(cfg) -> None:
    """Run the supervised classification pipeline for a single classifier."""
    if cfg.balance not in ("undersample", "none"):
        raise ValueError(
            f"Unknown balance: {cfg.balance!r}. Valid: 'undersample', 'none'."
        )

    _seed_everything(cfg.seed)
    set_figure_format(cfg.figure_format)
    paths = paths_from_cfg(cfg)

    df_meta_path = paths.shared / "metadata/df_meta.json"
    if not df_meta_path.exists():
        raise FileNotFoundError(f"Missing {df_meta_path}. Run `make prepare` first.")
    df_meta = load_from_json(df_meta_path)
    save_config(cfg, paths.configs / "config_composed.json")

    num_cols = list(cfg.data.num_cols) if cfg.data.num_cols else []
    cat_cols = list(cfg.data.cat_cols) if cfg.data.cat_cols else []
    label_col = "encoded_" + cfg.data.label_col
    feat_cols = num_cols + cat_cols

    data = DataConfig(
        processed_data_path=paths.processed_data,
        extension=cfg.data.extension,
        label_col=label_col,
        n_samples=cfg.n_samples,
        balance=cfg.balance,
    )

    use_kfold = cfg.kfold
    load_cfg = replace(data, balance="none", n_samples=None) if use_kfold else data
    train_df, val_df, test_df = _load_data(load_cfg, cfg.seed)
    logger.info(
        "Data loaded — train: %d, val: %d, test: %d samples",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    logger.info("Classifier: %s (kind=%s)", cfg.classifier.name, cfg.classifier.kind)

    bus = LogDispatcher()
    bus.subscribe(JSONSubscriber(paths.outputs))
    bus.subscribe(PickleSubscriber(paths.pickle))
    bus.subscribe(FilesystemFigureSubscriber(paths.figures))

    context = ClassifyContext(
        cfg=cfg,
        paths=paths,
        trainer=build_trainer(cfg, df_meta, num_cols, cat_cols, label_col),
        feat_cols=feat_cols,
        label_col=label_col,
        df_meta=df_meta,
        bus=bus,
    )
    plan = build_splits(cfg, paths, train_df, test_df, label_col)
    predictions = train_splits(context, plan, val_df)
    publish_evaluation(context, plan, predictions)

    logger.info("All stages completed.")


def main() -> None:
    """Entry point for the supervised classification stage."""
    cfg = load_config(
        config_path=Path(__file__).parent.parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )
    classify(cfg)
    flush_timing(Path(cfg.path.outputs) / "timing.json")


if __name__ == "__main__":
    main()
