import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from ignite.handlers.tensorboard_logger import TensorboardLogger
from sklearn.metrics import confusion_matrix

from src.common.config import load_config, save_config
from src.common.log import (
    JSONSubscriber,
    LogBundle,
    LogDispatcher,
    PickleSubscriber,
    TensorBoardSubscriber,
    setup_logger,
)
from src.common.paths import OutputPaths
from src.common.utils import (
    flush_timing,
    load_from_joblib,
    load_from_json,
    save_to_joblib,
    timed,
)
from src.data.io import load_listed_dfs
from src.data.preprocessing import subsample_df
from src.ml.evaluation import compute_classification_metrics, evaluate_predictions
from src.ml.figures import build_test_figures
from src.ml.training import fit_classifier, grid_search_classifier, predict_with_proba
from src.plot.style import apply_plot_style

setup_logger(log_file="resources/logs.txt")
apply_plot_style()
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Shared data parameters for train() and evaluate()."""

    processed_data_path: Path
    extension: str
    num_cols: list[str]
    cat_cols: list[str]
    label_col: str
    n_samples: int | None


def load_data(
    data: DataConfig, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test splits; optionally subsample the training set."""
    train_df, val_df, test_df = load_listed_dfs(
        data.processed_data_path,
        [
            f"train.{data.extension}",
            f"val.{data.extension}",
            f"test.{data.extension}",
        ],
    )
    if data.n_samples is not None:
        train_df = subsample_df(train_df, data.n_samples, random_state, data.label_col)
    return train_df, val_df, test_df


@timed
def train(
    classifier_cfg,
    grid_search_cfg,
    train_df: pd.DataFrame,
    feat_cols: list[str],
    label_col: str,
    paths: OutputPaths,
    bus: LogDispatcher,
) -> None:
    """Fit (optionally via grid search) and persist the estimator."""
    X = train_df[feat_cols].to_numpy()
    y = train_df[label_col].to_numpy()

    has_grid = "grid" in classifier_cfg and len(classifier_cfg.grid) > 0
    if grid_search_cfg.enabled and has_grid:
        logger.info(
            "Grid search for %s — scoring=%s, cv=%d",
            classifier_cfg.name,
            grid_search_cfg.scoring,
            grid_search_cfg.cv,
        )
        model, summary = grid_search_classifier(
            name=classifier_cfg.name,
            params=dict(classifier_cfg.params),
            grid=dict(classifier_cfg.grid),
            X=X,
            y=y,
            scoring=grid_search_cfg.scoring,
            cv=grid_search_cfg.cv,
        )
        logger.info(
            "Best params: %s | Best score (%s): %.4f",
            summary["best_params"],
            summary["scoring"],
            summary["best_score"],
        )
        bus.publish(LogBundle.from_dict({"json/training/grid_search": summary}))
    else:
        logger.info("Training %s ...", classifier_cfg.name)
        model = fit_classifier(
            name=classifier_cfg.name,
            params=dict(classifier_cfg.params),
            X=X,
            y=y,
        )

    save_to_joblib(model, paths.models / "model.joblib")
    logger.info("Model saved to %s", paths.models / "model.joblib")


@timed
def evaluate(
    model,
    test_df: pd.DataFrame,
    feat_cols: list[str],
    label_col: str,
    df_meta: dict,
) -> dict:
    """Evaluate model on test set; build metrics, prediction infos, figures."""
    X = test_df[feat_cols].to_numpy()
    y_true = test_df[label_col].to_numpy()
    clusters = (
        test_df["cluster"].to_numpy() if "cluster" in test_df.columns else None
    )

    y_pred, y_proba = predict_with_proba(model, X)

    scalars, full_metrics = compute_classification_metrics(y_true, y_pred)
    pred_infos = evaluate_predictions(y_true, y_pred, y_proba, clusters)
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true), normalize="true")
    figures = build_test_figures(X, y_true, y_pred, df_meta["label_mapping"])

    return {
        "pred_infos": pred_infos,
        "scalars": scalars,
        "figures": figures,
        "full_metrics": full_metrics,
        "confusion_matrix": cm,
    }


@timed
def ml_classify(cfg) -> None:
    """Run a single-classifier ML pipeline (training and/or evaluation)."""
    paths = OutputPaths(
        processed_data=Path(cfg.path.processed_data),
        data_logs=Path(cfg.path.data_logs),
        tb_logs=Path(cfg.path.tb_logs),
        configs=Path(cfg.path.configs),
        json_logs=Path(cfg.path.json_logs),
        pickle=Path(cfg.path.pickle),
        models=Path(cfg.path.models),
    )
    df_meta = load_from_json(paths.data_logs / "data/df_meta.json")
    save_config(cfg, paths.configs / "config_composed.json")

    num_cols = list(cfg.data.num_cols) if cfg.data.num_cols else []
    cat_cols = list(cfg.data.cat_cols) if cfg.data.cat_cols else []
    label_col = "encoded_" + cfg.data.label_col
    feat_cols = num_cols + cat_cols

    data = DataConfig(
        processed_data_path=paths.processed_data,
        extension=cfg.data.extension,
        num_cols=num_cols,
        cat_cols=cat_cols,
        label_col=label_col,
        n_samples=cfg.n_samples,
    )

    stage = cfg.stage
    if stage not in ("all", "training", "testing", "inference"):
        logger.error(
            "Unknown stage: %r. Valid: 'all', 'training', 'testing', 'inference'.",
            stage,
        )
        sys.exit(1)

    train_df, val_df, test_df = load_data(data, cfg.seed)
    logger.info(
        "Data loaded — train: %d, val: %d, test: %d samples",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    logger.info("Classifier: %s", cfg.classifier.name)

    train_tb_logger = None
    eval_tb_logger = None
    try:
        if stage in ("training", "all"):
            train_tb_logger = TensorboardLogger(log_dir=paths.tb_logs / "training")
            train_bus = LogDispatcher()
            train_bus.subscribe(JSONSubscriber(paths.json_logs))
            train_bus.subscribe(TensorBoardSubscriber(train_tb_logger.writer))
            train(
                classifier_cfg=cfg.classifier,
                grid_search_cfg=cfg.grid_search,
                train_df=train_df,
                feat_cols=feat_cols,
                label_col=label_col,
                paths=paths,
                bus=train_bus,
            )

        if stage in ("testing", "inference", "all"):
            logger.info("Loading model from %s", paths.models / "model.joblib")
            model = load_from_joblib(paths.models / "model.joblib")

            eval_tb_logger = TensorboardLogger(log_dir=paths.tb_logs / "testing")
            eval_bus = LogDispatcher()
            eval_bus.subscribe(JSONSubscriber(paths.json_logs))
            eval_bus.subscribe(TensorBoardSubscriber(eval_tb_logger.writer))
            eval_bus.subscribe(PickleSubscriber(paths.pickle))

            result = evaluate(
                model=model,
                test_df=test_df,
                feat_cols=feat_cols,
                label_col=label_col,
                df_meta=df_meta,
            )
            eval_bus.publish(
                LogBundle.from_dict(
                    {
                        **result["scalars"],
                        **result["figures"],
                        "json/testing/summary": result["full_metrics"],
                        "json/analysis/predictions/test": result["pred_infos"],
                        "pickle/analysis/confusion_matrices/test": result[
                            "confusion_matrix"
                        ],
                    }
                )
            )
    finally:
        if train_tb_logger is not None:
            train_tb_logger.close()
        if eval_tb_logger is not None:
            eval_tb_logger.close()

    logger.info("All stages completed.")


def main():
    """Main entry point for ML classification."""
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="ml_config",
        overrides=sys.argv[1:],
    )
    ml_classify(cfg)
    flush_timing(Path(cfg.path.json_logs) / "timing.json")


if __name__ == "__main__":
    main()
