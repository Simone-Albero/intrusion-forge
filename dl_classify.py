import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ignite.engine import Events
from ignite.handlers.tensorboard_logger import TensorboardLogger
from ignite.metrics import Average
from sklearn.metrics import confusion_matrix, f1_score

from src.common.config import load_config, save_config
from src.common.log import (
    LogBundle,
    LogDispatcher,
    JSONSubscriber,
    PickleSubscriber,
    TensorBoardSubscriber,
    setup_logger,
)
from src.common.paths import OutputPaths
from src.common.utils import flush_timing, load_from_json, timed

from src.data.io import load_listed_dfs
from src.data.preprocessing import subsample_df

from src.ml.evaluation import compute_classification_metrics, evaluate_predictions
from src.ml.figures import build_test_figures

from src.plot.style import apply_plot_style

from torch.utils.data import DataLoader

from src.torch.builders import (
    create_dataloader,
    create_dataset,
    create_loss,
    create_model,
    create_optimizer,
    create_scheduler,
)
from src.torch.engine import eval_step, train_step
from src.torch.infer import df_to_tensors, get_predictions
from src.torch.module.checkpoint import load_best_checkpoint

from src.ignite.builders import EngineBuilder

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


def _make_loader(df, num_cols, cat_cols, label_cols, dataloader_cfg):
    return create_dataloader(
        create_dataset(df, num_cols, cat_cols, label_cols), dataloader_cfg
    )


def _build_trainer(
    model, loss_fn, optimizer, scheduler, device, max_grad_norm, tb_logger
):
    """Configure the training engine with TensorBoard logging."""
    return (
        EngineBuilder(train_step)
        .with_state(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            max_grad_norm=max_grad_norm,
        )
        .with_metric("loss", Average(output_transform=lambda x: x["loss"]))
        .with_tensorboard(
            tb_logger=tb_logger,
            tag="train",
            output_transform=lambda x: {"loss": x["loss"], "grad_norm": x["grad_norm"]},
        )
        .with_optimizer_logging(tb_logger=tb_logger, optimizer=optimizer)
        .with_weights_logging(tb_logger=tb_logger, model=model)
        .with_gradients_logging(tb_logger=tb_logger, model=model)
        .build()
    )


def _build_validator(
    model,
    loss_fn,
    device,
    tb_logger,
    trainer,
    early_stopping_patience,
    early_stopping_min_delta,
    models_path,
):
    """Configure the validation engine with early stopping and checkpointing."""
    return (
        EngineBuilder(eval_step)
        .with_state(model=model, loss_fn=loss_fn, device=device)
        .with_metric("loss", Average(output_transform=lambda x: x["loss"]))
        .with_early_stopping(
            trainer=trainer,
            metric="loss",
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
        )
        .with_checkpointing(
            trainer=trainer,
            checkpoint_dir=models_path,
            objects_to_save={"model": model},
            metric="loss",
        )
        .with_tensorboard(
            tb_logger=tb_logger,
            event=Events.COMPLETED,
            tag="validation",
            metric_names=["loss"],
            trainer=trainer,
        )
        .build()
    )


@timed
def evaluate(
    model: nn.Module,
    inputs: list[torch.Tensor],
    y: torch.Tensor,
    X: np.ndarray,
    clusters: np.ndarray | None,
    label_mapping: dict,
    device: torch.device,
) -> dict:
    """Evaluate the model on the test set.

    Accepts pre-extracted tensors and arrays — conversion from df happens at the call site.
    Computes sklearn metrics via compute_classification_metrics.
    Calls evaluate_predictions for failure rates and cluster error rates.
    Calls build_test_figures for confusion matrix and projection figures.
    """
    y_true_t, y_pred_t, z_t, confidences_t = get_predictions(model, inputs, y, device)

    y_true = y_true_t.numpy()
    y_pred = y_pred_t.numpy()
    z = z_t.numpy() if z_t is not None else None
    confidences = confidences_t.numpy()

    scalars, full_metrics = compute_classification_metrics(y_true, y_pred)
    pred_infos = evaluate_predictions(y_true, y_pred, confidences, clusters)
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true), normalize="true")
    figures = build_test_figures(X, y_true, y_pred, label_mapping, z=z)

    return {
        "pred_infos": pred_infos,
        "scalars": scalars,
        "figures": figures,
        "full_metrics": full_metrics,
        "confusion_matrix": cm,
    }


@timed
def train(
    model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    paths: OutputPaths,
    max_epochs: int,
    max_grad_norm: float,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    device: torch.device,
) -> None:
    """Train the supervised classifier in-place.

    TensorBoard per-step logging (loss/step, grad norms, weight histograms) is
    handled internally by EngineBuilder.
    """
    logger.info("Starting training phase...")

    log_dir = paths.tb_logs / "training"
    log_dir.mkdir(parents=True, exist_ok=True)
    tb_logger = TensorboardLogger(log_dir=log_dir)

    trainer = _build_trainer(
        model, loss_fn, optimizer, scheduler, device, max_grad_norm, tb_logger
    )
    validator = _build_validator(
        model,
        loss_fn,
        device,
        tb_logger,
        trainer,
        early_stopping_patience,
        early_stopping_min_delta,
        paths.models,
    )

    epoch_start: list[float] = []

    @trainer.on(Events.EPOCH_STARTED)
    def record_epoch_start(_engine):
        epoch_start.clear()
        epoch_start.append(time.perf_counter())

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation(engine):
        epoch_duration = time.perf_counter() - epoch_start[0]
        tb_logger.writer.add_scalar(
            "train/epoch_duration_s", epoch_duration, engine.state.epoch
        )
        logger.info(
            "Epoch [%d] Train Loss: %.6f | Duration: %.2fs",
            engine.state.epoch,
            engine.state.metrics["loss"],
            epoch_duration,
        )
        validator.run(val_loader)
        logger.info(
            "Epoch [%d] Val Loss: %.6f",
            engine.state.epoch,
            validator.state.metrics["loss"],
        )

    try:
        trainer.run(train_loader, max_epochs=max_epochs)
    finally:
        tb_logger.close()

    logger.info("Training completed.")


@timed
def classify(cfg) -> None:
    """Run supervised classification pipeline (training and/or evaluation)."""
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
    cfg.model.params.num_classes = df_meta["num_classes"]
    cfg.loss.params.class_weight = df_meta["class_weights"]

    save_config(cfg, Path(cfg.path.configs) / "config_composed.json")

    device = torch.device(cfg.device)
    logger.info("Using device: %s", device)

    num_cols = list(cfg.data.num_cols) if cfg.data.num_cols else []
    cat_cols = list(cfg.data.cat_cols) if cfg.data.cat_cols else []
    label_col = "encoded_" + cfg.data.label_col
    feat_cols = num_cols + cat_cols

    data = DataConfig(
        processed_data_path=Path(cfg.path.processed_data),
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

    model = create_model(cfg.model.name, cfg.model.params, device)
    loss_fn = create_loss(cfg.loss.name, cfg.loss.params, device)
    train_loader = _make_loader(
        train_df,
        data.num_cols,
        data.cat_cols,
        [data.label_col],
        cfg.loops.training.dataloader,
    )
    val_loader = _make_loader(
        val_df,
        data.num_cols,
        data.cat_cols,
        [data.label_col],
        cfg.loops.validation.dataloader,
    )

    if stage in ("testing", "inference"):
        logger.info("Loading best checkpoint from %s ...", paths.models)
        load_best_checkpoint(paths.models, model, device)

    if stage in ("training", "all"):
        logger.info("Starting training stage ...")
        optimizer = create_optimizer(
            cfg.optimizer.name, cfg.optimizer.params, model, loss_fn
        )
        scheduler = create_scheduler(
            cfg.scheduler.name, cfg.scheduler.params, optimizer, train_loader
        )
        train(
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            paths=paths,
            max_epochs=cfg.loops.training.epochs,
            max_grad_norm=cfg.loops.training.max_grad_norm,
            early_stopping_patience=cfg.loops.training.early_stopping.patience,
            early_stopping_min_delta=cfg.loops.training.early_stopping.min_delta,
            device=device,
        )
        load_best_checkpoint(paths.models, model, device)
        logger.info("Best checkpoint reloaded after training.")

    if stage in ("testing", "inference", "all"):
        logger.info("Starting evaluation stage ...")
        *inputs, y = df_to_tensors(
            test_df,
            [data.num_cols, data.cat_cols, [data.label_col]],
            [torch.float32, torch.long, torch.long],
        )
        X = test_df[feat_cols].to_numpy()
        clusters = (
            test_df["cluster"].to_numpy() if "cluster" in test_df.columns else None
        )

        eval_bus = LogDispatcher()
        tb_eval_logger = TensorboardLogger(log_dir=paths.tb_logs / "testing")
        eval_bus.subscribe(TensorBoardSubscriber(tb_eval_logger.writer))
        eval_bus.subscribe(JSONSubscriber(paths.json_logs))
        eval_bus.subscribe(PickleSubscriber(paths.pickle))
        try:
            result = evaluate(
                model,
                inputs,
                y,
                X,
                clusters,
                df_meta["label_mapping"],
                device,
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
            tb_eval_logger.close()

    logger.info("All stages completed.")


def main():
    """Main entry point for supervised classification."""
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )
    classify(cfg)
    flush_timing(Path(cfg.path.json_logs) / "timing.json")


if __name__ == "__main__":
    main()
