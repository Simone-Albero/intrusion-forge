from typing import Optional, Tuple
from pathlib import Path
import logging
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ignite.engine import Events
from ignite.metrics import Average
from ignite.handlers.tensorboard_logger import TensorboardLogger
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support

from src.common.config import load_config
from src.common.logging import setup_logger

from src.data.io import load_data_splits

from src.torch.module.checkpoint import load_best_checkpoint
from src.torch.engine import train_step, eval_step, test_step
from src.torch.builders import (
    create_dataloader,
    create_dataset,
    create_model,
    create_loss,
    create_optimizer,
    create_scheduler,
)

from src.ignite.builders import EngineBuilder

from src.ml.projection import tsne_projection, create_subsample_mask

from src.plot.array import vectors_plot
from src.plot.dict import dict_to_bar_plot

setup_logger()
logger = logging.getLogger(__name__)

# Constants
VISUALIZATION_SAMPLES = 3000


def prepare_loader(
    cfg,
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """Prepare train, validation, and test data loaders for unsupervised learning."""
    logger.info("Preparing data for PyTorch...")

    base_path = Path(cfg.path.processed_data)
    train_df, val_df, test_df = load_data_splits(
        base_path, cfg.data.file_name, cfg.data.extension
    )

    benign_train_df = train_df[train_df[cfg.data.label_col] == cfg.data.benign_tag]

    if cfg.noise_fraction > 0:
        malicious_train_df = train_df[
            train_df[cfg.data.label_col] != cfg.data.benign_tag
        ]
        n_noise = int(len(benign_train_df) * cfg.noise_fraction)
        noise_train = malicious_train_df.sample(
            n=min(n_noise, len(malicious_train_df)), random_state=cfg.seed
        )
        train_df = (
            pd.concat([benign_train_df, noise_train])
            .sample(frac=1, random_state=cfg.seed)
            .reset_index(drop=True)
        )
    else:
        train_df = benign_train_df

    # Filter validation data to benign samples only
    val_df = val_df[val_df[cfg.data.label_col] == cfg.data.benign_tag]

    num_cols = list(cfg.data.num_cols)
    cat_cols = list(cfg.data.cat_cols)

    train_dataset = create_dataset(train_df, num_cols, cat_cols)
    val_dataset = create_dataset(val_df, num_cols, cat_cols)
    test_dataset = create_dataset(test_df, num_cols, cat_cols)

    train_loader = create_dataloader(
        train_dataset,
        cfg.loops.training.dataloader,
    )
    val_loader = create_dataloader(
        val_dataset,
        cfg.loops.validation.dataloader,
    )
    test_loader = create_dataloader(
        test_dataset,
        cfg.loops.test.dataloader,
    )

    # Extract test labels
    bin_test_labels = test_df[f"bin_{cfg.data.label_col}"].values
    multi_test_labels = test_df[f"multi_{cfg.data.label_col}"].values

    logger.info("Data preparation for PyTorch completed.")
    return train_loader, val_loader, test_loader, bin_test_labels, multi_test_labels


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    patience: int,
    min_delta: float,
    log_dir: Path,
    checkpoint_dir: Path,
    max_epochs: int = 50,
    max_grad_norm: float = 1.0,
) -> None:
    """Train the model with validation and checkpointing."""
    tb_logger = TensorboardLogger(log_dir=log_dir)

    trainer = (
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
            tag="training",
            output_transform=lambda x: {"loss": x["loss"], "grad_norm": x["grad_norm"]},
        )
        .with_optimizer_logging(tb_logger=tb_logger, optimizer=optimizer)
        .build()
    )

    validator = (
        EngineBuilder(eval_step)
        .with_state(model=model, loss_fn=loss_fn, device=device)
        .with_metric("loss", Average(output_transform=lambda x: x["loss"]))
        .with_early_stopping(
            trainer=trainer,
            metric="loss",
            patience=patience,
            min_delta=min_delta,
        )
        .with_checkpointing(
            trainer=trainer,
            checkpoint_dir=checkpoint_dir,
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

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_eval(engine):
        train_loss = engine.state.metrics["loss"]
        logger.info(f"Epoch [{engine.state.epoch}] Train Loss: {train_loss:.6f}")
        validator.run(val_loader)
        val_loss = validator.state.metrics["loss"]
        logger.info(f"Epoch [{engine.state.epoch}] Val Loss: {val_loss:.6f}")

    try:
        trainer.run(train_loader, max_epochs=max_epochs)
    finally:
        tb_logger.close()


def _compute_anomaly_metrics(
    bin_labels: np.ndarray, predictions: np.ndarray
) -> Tuple[float, float, float, float, float]:
    """Compute anomaly detection metrics including optimal threshold."""
    fpr, tpr, thresholds = roc_curve(bin_labels, predictions)
    roc_auc = auc(fpr, tpr)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    predictions_binary = (predictions >= optimal_threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        bin_labels,
        predictions_binary,
        average="macro",
        zero_division=0,
    )

    return precision, recall, f1, roc_auc, optimal_threshold


def test(
    test_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    log_dir: Path,
    loss_fn: nn.Module,
    bin_labels: np.ndarray,
    multi_labels: np.ndarray,
    noise_fraction: float = 0.0,
    run_id: int = 0,
) -> None:
    """Test the model for anomaly detection and log results to TensorBoard."""
    logger.info(
        f"Running test evaluation (noise_fraction={noise_fraction:.3f}, run_id={run_id})..."
    )

    # Use the same log directory for all runs to enable comparison
    tb_logger = TensorboardLogger(log_dir=log_dir)

    # Storage for latent representations and predictions
    all_z = []
    all_preds = []

    # Set loss reduction to none for per-sample anomaly scores
    loss_fn.reduction = "none"

    tester = (
        EngineBuilder(test_step)
        .with_state(model=model, loss_fn=loss_fn, device=device)
        .build()
    )

    @tester.on(Events.ITERATION_COMPLETED)
    def store_outputs(engine):
        """Store latent representations and predictions for analysis."""
        output = engine.state.output
        all_z.append(output["output"]["z"].detach().cpu().numpy())
        all_preds.append(output["loss"].detach().cpu().numpy())

    @tester.on(Events.COMPLETED)
    def log_metrics_to_console(engine):
        """Log anomaly detection metrics to console."""
        predictions = np.concatenate(all_preds)
        precision, recall, f1, roc_auc, optimal_threshold = _compute_anomaly_metrics(
            bin_labels, predictions
        )

        logger.info(
            f"Test Results (noise_fraction={noise_fraction:.3f}, run_id={run_id}):"
        )
        logger.info(f"  precision_macro: {precision:.4f}")
        logger.info(f"  recall_macro: {recall:.4f}")
        logger.info(f"  f1_macro: {f1:.4f}")
        logger.info(f"  auc_roc: {roc_auc:.4f}")
        logger.info(f"  optimal_threshold: {optimal_threshold:.4f}")

    @tester.on(Events.COMPLETED)
    def log_metrics_to_tensorboard(engine):
        """Log visualizations and metrics to TensorBoard."""
        predictions = np.concatenate(all_preds)
        precision, recall, f1, roc_auc, _ = _compute_anomaly_metrics(
            bin_labels, predictions
        )

        global_metrics = {
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "auc_roc": roc_auc,
        }

        z_array = np.vstack(all_z)
        mask = create_subsample_mask(
            multi_labels,
            n_samples=min(VISUALIZATION_SAMPLES, len(multi_labels)),
            stratify=True,
        )
        subsampled_z = z_array[mask]
        subsampled_labels = multi_labels[mask]
        projected_z = tsne_projection(subsampled_z)

        latent_figure = vectors_plot(projected_z, subsampled_labels)

        # Log individual metrics with organized tags
        tb_logger.writer.add_scalar("test/metrics/auc_roc", roc_auc, run_id)
        tb_logger.writer.add_scalar("test/metrics/f1_macro", f1, run_id)
        tb_logger.writer.add_scalar("test/metrics/precision_macro", precision, run_id)
        tb_logger.writer.add_scalar("test/metrics/recall_macro", recall, run_id)

        # Log noise_fraction as a scalar to track the experimental parameter
        tb_logger.writer.add_scalar(
            "test/experiment/noise_fraction", noise_fraction, run_id
        )

        # Log bar plot and latent space visualization
        tb_logger.writer.add_figure(
            f"test/metrics_summary/run_{run_id}",
            dict_to_bar_plot(global_metrics),
            run_id,
        )
        tb_logger.writer.add_figure(
            f"test/latent_space/run_{run_id}", latent_figure, run_id
        )

        # Log as text for easy reference
        metrics_text = (
            f"Run ID: {run_id}\n"
            f"Noise Fraction: {noise_fraction:.3f}\n"
            f"AUC-ROC: {roc_auc:.4f}\n"
            f"F1 Macro: {f1:.4f}\n"
            f"Precision Macro: {precision:.4f}\n"
            f"Recall Macro: {recall:.4f}"
        )
        tb_logger.writer.add_text(
            f"test/run_summary/run_{run_id}", metrics_text, run_id
        )

        logger.info("Test results logged to TensorBoard.")

    try:
        tester.run(test_loader)
    finally:
        tb_logger.close()


def main():
    """Main training and testing pipeline for unsupervised anomaly detection."""
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )

    device = torch.device(cfg.device)
    logger.info(f"Using device: {device}")

    train_loader, val_loader, test_loader, bin_test_labels, multi_test_labels = (
        prepare_loader(cfg)
    )

    model = create_model(cfg.model.name, cfg.model.params, device)
    loss_fn = create_loss(cfg.loss.name, cfg.loss.params, device)
    optimizer = create_optimizer(
        cfg.optimizer.name, cfg.optimizer.params, model, loss_fn
    )
    scheduler = create_scheduler(
        cfg.scheduler.name, cfg.scheduler.params, optimizer, train_loader
    )

    checkpoint_dir = Path(cfg.path.models)
    train(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        patience=cfg.loops.training.early_stopping.patience,
        min_delta=cfg.loops.training.early_stopping.min_delta,
        log_dir=Path(cfg.path.logs),
        checkpoint_dir=checkpoint_dir,
        max_epochs=cfg.loops.training.epochs,
    )

    load_best_checkpoint(checkpoint_dir, model, device)
    test(
        test_loader=test_loader,
        model=model,
        device=device,
        log_dir=Path(cfg.path.logs),
        loss_fn=loss_fn,
        bin_labels=bin_test_labels,
        multi_labels=multi_test_labels,
        noise_fraction=cfg.noise_fraction,
        run_id=cfg.get("run_id", 0),
    )


if __name__ == "__main__":
    main()
