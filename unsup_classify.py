from typing import Optional, Tuple
from pathlib import Path
import logging
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ignite.engine import Engine, Events
from ignite.handlers.tensorboard_logger import TensorboardLogger
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support

from src.data.io import load_data_splits

from src.common.config import load_config
from src.common.logging import setup_logger

from src.torch.data.loaders import create_dataset, create_dataloader
from src.torch.module.checkpoint import load_best_checkpoint
from src.torch.model import ModelFactory
from src.torch.loss import LossFactory
from src.torch.engine import test_step

from src.ignite.setup import (
    setup_trainer,
    setup_validator,
    attach_early_stopping_and_checkpointing,
    attach_tensorboard_logging,
)

from src.ml.projection import tsne_projection, subsample_data_and_labels

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

    # Create datasets
    train_dataset = create_dataset(train_df, num_cols, cat_cols)
    val_dataset = create_dataset(val_df, num_cols, cat_cols)
    test_dataset = create_dataset(test_df, num_cols, cat_cols)

    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        cfg.loops.training.dataloader.batch_size,
        shuffle=True,
        num_workers=cfg.loops.training.dataloader.num_workers,
        pin_memory=cfg.loops.training.dataloader.pin_memory,
    )
    val_loader = create_dataloader(
        val_dataset,
        cfg.loops.validation.dataloader.batch_size,
        shuffle=False,
        num_workers=cfg.loops.validation.dataloader.num_workers,
        pin_memory=cfg.loops.validation.dataloader.pin_memory,
    )
    test_loader = create_dataloader(
        test_dataset,
        cfg.loops.test.dataloader.batch_size,
        shuffle=False,
        num_workers=cfg.loops.test.dataloader.num_workers,
        pin_memory=cfg.loops.test.dataloader.pin_memory,
    )

    # Extract test labels
    bin_test_labels = test_df[f"bin_{cfg.data.label_col}"].values
    multi_test_labels = test_df[f"multi_{cfg.data.label_col}"].values

    logger.info("Data preparation for PyTorch completed.")
    return train_loader, val_loader, test_loader, bin_test_labels, multi_test_labels


def create_model_and_loss(cfg) -> Tuple[nn.Module, nn.Module]:
    logger.info("Creating model and loss function...")

    model = ModelFactory.create(cfg.model.name, cfg.model.params).to(cfg.device)
    loss_fn = LossFactory.create(cfg.loss.name, cfg.loss.params).to(cfg.device)

    return model, loss_fn


def create_optimizer(
    cfg, model: nn.Module, loss_fn: Optional[nn.Module] = None
) -> torch.optim.Optimizer:
    logger.info("Creating optimizer...")

    params = model.parameters()
    if loss_fn is not None and len(list(loss_fn.parameters())) > 0:
        params = list(model.parameters()) + list(loss_fn.parameters())

    optimizer = torch.optim.__dict__[cfg.optimizer.name](params, **cfg.optimizer.params)

    return optimizer


def create_scheduler(
    cfg, optimizer: torch.optim.Optimizer, dataloader: DataLoader
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    logger.info("Creating learning rate scheduler...")

    if cfg.scheduler is None:
        return None

    if cfg.scheduler.params.steps_per_epoch == "auto":
        cfg.scheduler.params.steps_per_epoch = len(dataloader)

    scheduler = torch.optim.lr_scheduler.__dict__[cfg.scheduler.name](
        optimizer, **cfg.scheduler.params
    )

    return scheduler


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

    trainer = setup_trainer(model, loss_fn, optimizer, device, scheduler, max_grad_norm)
    validator = setup_validator(model, loss_fn, device)
    attach_early_stopping_and_checkpointing(
        trainer, validator, model, patience, min_delta, checkpoint_dir
    )
    attach_tensorboard_logging(
        trainer, validator, model, optimizer, tb_logger, log_weights=False
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


def _setup_tester(model: nn.Module, device: torch.device, loss_fn: nn.Module) -> Engine:
    """Setup and configure the test engine."""
    tester = Engine(test_step)
    tester.state.model = model
    tester.state.device = device
    loss_fn.reduction = "none"
    tester.state.loss_fn = loss_fn
    return tester


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


def _log_test_metrics(
    precision: float, recall: float, f1: float, roc_auc: float
) -> None:
    """Log test metrics to console."""
    logger.info("Test Results:")
    logger.info(f"Test Precision (Macro): {precision:.4f}")
    logger.info(f"Test Recall (Macro): {recall:.4f}")
    logger.info(f"Test F1 Score (Macro): {f1:.4f}")
    logger.info(f"Test AUC-ROC: {roc_auc:.4f}")


def _prepare_visualization(
    all_z: list, multi_labels: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Prepare latent space visualization using t-SNE."""
    try:
        z_array = np.vstack(all_z)
        subsampled_z, subsampled_labels = subsample_data_and_labels(
            z_array,
            multi_labels,
            n_samples=min(VISUALIZATION_SAMPLES, len(multi_labels)),
            stratify=True,
        )
        projected_z = tsne_projection(subsampled_z)
        return projected_z, subsampled_labels
    except Exception as e:
        logger.warning(f"Error during visualization: {e}")
        return None, None


def _log_to_tensorboard(
    tb_logger: TensorboardLogger,
    precision: float,
    recall: float,
    f1: float,
    roc_auc: float,
    projected_z: Optional[np.ndarray],
    subsampled_labels: Optional[np.ndarray],
    global_step: int,
) -> None:
    """Log metrics and visualizations to TensorBoard."""
    global_metrics = {
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "auc_roc": roc_auc,
    }

    tb_logger.writer.add_figure(
        "test/global_metrics", dict_to_bar_plot(global_metrics), global_step
    )
    tb_logger.writer.add_scalar("test/auc_roc", roc_auc, global_step)
    tb_logger.writer.add_scalar("test/f1_macro", f1, global_step)

    if projected_z is not None and subsampled_labels is not None:
        latent_figure = vectors_plot(projected_z, subsampled_labels)
        tb_logger.writer.add_figure("test/latent_space_2d", latent_figure, global_step)


def test(
    test_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    log_dir: Path,
    loss_fn: nn.Module,
    bin_labels: np.ndarray,
    multi_labels: np.ndarray,
    global_step: int = 0,
) -> None:
    """Test the model for anomaly detection and log results to TensorBoard."""
    tb_logger = TensorboardLogger(log_dir=log_dir)
    tester = _setup_tester(model, device, loss_fn)

    all_z = []
    all_preds = []

    @tester.on(Events.ITERATION_COMPLETED)
    def store_outputs(engine):
        output = engine.state.output
        all_z.append(output["output"]["z"].detach().cpu().numpy())
        all_preds.append(output["loss"].detach().cpu().numpy())

    @tester.on(Events.COMPLETED)
    def log_results(engine):
        predictions = np.concatenate(all_preds)
        precision, recall, f1, roc_auc, _ = _compute_anomaly_metrics(
            bin_labels, predictions
        )

        _log_test_metrics(precision, recall, f1, roc_auc)

        projected_z, subsampled_labels = _prepare_visualization(all_z, multi_labels)

        _log_to_tensorboard(
            tb_logger,
            precision,
            recall,
            f1,
            roc_auc,
            projected_z,
            subsampled_labels,
            global_step,
        )

        logger.info("Test results logged to TensorBoard.")

    logger.info("Running test evaluation...")
    tester.run(test_loader)
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

    # Prepare data
    train_loader, val_loader, test_loader, bin_test_labels, multi_test_labels = (
        prepare_loader(cfg)
    )

    # Setup model and training components
    model, loss_fn = create_model_and_loss(cfg)
    optimizer = create_optimizer(cfg, model, loss_fn)
    scheduler = create_scheduler(cfg, optimizer, train_loader)

    # Training
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

    # Load best model and test
    load_best_checkpoint(checkpoint_dir, model, device)
    test(
        test_loader=test_loader,
        model=model,
        device=device,
        log_dir=Path(cfg.path.logs),
        loss_fn=loss_fn,
        bin_labels=bin_test_labels,
        multi_labels=multi_test_labels,
        global_step=cfg.global_step,
    )


if __name__ == "__main__":
    main()
