from typing import Optional, Tuple
from pathlib import Path
import logging
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Precision, Recall, ConfusionMatrix
from ignite.handlers.tensorboard_logger import TensorboardLogger

from src.common.config import load_config
from src.common.logging import setup_logger

from src.data.io import load_data_splits

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

from src.plot.array import confusion_matrix_to_plot, vectors_plot
from src.plot.dict import dict_to_bar_plot

setup_logger()
logger = logging.getLogger(__name__)

# Constants
EPSILON = 1e-8
VISUALIZATION_SAMPLES = 3000


def prepare_loader(cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Prepare train, validation, and test data loaders."""
    logger.info("Preparing data for PyTorch...")

    base_path = Path(cfg.path.processed_data)
    train_df, val_df, test_df = load_data_splits(
        base_path, cfg.data.file_name, cfg.data.extension
    )
    test_df = test_df[
        test_df[cfg.data.label_col] != cfg.data.benign_tag
    ]  # Ever exclude benign samples from test set

    num_cols = list(cfg.data.num_cols)
    cat_cols = list(cfg.data.cat_cols)
    label_col = f"multi_{cfg.data.label_col}"

    # Create datasets
    train_dataset = create_dataset(train_df, num_cols, cat_cols, label_col)
    val_dataset = create_dataset(val_df, num_cols, cat_cols, label_col)
    test_dataset = create_dataset(test_df, num_cols, cat_cols, label_col)

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

    logger.info("Data preparation for PyTorch completed.")
    return train_loader, val_loader, test_loader


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

    trainer = setup_trainer(model, loss_fn, optimizer, scheduler, device, max_grad_norm)
    validator = setup_validator(model, loss_fn, device)
    attach_early_stopping_and_checkpointing(
        trainer, validator, model, patience, min_delta, checkpoint_dir
    )
    attach_tensorboard_logging(
        trainer, validator, model, optimizer, tb_logger, log_weights=True
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


def _setup_tester(model: nn.Module, device: torch.device, num_classes: int) -> Engine:
    """Setup and configure the test engine with metrics."""
    tester = Engine(test_step)
    tester.state.model = model
    tester.state.device = device

    prepare_output = lambda x: (x["output"]["logits"], x["y_true"])

    Accuracy(output_transform=prepare_output).attach(tester, "accuracy")

    Precision(average=True, output_transform=prepare_output).attach(
        tester, "precision_macro"
    )
    Recall(average=True, output_transform=prepare_output).attach(tester, "recall_macro")

    Precision(average=False, output_transform=prepare_output).attach(
        tester, "precision_per_class"
    )
    Recall(average=False, output_transform=prepare_output).attach(
        tester, "recall_per_class"
    )

    ConfusionMatrix(num_classes=num_classes, output_transform=prepare_output).attach(
        tester, "confusion_matrix"
    )

    return tester


def _compute_f1_scores(precision, recall):
    """Compute F1 scores from precision and recall."""
    return (2 * precision * recall) / (precision + recall + EPSILON)


def _log_test_metrics(metrics: dict) -> Tuple:
    """Log test metrics and compute F1 scores."""
    accuracy = metrics["accuracy"]
    precision_macro = metrics["precision_macro"]
    recall_macro = metrics["recall_macro"]
    precision_per_class = metrics["precision_per_class"]
    recall_per_class = metrics["recall_per_class"]
    confusion_matrix = metrics["confusion_matrix"]

    f1_macro = _compute_f1_scores(precision_macro, recall_macro)
    f1_per_class = _compute_f1_scores(precision_per_class, recall_per_class)

    logger.info("Test Results:")
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test Precision (Macro): {precision_macro:.4f}")
    logger.info(f"Test Recall (Macro): {recall_macro:.4f}")
    logger.info(f"Test F1 Score (Macro): {f1_macro:.4f}")
    logger.info(f"Test Precision (Per Class): {precision_per_class}")
    logger.info(f"Test Recall (Per Class): {recall_per_class}")
    logger.info(f"Test F1 Score (Per Class): {f1_per_class}")
    logger.info(f"Test Confusion Matrix:\n{confusion_matrix}")

    return (
        accuracy,
        precision_macro,
        recall_macro,
        f1_macro,
        f1_per_class,
        confusion_matrix,
    )


def _prepare_visualization(
    all_z: list, all_labels: list
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Prepare latent space visualization using t-SNE."""
    try:
        z_array = np.vstack(all_z)
        labels_array = np.concatenate(all_labels)

        subsampled_z, subsampled_labels = subsample_data_and_labels(
            z_array,
            labels_array,
            n_samples=min(VISUALIZATION_SAMPLES, len(labels_array)),
            stratify=True,
        )
        projected_z = tsne_projection(subsampled_z)
        return projected_z, subsampled_labels
    except Exception as e:
        logger.warning(f"Error during visualization: {e}")
        return None, None


def _log_to_tensorboard(
    tb_logger: TensorboardLogger,
    metrics: dict,
    f1_macro: float,
    f1_per_class,
    projected_z: Optional[np.ndarray],
    subsampled_labels: Optional[np.ndarray],
) -> None:
    """Log metrics and visualizations to TensorBoard."""
    accuracy = metrics["accuracy"]
    precision_macro = metrics["precision_macro"]
    recall_macro = metrics["recall_macro"]
    confusion_matrix = metrics["confusion_matrix"]

    global_metrics = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
    }

    f1_per_class_dict = {f"f1_{i}": f1.item() for i, f1 in enumerate(f1_per_class)}

    # Plot confusion matrix and metrics
    cm_figure = confusion_matrix_to_plot(
        cm=confusion_matrix.cpu().numpy(),
        title="Test Confusion Matrix",
        normalize="true",
    )
    tb_logger.writer.add_figure("test/confusion_matrix", cm_figure, 0)
    tb_logger.writer.add_figure(
        "test/global_metrics", dict_to_bar_plot(global_metrics), 0
    )
    tb_logger.writer.add_figure(
        "test/f1_per_class", dict_to_bar_plot(f1_per_class_dict), 0
    )

    # Plot latent space if available
    if projected_z is not None and subsampled_labels is not None:
        latent_figure = vectors_plot(projected_z, subsampled_labels)
        tb_logger.writer.add_figure("test/latent_space_2d", latent_figure, 0)


def test(
    test_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    log_dir: Path,
    num_classes: int,
) -> None:
    """Test the model and log results to TensorBoard."""
    tb_logger = TensorboardLogger(log_dir=log_dir)
    tester = _setup_tester(model, device, num_classes)

    all_z = []
    all_labels = []

    @tester.on(Events.ITERATION_COMPLETED)
    def store_outputs(engine):
        output = engine.state.output
        all_z.append(output["output"]["z"].detach().cpu().numpy())
        all_labels.append(output["y_true"].detach().cpu().numpy())

    @tester.on(Events.COMPLETED)
    def log_results(engine):
        (
            accuracy,
            precision_macro,
            recall_macro,
            f1_macro,
            f1_per_class,
            confusion_matrix,
        ) = _log_test_metrics(engine.state.metrics)

        projected_z, subsampled_labels = _prepare_visualization(all_z, all_labels)

        _log_to_tensorboard(
            tb_logger,
            engine.state.metrics,
            f1_macro,
            f1_per_class,
            projected_z,
            subsampled_labels,
        )

        logger.info("Test results logged to TensorBoard.")

    logger.info("Running test evaluation...")
    tester.run(test_loader)
    tb_logger.close()


def main():
    """Main training and testing pipeline."""
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )

    device = torch.device(cfg.device)
    logger.info(f"Using device: {device}")

    # Prepare data
    train_loader, val_loader, test_loader = prepare_loader(cfg)

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
        num_classes=cfg.model.params.num_classes,
    )


if __name__ == "__main__":
    main()
