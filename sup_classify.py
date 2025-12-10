from typing import Optional, Tuple, List
from pathlib import Path
import logging
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ignite.engine import Events
from ignite.metrics import Accuracy, ConfusionMatrix, Average
from ignite.handlers.tensorboard_logger import TensorboardLogger

from src.common.config import load_config
from src.common.logging import setup_logger

from src.data.io import load_data_splits
from src.data.preprocessing import subsample_df, equalize_classes

from src.torch.module.checkpoint import load_latest_checkpoint
from src.torch.engine import train_step, eval_step, test_step, filter_ignored_classes
from src.torch.builders import (
    create_dataloader,
    create_dataset,
    create_model,
    create_loss,
    create_optimizer,
    create_scheduler,
)

from src.ignite.builders import EngineBuilder
from src.ignite.metrics import F1, Precision, Recall

from src.ml.projection import tsne_projection, create_subsample_mask

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
    train_df = equalize_classes(
        train_df,
        label_col=f"multi_{cfg.data.label_col}",
        random_state=cfg.seed,
    )

    if cfg.n_samples is not None:
        train_df = subsample_df(
            train_df,
            n_samples=cfg.n_samples,
            random_state=cfg.seed,
            label_col=cfg.data.label_col,
        )

    num_cols = list(cfg.data.num_cols)
    cat_cols = list(cfg.data.cat_cols)
    label_col = f"multi_{cfg.data.label_col}"

    train_dataset = create_dataset(train_df, num_cols, cat_cols, label_col)
    val_dataset = create_dataset(val_df, num_cols, cat_cols, label_col)
    test_dataset = create_dataset(test_df, num_cols, cat_cols, label_col)

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

    logger.info("Data preparation for PyTorch completed.")
    return train_loader, val_loader, test_loader


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
        .with_weights_logging(tb_logger=tb_logger, model=model)
        .with_gradients_logging(tb_logger=tb_logger, model=model)
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


def test(
    test_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    log_dir: Path,
    num_classes: int,
    ignore_classes: Optional[List[int]] = None,
    run_id: int = 0,
) -> None:
    """Test the model and log results to TensorBoard."""
    logger.info(f"Running test evaluation (run_id={run_id})...")

    tb_logger = TensorboardLogger(log_dir=log_dir)

    # Storage for latent representations and labels
    all_z = []
    all_labels = []

    # Setup output transform for metrics
    if ignore_classes is not None:
        prepare_output = lambda x: filter_ignored_classes(
            x["output"]["logits"], x["y_true"], ignore_classes
        )
    else:
        prepare_output = lambda x: (x["output"]["logits"], x["y_true"])

    tester = (
        EngineBuilder(test_step)
        .with_state(model=model, device=device)
        .with_metric("accuracy", Accuracy(output_transform=prepare_output))
        .with_metric(
            "precision_macro",
            Precision(average="macro", output_transform=prepare_output),
        )
        .with_metric(
            "recall_macro",
            Recall(average="macro", output_transform=prepare_output),
        )
        .with_metric(
            "precision_weighted",
            Precision(average="weighted", output_transform=prepare_output),
        )
        .with_metric(
            "recall_weighted",
            Recall(average="weighted", output_transform=prepare_output),
        )
        .with_metric(
            "precision_per_class",
            Precision(average=None, output_transform=prepare_output),
        )
        .with_metric(
            "recall_per_class",
            Recall(average=None, output_transform=prepare_output),
        )
        .with_metric(
            "f1_macro",
            F1(average="macro", output_transform=prepare_output),
        )
        .with_metric(
            "f1_weighted",
            F1(average="weighted", output_transform=prepare_output),
        )
        .with_metric(
            "f1_per_class",
            F1(average=None, output_transform=prepare_output),
        )
        .with_metric(
            "confusion_matrix",
            ConfusionMatrix(num_classes=num_classes, output_transform=prepare_output),
        )
        .build()
    )

    @tester.on(Events.ITERATION_COMPLETED)
    def collect_latents(engine):
        """Store latent representations and labels for visualization."""
        output = engine.state.output
        all_z.append(output["output"]["z"].detach().cpu().numpy())
        all_labels.append(output["y_true"].detach().cpu().numpy())

    @tester.on(Events.COMPLETED)
    def log_metrics_to_console(engine):
        """Log all metrics to console using a loop."""
        logger.info(f"Test Results (run_id={run_id}):")
        for metric_name, metric_value in engine.state.metrics.items():
            if isinstance(metric_value, torch.Tensor):
                if metric_value.numel() == 1:
                    logger.info(f"  {metric_name}: {metric_value.item():.4f}")
                else:
                    logger.info(f"  {metric_name}: {metric_value}")
            else:
                logger.info(f"  {metric_name}: {metric_value:.4f}")

    @tester.on(Events.COMPLETED)
    def log_metrics_to_tensorboard(engine):
        """Log visualizations and metrics to TensorBoard."""
        metrics = engine.state.metrics
        accuracy = metrics["accuracy"]
        precision_macro = metrics["precision_macro"]
        recall_macro = metrics["recall_macro"]
        f1_macro = metrics["f1_macro"]
        precision_weighted = metrics["precision_weighted"]
        recall_weighted = metrics["recall_weighted"]
        f1_weighted = metrics["f1_weighted"]
        f1_per_class = metrics["f1_per_class"]
        confusion_matrix = metrics["confusion_matrix"]

        global_metrics = {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
        }
        f1_per_class_dict = {f"f1_{i}": f1.item() for i, f1 in enumerate(f1_per_class)}

        z_array = np.vstack(all_z)
        labels_array = np.concatenate(all_labels)

        mask = create_subsample_mask(
            labels_array,
            n_samples=min(VISUALIZATION_SAMPLES, len(labels_array)),
            stratify=True,
        )
        subsampled_z = z_array[mask]
        subsampled_labels = labels_array[mask]
        projected_z = tsne_projection(subsampled_z)

        latent_figure = vectors_plot(projected_z, subsampled_labels)

        cm_figure = confusion_matrix_to_plot(
            cm=confusion_matrix.cpu().numpy(),
            title="Test Confusion Matrix",
            normalize="true",
        )

        # Log individual metrics as scalars for trend visualization
        tb_logger.writer.add_scalar("test/metrics/accuracy", accuracy, run_id)
        tb_logger.writer.add_scalar(
            "test/metrics/precision_macro", precision_macro, run_id
        )
        tb_logger.writer.add_scalar("test/metrics/recall_macro", recall_macro, run_id)
        tb_logger.writer.add_scalar("test/metrics/f1_macro", f1_macro, run_id)
        tb_logger.writer.add_scalar(
            "test/metrics/precision_weighted", precision_weighted, run_id
        )
        tb_logger.writer.add_scalar(
            "test/metrics/recall_weighted", recall_weighted, run_id
        )
        tb_logger.writer.add_scalar("test/metrics/f1_weighted", f1_weighted, run_id)

        tb_logger.writer.add_figure(f"test/confusion_matrix", cm_figure, run_id)
        tb_logger.writer.add_figure(
            f"test/metrics_summary",
            dict_to_bar_plot(global_metrics),
            run_id,
        )
        tb_logger.writer.add_figure(
            f"test/f1_per_class",
            dict_to_bar_plot(f1_per_class_dict),
            run_id,
        )
        tb_logger.writer.add_figure(f"test/latent_space", latent_figure, run_id)

        # Log text summary
        metrics_text = (
            f"Run ID: {run_id}\n"
            f"Accuracy: {accuracy:.4f}\n"
            f"Precision Macro: {precision_macro:.4f}\n"
            f"Recall Macro: {recall_macro:.4f}\n"
            f"F1 Macro: {f1_macro:.4f}\n"
            f"Precision Weighted: {precision_weighted:.4f}\n"
            f"Recall Weighted: {recall_weighted:.4f}\n"
            f"F1 Weighted: {f1_weighted:.4f}"
        )
        tb_logger.writer.add_text(f"test/run_summary", metrics_text, run_id)

        logger.info("Test results logged to TensorBoard.")

    try:
        tester.run(test_loader)
    finally:
        tb_logger.close()


def run_training(cfg, device):
    """Phase 1: Training the supervised classifier."""
    logger.info("TRAINING")

    train_loader, val_loader, test_loader = prepare_loader(cfg)

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

    logger.info("Training phase completed successfully")
    return model, checkpoint_dir


def run_testing(cfg, device):
    """Phase 2: Testing the trained classifier."""
    logger.info("TESTING")

    _, _, test_loader = prepare_loader(cfg)

    model = create_model(cfg.model.name, cfg.model.params, device)
    checkpoint_dir = Path(cfg.path.models)
    load_latest_checkpoint(checkpoint_dir, model, device)

    test(
        test_loader=test_loader,
        model=model,
        device=device,
        log_dir=Path(cfg.path.logs),
        num_classes=cfg.model.params.num_classes,
        ignore_classes=list(cfg.ignore_classes) if cfg.ignore_classes else None,
        run_id=cfg.get("run_id", 0),
    )

    logger.info("Testing phase completed successfully")


def main():
    """Main training pipeline for supervised learning.

    Supported stages (controlled by cfg.stage):
    - 'all': Run all stages (training → testing)
    - 'training': Run only training
    - 'testing': Run only testing
    """
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )

    device = torch.device(cfg.device)
    logger.info(f"Using device: {device}")

    # Get the stage to run from config (default to 'all')
    stage = cfg.get("stage", "all")
    logger.info(f"Running stage: {stage}")

    # Execute the appropriate pipeline based on the stage
    if stage == "all":
        run_training(cfg, device)
        run_testing(cfg, device)

    elif stage == "training":
        run_training(cfg, device)

    elif stage == "testing":
        run_testing(cfg, device)

    else:
        logger.error(f"Unknown stage: {stage}")
        logger.info("Valid stages are: 'all', 'training', 'testing'")
        sys.exit(1)


if __name__ == "__main__":
    main()
