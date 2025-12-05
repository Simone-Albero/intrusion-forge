from typing import Optional, Tuple, List
from pathlib import Path
import logging
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ignite.engine import Events
from ignite.metrics import Accuracy, Precision, Recall, ConfusionMatrix, Average
from ignite.handlers.tensorboard_logger import TensorboardLogger

from src.common.config import load_config
from src.common.logging import setup_logger

from src.data.io import load_data_splits
from src.data.preprocessing import subsample_df

from src.plot.array import vectors_plot
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
from src.ignite.metrics import F1Score

from src.ml.projection import tsne_projection, create_subsample_mask

from src.plot.array import confusion_matrix_to_plot, vectors_plot
from src.plot.dict import dict_to_bar_plot, dict_to_table
from src.ml.latent_quality import (
    trustworthiness,
    continuity,
    local_distance_consistency,
    reconstruction_latent_correlation,
)
from src.ml.clustering import (
    kmeans_grid_search,
    hdbscan_grid_search,
    compute_cluster_quality_measures,
)

setup_logger()
logger = logging.getLogger(__name__)

# Constants
VISUALIZATION_SAMPLES = 3000


def prepare_loader(
    cfg,
    is_unsupervised: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Prepare train, validation, and test data loaders for semi-supervised learning."""
    logger.info("Preparing data for PyTorch...")

    base_path = Path(cfg.path.processed_data)
    train_df, val_df, test_df = load_data_splits(
        base_path, cfg.data.file_name, cfg.data.extension
    )

    if is_unsupervised:
        train_df = train_df[train_df[cfg.data.label_col] != cfg.data.benign_tag]
        val_df = val_df[val_df[cfg.data.label_col] != cfg.data.benign_tag]
    else:
        # Subsample training data for fine-tuning if specified
        if cfg.n_samples is not None:
            train_df = subsample_df(
                train_df,
                n_samples=cfg.n_samples,
                random_state=cfg.seed,
                label_col=cfg.data.label_col,
            )

    num_cols = list(cfg.data.num_cols)
    cat_cols = list(cfg.data.cat_cols)
    label_col = cfg.data.label_col if not is_unsupervised else None

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
    test_loader: Optional[DataLoader] = None,
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

    if test_loader is not None:
        tester = (
            EngineBuilder(test_step)
            .with_state(model=model, device=device, loss_fn=loss_fn)
            .build()
        )

    all_inputs = []
    all_z = []
    all_recons_err = []
    all_labels = []

    @tester.on(Events.COMPLETED)
    def collect_latents(engine):
        output = engine.state.output
        all_inputs.append(output["input"])
        all_z.append(output["output"]["z"].detach().cpu().numpy())
        all_recons_err.append(output["loss"].detach().cpu().numpy())
        all_labels.append(output["y_true"].detach().cpu().numpy())

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_eval(engine):
        train_loss = engine.state.metrics["loss"]
        logger.info(f"Epoch [{engine.state.epoch}] Train Loss: {train_loss:.6f}")
        validator.run(val_loader)
        val_loss = validator.state.metrics["loss"]
        logger.info(f"Epoch [{engine.state.epoch}] Val Loss: {val_loss:.6f}")
        if test_loader is not None:
            tester.run(test_loader)

            inputs_array = np.vstack(all_inputs)
            z_array = np.vstack(all_z)
            recons_err_array = np.array(all_recons_err)
            labels_array = np.concatenate(all_labels)

            mask = create_subsample_mask(
                z_array,
                labels_array,
                n_samples=min(VISUALIZATION_SAMPLES, len(labels_array)),
                stratify=True,
            )

            subsampled_inputs = inputs_array[mask]
            subsampled_z = z_array[mask]
            subsampled_recons_err = recons_err_array[mask]
            subsampled_labels = labels_array[mask]

            projected_z = tsne_projection(subsampled_z)

            latent_figure = vectors_plot(projected_z, subsampled_labels)
            tb_logger.writer.add_figure(
                "test/pretraining/latent_space_2d", latent_figure, engine.state.epoch
            )

            # Compute latent quality metrics
            trust = trustworthiness(subsampled_inputs, subsampled_z)
            cont = continuity(subsampled_inputs, subsampled_z)
            ldc = local_distance_consistency(subsampled_inputs, subsampled_z)
            rlc = reconstruction_latent_correlation(subsampled_z, subsampled_recons_err)

            tb_logger.writer.add_scalar(
                "test/pretraining/trustworthiness", trust, engine.state.epoch
            )
            tb_logger.writer.add_scalar(
                "test/pretraining/continuity", cont, engine.state.epoch
            )
            tb_logger.writer.add_scalar(
                "test/pretraining/consistency",
                ldc,
                engine.state.epoch,
            )
            tb_logger.writer.add_scalar(
                "test/pretraining/reconstruction_correlation",
                rlc,
                engine.state.epoch,
            )

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
) -> None:
    """Test the model and log results to TensorBoard."""
    logger.info("Running test evaluation...")

    tb_logger = TensorboardLogger(log_dir=log_dir)

    # Storage for latent representations and labels
    all_z = []
    all_labels = []

    # Setup output transform for metrics
    prepare_output = lambda x: (x["output"]["logits"], x["y_true"])

    tester = (
        EngineBuilder(test_step)
        .with_state(model=model, device=device)
        .with_metric("accuracy", Accuracy(output_transform=prepare_output))
        .with_metric(
            "precision_macro", Precision(average=True, output_transform=prepare_output)
        )
        .with_metric(
            "recall_macro", Recall(average=True, output_transform=prepare_output)
        )
        .with_metric(
            "precision_per_class",
            Precision(average=False, output_transform=prepare_output),
        )
        .with_metric(
            "recall_per_class", Recall(average=False, output_transform=prepare_output)
        )
        .with_metric(
            "f1_macro",
            F1Score(average="macro", output_transform=prepare_output),
        )
        .with_metric(
            "f1_per_class",
            F1Score(average=None, output_transform=prepare_output),
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
        logger.info("Test Results:")
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
        f1_per_class = metrics["f1_per_class"]
        confusion_matrix = metrics["confusion_matrix"]

        global_metrics = {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
        }
        f1_per_class_dict = {f"f1_{i}": f1.item() for i, f1 in enumerate(f1_per_class)}

        z_array = np.vstack(all_z)
        labels_array = np.concatenate(all_labels)

        mask = create_subsample_mask(
            z_array,
            labels_array,
            n_samples=min(VISUALIZATION_SAMPLES, len(labels_array)),
            stratify=True,
        )
        subsampled_z = z_array[mask]
        subsampled_labels = labels_array[mask]

        logging.info("Computing kmeans clustering...")
        kmeans_model, _ = kmeans_grid_search(subsampled_z)

        logging.info("Computing HDBSCAN clustering...")
        hdbscan_model, _ = hdbscan_grid_search(subsampled_z)

        kmeans_labels = kmeans_model.labels_
        hdbscan_labels = hdbscan_model.labels_

        kmeans_cluster_measures = compute_cluster_quality_measures(
            subsampled_z, kmeans_labels
        )
        hdbscan_cluster_measures = compute_cluster_quality_measures(
            subsampled_z, hdbscan_labels
        )

        tb_logger.writer.add_figure(
            "test/kmeans_cluster_measures",
            dict_to_table(kmeans_cluster_measures),
            0,
        )

        tb_logger.writer.add_figure(
            "test/hdbscan_cluster_measures",
            dict_to_table(hdbscan_cluster_measures),
            0,
        )

        projected_z = tsne_projection(subsampled_z)

        latent_gt = vectors_plot(projected_z, subsampled_labels)
        latent_kmeans = vectors_plot(projected_z, kmeans_labels)
        latent_hdbscan = vectors_plot(projected_z, hdbscan_labels)

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
        tb_logger.writer.add_figure("test/latent_space_2d/ground_truth", latent_gt, 0)
        tb_logger.writer.add_figure("test/latent_space_2d/kmeans", latent_kmeans, 0)
        tb_logger.writer.add_figure("test/latent_space_2d/hdbscan", latent_hdbscan, 0)

        logger.info("Test results logged to TensorBoard.")

    try:
        tester.run(test_loader)
    finally:
        tb_logger.close()


def main():
    """Main training pipeline for semi-supervised learning (pretraining + fine-tuning)."""
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )

    device = torch.device(cfg.device)
    logger.info(f"Using device: {device}")

    # Phase 1: Unsupervised pretraining on malicious samples
    logger.info("Starting unsupervised pretraining phase...")
    train_loader, val_loader, test_loader = prepare_loader(cfg, is_unsupervised=True)

    autoencoder = create_model(
        cfg.pretraining.model.name, cfg.pretraining.model.params, device
    )
    pretrain_loss_fn = create_loss(
        cfg.pretraining.loss.name, cfg.pretraining.loss.params, device
    )

    pretrain_optimizer = create_optimizer(
        cfg.optimizer.name, cfg.optimizer.params, autoencoder, pretrain_loss_fn
    )
    pretrain_scheduler = create_scheduler(
        cfg.scheduler.name, cfg.scheduler.params, pretrain_optimizer, train_loader
    )

    pretrain_checkpoint_dir = Path(cfg.path.models) / "autoencoder"
    train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model=autoencoder,
        loss_fn=pretrain_loss_fn,
        optimizer=pretrain_optimizer,
        scheduler=pretrain_scheduler,
        device=device,
        patience=cfg.loops.training.early_stopping.patience,
        min_delta=cfg.loops.training.early_stopping.min_delta,
        log_dir=Path(cfg.path.logs) / "pretraining",
        checkpoint_dir=pretrain_checkpoint_dir,
        max_epochs=cfg.loops.training.epochs,
    )

    # Load best pretrained autoencoder
    load_best_checkpoint(pretrain_checkpoint_dir, autoencoder, device)

    # Phase 2: Supervised fine-tuning on labeled data
    logger.info("Starting supervised fine-tuning phase...")
    train_loader, val_loader, test_loader = prepare_loader(cfg, is_unsupervised=False)

    classifier = create_model(
        cfg.finetuning.model.name, cfg.finetuning.model.params, device
    )
    finetune_loss_fn = create_loss(
        cfg.finetuning.loss.name, cfg.finetuning.loss.params, device
    )

    # Transfer pretrained encoder to classifier
    classifier.encoder_module = autoencoder.encoder_module
    logger.info("Transferred pretrained encoder to classifier")

    finetune_optimizer = create_optimizer(
        cfg.optimizer.name, cfg.optimizer.params, classifier, finetune_loss_fn
    )
    finetune_scheduler = create_scheduler(
        cfg.scheduler.name, cfg.scheduler.params, finetune_optimizer, train_loader
    )

    finetune_checkpoint_dir = Path(cfg.path.models) / "classifier"
    train(
        train_loader=train_loader,
        val_loader=val_loader,
        model=classifier,
        loss_fn=finetune_loss_fn,
        optimizer=finetune_optimizer,
        scheduler=finetune_scheduler,
        device=device,
        patience=cfg.loops.training.early_stopping.patience,
        min_delta=cfg.loops.training.early_stopping.min_delta,
        log_dir=Path(cfg.path.logs) / "finetuning",
        checkpoint_dir=finetune_checkpoint_dir,
        max_epochs=cfg.loops.training.epochs,
    )

    logger.info("Semi-supervised training completed successfully")

    load_best_checkpoint(finetune_checkpoint_dir, classifier, device)
    test(
        test_loader=test_loader,
        model=classifier,
        device=device,
        log_dir=Path(cfg.path.logs),
        num_classes=cfg.model.params.num_classes,
    )


if __name__ == "__main__":
    main()
