from typing import Optional, Tuple
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
from src.ignite.metrics import F1, Precision, Recall

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
    label_col = "multi_" + cfg.data.label_col if not is_unsupervised else None

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

    all_inputs = []
    all_z = []
    all_recons_err = []

    if test_loader is not None:
        tester = (
            EngineBuilder(test_step)
            .with_state(model=model, device=device, loss_fn=loss_fn)
            .build()
        )

        @tester.on(Events.ITERATION_COMPLETED)
        def collect_latents(engine):
            output = engine.state.output
            all_inputs.append(output["input"].detach().cpu().numpy())
            all_z.append(output["output"]["z"].detach().cpu().numpy())
            all_recons_err.append(output["loss"].detach().cpu().numpy())

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_eval(engine):
        train_loss = engine.state.metrics["loss"]
        logger.info(f"Epoch [{engine.state.epoch}] Train Loss: {train_loss:.6f}")
        validator.run(val_loader)
        val_loss = validator.state.metrics["loss"]
        logger.info(f"Epoch [{engine.state.epoch}] Val Loss: {val_loss:.6f}")
        if test_loader is not None:
            all_inputs.clear()
            all_z.clear()
            all_recons_err.clear()

            tester.state.loss_fn.reduction = "none"
            tester.run(test_loader)

            inputs_array = np.vstack(all_inputs)
            z_array = np.vstack(all_z)
            recons_err_array = np.concatenate(all_recons_err)

            mask = create_subsample_mask(
                recons_err_array,
                n_samples=min(VISUALIZATION_SAMPLES, len(recons_err_array)),
                stratify=False,
            )

            subsampled_inputs = inputs_array[mask]
            subsampled_z = z_array[mask]
            subsampled_recons_err = recons_err_array[mask]

            # Compute latent quality metrics
            logging.info("Computing latent quality metrics...")
            trust = trustworthiness(subsampled_inputs, subsampled_z)
            cont = continuity(subsampled_inputs, subsampled_z)
            ldc = local_distance_consistency(subsampled_inputs, subsampled_z)
            rlc = reconstruction_latent_correlation(subsampled_z, subsampled_recons_err)

            tb_logger.writer.add_scalar(
                "pretraining/trustworthiness", trust, engine.state.epoch
            )
            tb_logger.writer.add_scalar(
                "pretraining/continuity", cont, engine.state.epoch
            )
            tb_logger.writer.add_scalar(
                "pretraining/consistency",
                ldc,
                engine.state.epoch,
            )
            tb_logger.writer.add_scalar(
                "pretraining/reconstruction_correlation",
                rlc,
                engine.state.epoch,
            )

            tester.state.loss_fn.reduction = "mean"

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
    run_id: int = 0,
) -> None:
    """Test the model and log results to TensorBoard."""
    logger.info(f"Running test evaluation (run_id={run_id})...")

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
        gt_cluster_measures = compute_cluster_quality_measures(
            subsampled_z, subsampled_labels
        )

        tb_logger.writer.add_figure(
            f"test/cluster_measures/kmeans/run_{run_id}",
            dict_to_table(kmeans_cluster_measures),
            run_id,
        )
        tb_logger.writer.add_figure(
            f"test/cluster_measures/hdbscan/run_{run_id}",
            dict_to_table(hdbscan_cluster_measures),
            run_id,
        )
        tb_logger.writer.add_figure(
            f"test/cluster_measures/ground_truth/run_{run_id}",
            dict_to_table(gt_cluster_measures),
            run_id,
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
        tb_logger.writer.add_figure(
            f"test/latent_space/ground_truth", latent_gt, run_id
        )
        tb_logger.writer.add_figure(f"test/latent_space/kmeans", latent_kmeans, run_id)
        tb_logger.writer.add_figure(
            f"test/latent_space/hdbscan", latent_hdbscan, run_id
        )

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


def run_pretraining(cfg, device):
    """Phase 1: Unsupervised pretraining on malicious samples."""
    logger.info("UNSUPERVISED PRETRAINING")

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

    logger.info("Pretraining phase completed successfully")
    return autoencoder, pretrain_checkpoint_dir


def run_finetuning(cfg, device, autoencoder=None):
    """Phase 2: Supervised fine-tuning on labeled data."""
    logger.info("SUPERVISED FINE-TUNING")

    train_loader, val_loader, test_loader = prepare_loader(cfg, is_unsupervised=False)

    classifier = create_model(
        cfg.finetuning.model.name, cfg.finetuning.model.params, device
    )
    finetune_loss_fn = create_loss(
        cfg.finetuning.loss.name, cfg.finetuning.loss.params, device
    )

    # Transfer pretrained encoder to classifier if provided
    if autoencoder is not None:
        classifier.encoder_module = autoencoder.encoder_module
        logger.info("Transferred pretrained encoder to classifier")
    else:
        pretrain_checkpoint_dir = Path(cfg.path.models) / "autoencoder"
        if pretrain_checkpoint_dir.exists():
            temp_autoencoder = create_model(
                cfg.pretraining.model.name, cfg.pretraining.model.params, device
            )
            load_best_checkpoint(pretrain_checkpoint_dir, temp_autoencoder, device)
            classifier.encoder_module = temp_autoencoder.encoder_module
            logger.info("Loaded and transferred pretrained encoder from checkpoint")
        else:
            logger.warning("No pretrained encoder found. Training from scratch.")

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

    logger.info("Fine-tuning phase completed successfully")
    return classifier, finetune_checkpoint_dir


def run_testing(cfg, device):
    """Phase 3: Testing the fine-tuned classifier."""
    logger.info("TESTING")

    _, _, test_loader = prepare_loader(cfg, is_unsupervised=False)

    classifier = create_model(
        cfg.finetuning.model.name, cfg.finetuning.model.params, device
    )

    finetune_checkpoint_dir = Path(cfg.path.models) / "classifier"
    load_best_checkpoint(finetune_checkpoint_dir, classifier, device)

    test(
        test_loader=test_loader,
        model=classifier,
        device=device,
        log_dir=Path(cfg.path.logs) / "finetuning",
        num_classes=cfg.finetuning.model.params.num_classes,
        run_id=cfg.get("run_id", 0),
    )

    logger.info("Testing phase completed successfully")


def main():
    """Main training pipeline for semi-supervised learning.

    Supported stages (controlled by cfg.stage):
    - 'all': Run all stages (pretraining → fine-tuning → testing)
    - 'pretraining': Run only unsupervised pretraining
    - 'finetuning': Run only supervised fine-tuning (loads pretrained encoder if available)
    - 'testing': Run only testing (loads fine-tuned model)
    - 'pretrain_finetune': Run pretraining and fine-tuning (skip testing)
    - 'finetune_test': Run fine-tuning and testing (skip pretraining)
    """
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )

    device = torch.device(cfg.device)
    logger.info(f"Using device: {device}")

    stage = cfg.get("stage", "all")
    logger.info(f"Running stage: {stage}")

    # Execute the appropriate pipeline based on the stage
    if stage == "all":
        autoencoder, _ = run_pretraining(cfg, device)
        classifier, _ = run_finetuning(cfg, device, autoencoder=autoencoder)
        run_testing(cfg, device)

    elif stage == "pretraining":
        run_pretraining(cfg, device)

    elif stage == "finetuning":
        run_finetuning(cfg, device, autoencoder=None)

    elif stage == "testing":
        run_testing(cfg, device)

    elif stage == "pretrain_finetune":
        autoencoder, _ = run_pretraining(cfg, device)
        run_finetuning(cfg, device, autoencoder=autoencoder)

    elif stage == "finetune_test":
        run_finetuning(cfg, device, autoencoder=None)
        run_testing(cfg, device)

    else:
        logger.error(f"Unknown stage: {stage}")
        logger.info(
            "Valid stages are: 'all', 'pretraining', 'finetuning', 'testing', "
            "'pretrain_finetune', 'finetune_test'"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
