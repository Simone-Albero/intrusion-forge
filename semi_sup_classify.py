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

from src.common.config import load_config
from src.common.logging import setup_logger

from src.data.io import load_data_splits
from src.data.preprocessing import subsample_df

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

        # plot the latent space (only for unsupervised pretraining)
        # compute latent space sparsity (only for unsupervised pretraining)

    try:
        trainer.run(train_loader, max_epochs=max_epochs)
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


if __name__ == "__main__":
    main()
