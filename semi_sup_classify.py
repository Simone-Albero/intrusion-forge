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
from ignite.handlers.tensorboard_logger import TensorboardLogger

from src.data.io import load_data_splits
from src.data.preprocessing import subsample_df

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


def create_model_and_loss(
    model_name: str,
    model_params: dict,
    loss_name: str,
    loss_params: dict,
    device: torch.device,
) -> Tuple[nn.Module, nn.Module]:
    """Create model and loss function."""
    logger.info("Creating model and loss function...")

    model = ModelFactory.create(model_name, model_params).to(device)
    loss_fn = LossFactory.create(loss_name, loss_params).to(device)
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

        # Plot latent space visualization for each epoch
        # Compute and log clustering metrics over the latent space (sparsity, separability, etc.)

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

    autoencoder, pretrain_loss_fn = create_model_and_loss(
        cfg.pretraining.model.name,
        cfg.pretraining.model.params,
        cfg.pretraining.loss.name,
        cfg.pretraining.loss.params,
        device,
    )

    pretrain_optimizer = create_optimizer(cfg, autoencoder, pretrain_loss_fn)
    pretrain_scheduler = create_scheduler(cfg, pretrain_optimizer, train_loader)

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
        log_dir=Path(cfg.path.logs),
        checkpoint_dir=pretrain_checkpoint_dir,
        max_epochs=cfg.loops.training.epochs,
    )

    # Load best pretrained autoencoder
    load_best_checkpoint(pretrain_checkpoint_dir, autoencoder, device)

    # Phase 2: Supervised fine-tuning on labeled data
    logger.info("Starting supervised fine-tuning phase...")
    train_loader, val_loader, test_loader = prepare_loader(cfg, is_unsupervised=False)

    classifier, finetune_loss_fn = create_model_and_loss(
        cfg.finetuning.model.name,
        cfg.finetuning.model.params,
        cfg.finetuning.loss.name,
        cfg.finetuning.loss.params,
        device,
    )

    # Transfer pretrained encoder to classifier
    classifier.encoder_module = autoencoder.encoder_module
    logger.info("Transferred pretrained encoder to classifier")

    finetune_optimizer = create_optimizer(cfg, classifier, finetune_loss_fn)
    finetune_scheduler = create_scheduler(cfg, finetune_optimizer, train_loader)

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
        log_dir=Path(cfg.path.logs),
        checkpoint_dir=finetune_checkpoint_dir,
        max_epochs=cfg.loops.training.epochs,
    )

    logger.info("Semi-supervised training completed successfully")


if __name__ == "__main__":
    main()
