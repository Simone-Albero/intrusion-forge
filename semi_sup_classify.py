from typing import Optional, Tuple, List
from pathlib import Path
import logging
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Average
from ignite.handlers.tensorboard_logger import *
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support

from src.data.io import load_df
from src.data.preprocessing import subsample_df

from src.common.config import load_config
from src.common.logging import setup_logger

from src.torch.dataset import MixedTabularDataset
from src.torch.batch import default_collate
from src.torch.model import ModelFactory
from src.torch.loss import LossFactory
from src.torch.engine import train_step, eval_step, test_step, filter_output

from src.ml.projection import tsne_projection, subsample_data_and_labels

from src.plot.array import confusion_matrix_to_plot, vectors_plot
from src.plot.dict import dict_to_bar_plot, dict_to_table

setup_logger()
logger = logging.getLogger(__name__)


def prepare_loader(
    cfg,
    is_unsupervised: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    logger.info("Preparing data for PyTorch...")

    base_path = Path(cfg.path.processed_data)
    file_base = f"{cfg.data.file_name}"
    ext = cfg.data.extension

    train_df = load_df(base_path / f"{file_base}_train.{ext}")
    train_df = (
        train_df[train_df[cfg.data.label_col] != cfg.data.benign_tag]
        if is_unsupervised
        else train_df
    )

    train_df = (
        subsample_df(
            train_df,
            n_samples=cfg.n_samples,
            random_state=cfg.seed,
            label_col=cfg.data.label_col,
        )
        if cfg.n_samples is not None and not is_unsupervised
        else train_df
    )

    val_df = load_df(base_path / f"{file_base}_val.{ext}")
    val_df = (
        val_df[val_df[cfg.data.label_col] != cfg.data.benign_tag]
        if is_unsupervised
        else val_df
    )
    test_df = load_df(base_path / f"{file_base}_test.{ext}")

    num_cols = list(cfg.data.num_cols)
    cat_cols = list(cfg.data.cat_cols)

    train_dataset = MixedTabularDataset(
        train_df,
        num_cols=num_cols,
        cat_cols=cat_cols,
        label_col=cfg.data.label_col if not is_unsupervised else None,
    )
    val_dataset = MixedTabularDataset(
        val_df,
        num_cols=num_cols,
        cat_cols=cat_cols,
        label_col=cfg.data.label_col if not is_unsupervised else None,
    )
    test_dataset = MixedTabularDataset(
        test_df,
        num_cols=num_cols,
        cat_cols=cat_cols,
        label_col=cfg.data.label_col if not is_unsupervised else None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.loops.training.dataloader.batch_size,
        shuffle=True,
        num_workers=cfg.loops.training.dataloader.num_workers,
        pin_memory=cfg.loops.training.dataloader.pin_memory,
        collate_fn=default_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.loops.validation.dataloader.batch_size,
        shuffle=False,
        num_workers=cfg.loops.validation.dataloader.num_workers,
        pin_memory=cfg.loops.validation.dataloader.pin_memory,
        collate_fn=default_collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.loops.test.dataloader.batch_size,
        shuffle=False,
        num_workers=cfg.loops.test.dataloader.num_workers,
        pin_memory=cfg.loops.test.dataloader.pin_memory,
        collate_fn=default_collate,
    )

    logger.info("Data preparation for PyTorch completed.")
    return train_loader, val_loader, test_loader


def create_model_and_loss(
    model_name, model_params, loss_name, loss_params, device
) -> Tuple[nn.Module, nn.Module]:
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
    max_grad_norm: float = 1.0,
    max_epochs: int = 50,
    checkpoint_dir: Path = Path("./checkpoints"),
) -> None:
    tb_logger = TensorboardLogger(log_dir=cfg.path.logs)

    trainer = Engine(train_step)
    trainer.state.model = model
    trainer.state.loss_fn = loss_fn
    trainer.state.optimizer = optimizer
    trainer.state.scheduler = scheduler
    trainer.state.device = device
    trainer.state.max_grad_norm = max_grad_norm

    # Attach metrics to the trainer
    Average(output_transform=lambda x: x["loss"]).attach(trainer, "loss")

    validator = Engine(eval_step)
    validator.state.model = model
    validator.state.loss_fn = loss_fn
    validator.state.device = device

    # Attach metrics to the validator
    Average(output_transform=lambda x: x["loss"]).attach(validator, "loss")

    early_stopping_handler = EarlyStopping(
        patience=cfg.loops.training.early_stopping.patience,
        min_delta=cfg.loops.training.early_stopping.min_delta,
        score_function=lambda engine: -engine.state.metrics["loss"],
        trainer=trainer,
    )
    validator.add_event_handler(Events.COMPLETED, early_stopping_handler)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_handler = ModelCheckpoint(
        dirname=checkpoint_dir,
        filename_prefix="best",
        score_function=lambda engine: -engine.state.metrics["loss"],
        score_name="val_loss",
        n_saved=1,
        global_step_transform=lambda engine, event_name: trainer.state.epoch,
        require_empty=False,
    )
    validator.add_event_handler(Events.COMPLETED, best_model_handler, {"model": model})

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_eval(engine):
        train_loss = engine.state.metrics["loss"]
        logger.info(f"Epoch [{engine.state.epoch}] Train Loss: {train_loss:.6f}")
        validator.run(val_loader)
        val_loss = validator.state.metrics["loss"]
        logger.info(f"Epoch [{engine.state.epoch}] Val Loss: {val_loss:.6f}")

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="training",
        output_transform=lambda x: {"loss": x["loss"], "grad_norm": x["grad_norm"]},
    )

    tb_logger.attach_output_handler(
        validator,
        event_name=Events.COMPLETED,
        tag="validation",
        metric_names=["loss"],
        global_step_transform=global_step_from_engine(trainer),
    )

    tb_logger.attach(
        trainer,
        log_handler=OptimizerParamsHandler(optimizer, param_name="lr"),
        event_name=Events.ITERATION_COMPLETED,
    )

    trainer.run(train_loader, max_epochs=max_epochs)


if __name__ == "__main__":
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )

    train_loader, val_loader, test_loader = prepare_loader(cfg, is_unsupervised=True)
    autoencoder, loss_fn = create_model_and_loss(
        cfg.pretraining.model.name,
        cfg.pretraining.model.params,
        cfg.pretraining.loss.name,
        cfg.pretraining.loss.params,
        cfg.device,
    )
    optimizer = create_optimizer(cfg, autoencoder, loss_fn)
    scheduler = create_scheduler(cfg, optimizer, train_loader)

    # Training
    train(
        train_loader,
        val_loader,
        autoencoder,
        loss_fn,
        optimizer,
        scheduler,
        device=torch.device(cfg.device),
        checkpoint_dir=Path(cfg.path.models) / "autoencoder",
        max_epochs=cfg.loops.training.epochs,
    )

    # Load best model for fine-tuning
    best_model_path = Path(cfg.path.models) / "autoencoder"
    checkpoint_files = list(best_model_path.glob("best_model_*.pt"))
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Loading best model from {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=cfg.device)
        autoencoder.load_state_dict(checkpoint)

    train_loader, val_loader, test_loader = prepare_loader(cfg, is_unsupervised=False)
    model, loss_fn = create_model_and_loss(
        cfg.finetuning.model.name,
        cfg.finetuning.model.params,
        cfg.finetuning.loss.name,
        cfg.finetuning.loss.params,
        torch.device(cfg.device),
    )

    model.encoder_module = autoencoder.encoder_module

    train(
        train_loader,
        val_loader,
        model,
        loss_fn,
        optimizer,
        scheduler,
        device=torch.device(cfg.device),
        checkpoint_dir=Path(cfg.path.models) / "classifier",
        max_epochs=cfg.loops.training.epochs,
    )
