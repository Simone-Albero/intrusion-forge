"""Ignite engine setup utilities."""

from typing import Optional
from pathlib import Path

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Average
from ignite.handlers.tensorboard_logger import (
    TensorboardLogger,
    global_step_from_engine,
    WeightsHistHandler,
    GradsHistHandler,
    OptimizerParamsHandler,
)

from src.torch.engine import train_step, eval_step


def setup_trainer(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    device,
    scheduler: Optional[_LRScheduler] = None,
    max_grad_norm: float = 1.0,
) -> Engine:
    """Setup and configure the training engine.

    Args:
        model: Neural network model
        loss_fn: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        device: Device to train on
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        Configured training engine
    """
    trainer = Engine(train_step)
    trainer.state.model = model
    trainer.state.loss_fn = loss_fn
    trainer.state.optimizer = optimizer
    trainer.state.scheduler = scheduler
    trainer.state.device = device
    trainer.state.max_grad_norm = max_grad_norm

    Average(output_transform=lambda x: x["loss"]).attach(trainer, "loss")
    return trainer


def setup_validator(
    model: nn.Module,
    loss_fn: nn.Module,
    device,
) -> Engine:
    """Setup and configure the validation engine.

    Args:
        model: Neural network model
        loss_fn: Loss function
        device: Device to validate on

    Returns:
        Configured validation engine
    """
    validator = Engine(eval_step)
    validator.state.model = model
    validator.state.loss_fn = loss_fn
    validator.state.device = device

    Average(output_transform=lambda x: x["loss"]).attach(validator, "loss")
    return validator


def attach_early_stopping_and_checkpointing(
    trainer: Engine,
    validator: Engine,
    model: nn.Module,
    patience: int,
    min_delta: float,
    checkpoint_dir: Path,
) -> None:
    """Attach early stopping and checkpointing handlers.

    Args:
        trainer: Training engine
        validator: Validation engine
        model: Model to checkpoint
        patience: Number of epochs to wait before early stopping
        min_delta: Minimum change in score to qualify as improvement
        checkpoint_dir: Directory to save checkpoints
    """
    early_stopping_handler = EarlyStopping(
        patience=patience,
        min_delta=min_delta,
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


def attach_tensorboard_logging(
    trainer: Engine,
    validator: Engine,
    model: nn.Module,
    optimizer: Optimizer,
    tb_logger: TensorboardLogger,
    log_weights: bool = False,
) -> None:
    """Attach TensorBoard logging handlers.

    Args:
        trainer: Training engine
        validator: Validation engine
        model: Model to log
        optimizer: Optimizer to log
        tb_logger: TensorBoard logger
        log_weights: Whether to log weight and gradient histograms
    """
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

    if log_weights:
        tb_logger.attach(
            trainer,
            log_handler=WeightsHistHandler(model),
            event_name=Events.EPOCH_COMPLETED,
        )

        tb_logger.attach(
            trainer,
            log_handler=GradsHistHandler(model),
            event_name=Events.EPOCH_COMPLETED,
        )

    tb_logger.attach(
        trainer,
        log_handler=OptimizerParamsHandler(optimizer, param_name="lr"),
        event_name=Events.ITERATION_COMPLETED,
    )
