from typing import Callable, Optional, Dict, Any
from pathlib import Path
import shutil

import torch.nn as nn
from torch.optim import Optimizer
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Metric, Average
from ignite.handlers.tensorboard_logger import (
    TensorboardLogger,
    global_step_from_engine,
    WeightsHistHandler,
    GradsHistHandler,
    OptimizerParamsHandler,
)


class EngineBuilder:
    """Builder for creating and configuring Ignite engines with dynamic state.

    Example:
        >>> from src.torch.engine import train_step
        >>>
        >>> engine = (EngineBuilder(train_step)
        ...     .with_state(model=model, optimizer=optimizer, device=device)
        ...     .with_metric("loss", Average(output_transform=lambda x: x["loss"]))
        ...     .build())
    """

    def __init__(self, step_function: Callable):
        """Initialize builder with a step function.

        Args:
            step_function: Function that defines engine behavior (train_step, eval_step, etc.)
        """
        self._step_function = step_function
        self._state_kwargs: Dict[str, Any] = {}
        self._metrics: Dict[str, Metric] = {}
        self._event_handlers: list = []

    def with_state(self, **kwargs) -> "EngineBuilder":
        """Add attributes to engine state dynamically.

        Args:
            **kwargs: Key-value pairs to add to engine.state
        """
        self._state_kwargs.update(kwargs)
        return self

    def with_metric(self, name: str, metric: Metric) -> "EngineBuilder":
        """Attach a metric to the engine.

        Args:
            name: Metric name
            metric: Ignite metric instance
        """
        self._metrics[name] = metric
        return self

    def with_handler(
        self, event: Events, handler: Callable, *args, **kwargs
    ) -> "EngineBuilder":
        """Add an event handler to the engine.

        Args:
            event: Ignite event to trigger the handler
            handler: Handler function or callable
            *args: Positional arguments for the handler
            **kwargs: Keyword arguments for the handler

        Returns:
            Self for method chaining
        """
        self._event_handlers.append((event, handler, args, kwargs))
        return self

    def with_early_stopping(
        self,
        trainer: Engine,
        metric: str = "loss",
        patience: int = 10,
        min_delta: float = 0.0,
        maximize: bool = False,
    ) -> "EngineBuilder":
        """Add early stopping handler (for validator engines).

        Args:
            trainer: Training engine to stop
            metric: Metric name to monitor
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
            maximize: Whether to maximize the metric (False = minimize)

        Returns:
            Self for method chaining
        """
        score_sign = 1 if maximize else -1
        handler = EarlyStopping(
            patience=patience,
            min_delta=min_delta,
            score_function=lambda engine: score_sign * engine.state.metrics[metric],
            trainer=trainer,
        )
        return self.with_handler(Events.COMPLETED, handler)

    def with_checkpointing(
        self,
        trainer: Engine,
        checkpoint_dir: Path,
        objects_to_save: Dict[str, Any],
        metric: str = "loss",
        maximize: bool = False,
        n_saved: int = 1,
        filename_prefix: str = "checkpoint",
    ) -> "EngineBuilder":
        """Add model checkpointing handler (for validator engines).

        Args:
            trainer: Training engine for epoch tracking
            checkpoint_dir: Directory to save checkpoints
            objects_to_save: Dict of objects to checkpoint (e.g., {"model": model})
            metric: Metric name to monitor
            maximize: Whether to maximize the metric (False = minimize)
            n_saved: Number of best checkpoints to keep
            filename_prefix: Prefix for checkpoint filenames
        """
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # pulisci la cartella prima di salvare i nuovi checkpoint
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)

        score_sign = 1 if maximize else -1

        handler = ModelCheckpoint(
            dirname=checkpoint_dir,
            filename_prefix=filename_prefix,
            score_function=lambda engine: score_sign * engine.state.metrics[metric],
            score_name=metric,
            n_saved=n_saved,
            global_step_transform=lambda engine, event_name: trainer.state.epoch,
            require_empty=False,
        )
        return self.with_handler(Events.COMPLETED, handler, objects_to_save)

    def with_tensorboard(
        self,
        tb_logger: TensorboardLogger,
        event: Events = Events.ITERATION_COMPLETED,
        tag: Optional[str] = None,
        output_transform: Optional[Callable] = None,
        metric_names: Optional[list] = None,
        trainer: Optional[Engine] = None,
    ) -> "EngineBuilder":
        """Add TensorBoard output logging.

        Args:
            tb_logger: TensorBoard logger instance
            event: Event to trigger logging
            tag: Tag for the logs
            output_transform: Transform function for outputs
            metric_names: List of metric names to log
            trainer: Reference trainer for global step (for validators)

        Returns:
            Self for method chaining
        """
        attach_kwargs = {"event_name": event}

        if tag:
            attach_kwargs["tag"] = tag
        if output_transform:
            attach_kwargs["output_transform"] = output_transform
        if metric_names:
            attach_kwargs["metric_names"] = metric_names
        if trainer:
            attach_kwargs["global_step_transform"] = global_step_from_engine(trainer)

        # Attach logger directly (no need to defer to STARTED event)
        def attach_logger(engine):
            tb_logger.attach_output_handler(engine, **attach_kwargs)

        # Use STARTED for attaching, but the actual logging happens at the specified event
        return self.with_handler(Events.STARTED, attach_logger)

    def with_weights_logging(
        self,
        tb_logger: TensorboardLogger,
        model: nn.Module,
        event: Events = Events.EPOCH_COMPLETED,
    ) -> "EngineBuilder":
        """Add weight histogram logging to TensorBoard.

        Args:
            tb_logger: TensorBoard logger instance
            model: Model to log weights from
            event: Event to trigger logging

        Returns:
            Self for method chaining
        """

        def attach_logger(engine):
            tb_logger.attach(
                engine,
                log_handler=WeightsHistHandler(model),
                event_name=event,
            )

        return self.with_handler(Events.STARTED, attach_logger)

    def with_gradients_logging(
        self,
        tb_logger: TensorboardLogger,
        model: nn.Module,
        event: Events = Events.EPOCH_COMPLETED,
    ) -> "EngineBuilder":
        """Add gradient histogram logging to TensorBoard.

        Args:
            tb_logger: TensorBoard logger instance
            model: Model to log gradients from
            event: Event to trigger logging

        Returns:
            Self for method chaining
        """

        def attach_logger(engine):
            tb_logger.attach(
                engine,
                log_handler=GradsHistHandler(model),
                event_name=event,
            )

        return self.with_handler(Events.STARTED, attach_logger)

    def with_optimizer_logging(
        self,
        tb_logger: TensorboardLogger,
        optimizer: Optimizer,
        param_name: str = "lr",
        event: Events = Events.ITERATION_COMPLETED,
    ) -> "EngineBuilder":
        """Add optimizer parameter logging to TensorBoard.

        Args:
            tb_logger: TensorBoard logger instance
            optimizer: Optimizer to log parameters from
            param_name: Parameter name to log (e.g., "lr")
            event: Event to trigger logging

        Returns:
            Self for method chaining
        """

        def attach_logger(engine):
            tb_logger.attach(
                engine,
                log_handler=OptimizerParamsHandler(optimizer, param_name=param_name),
                event_name=event,
            )

        return self.with_handler(Events.STARTED, attach_logger)

    def build(self) -> Engine:
        """Build and return the configured engine.

        Returns:
            Configured Ignite engine
        """
        engine = Engine(self._step_function)

        # Populate state
        for key, value in self._state_kwargs.items():
            setattr(engine.state, key, value)

        # Attach metrics
        for name, metric in self._metrics.items():
            metric.attach(engine, name)

        # Attach event handlers
        for event, handler, args, kwargs in self._event_handlers:
            engine.add_event_handler(event, handler, *args, **kwargs)

        return engine
