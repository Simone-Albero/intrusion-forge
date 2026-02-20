from pathlib import Path
from typing import Callable
import shutil

import torch.nn as nn
from torch.optim import Optimizer
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Metric
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
        >>> engine = (EngineBuilder(train_step)
        ...     .with_state(model=model, optimizer=optimizer, device=device)
        ...     .with_metric("loss", Average(output_transform=lambda x: x["loss"]))
        ...     .build())
    """

    def __init__(self, step_function: Callable):
        self._step_function = step_function
        self._state_kwargs: dict[str, object] = {}
        self._metrics: dict[str, Metric] = {}
        self._event_handlers: list = []

    def with_state(self, **kwargs) -> "EngineBuilder":
        """Add attributes to engine state."""
        self._state_kwargs.update(kwargs)
        return self

    def with_metric(self, name: str, metric: Metric) -> "EngineBuilder":
        """Attach a metric to the engine."""
        self._metrics[name] = metric
        return self

    def with_handler(
        self, event: Events, handler: Callable, *args, **kwargs
    ) -> "EngineBuilder":
        """Add an event handler."""
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
        """Add early stopping (for validator engines)."""
        sign = 1 if maximize else -1
        handler = EarlyStopping(
            patience=patience,
            min_delta=min_delta,
            score_function=lambda engine: sign * engine.state.metrics[metric],
            trainer=trainer,
        )
        return self.with_handler(Events.COMPLETED, handler)

    def with_checkpointing(
        self,
        trainer: Engine,
        checkpoint_dir: Path,
        objects_to_save: dict[str, object],
        metric: str = "loss",
        maximize: bool = False,
        n_saved: int = 1,
        filename_prefix: str = "",
    ) -> "EngineBuilder":
        """Add model checkpointing (for validator engines)."""
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True)

        sign = 1 if maximize else -1
        handler = ModelCheckpoint(
            dirname=checkpoint_dir,
            filename_prefix=filename_prefix,
            score_function=lambda engine: sign * engine.state.metrics[metric],
            score_name=metric,
            n_saved=n_saved,
            global_step_transform=lambda engine, _: trainer.state.epoch,
            require_empty=False,
        )
        return self.with_handler(Events.COMPLETED, handler, objects_to_save)

    def with_tensorboard(
        self,
        tb_logger: TensorboardLogger,
        event: Events = Events.ITERATION_COMPLETED,
        tag: str | None = None,
        output_transform: Callable | None = None,
        metric_names: list | None = None,
        trainer: Engine | None = None,
    ) -> "EngineBuilder":
        """Add TensorBoard output logging."""
        kwargs = {
            "event_name": event,
            **({} if tag is None else {"tag": tag}),
            **(
                {}
                if output_transform is None
                else {"output_transform": output_transform}
            ),
            **({} if metric_names is None else {"metric_names": metric_names}),
            **(
                {}
                if trainer is None
                else {"global_step_transform": global_step_from_engine(trainer)}
            ),
        }
        return self.with_handler(
            Events.STARTED,
            lambda engine: tb_logger.attach_output_handler(engine, **kwargs),
        )

    def with_weights_logging(
        self,
        tb_logger: TensorboardLogger,
        model: nn.Module,
        event: Events = Events.EPOCH_COMPLETED,
    ) -> "EngineBuilder":
        """Add weight histogram logging to TensorBoard."""
        return self.with_handler(
            Events.STARTED,
            lambda engine: tb_logger.attach(
                engine, log_handler=WeightsHistHandler(model), event_name=event
            ),
        )

    def with_gradients_logging(
        self,
        tb_logger: TensorboardLogger,
        model: nn.Module,
        event: Events = Events.EPOCH_COMPLETED,
    ) -> "EngineBuilder":
        """Add gradient histogram logging to TensorBoard."""
        return self.with_handler(
            Events.STARTED,
            lambda engine: tb_logger.attach(
                engine, log_handler=GradsHistHandler(model), event_name=event
            ),
        )

    def with_optimizer_logging(
        self,
        tb_logger: TensorboardLogger,
        optimizer: Optimizer,
        param_name: str = "lr",
        event: Events = Events.ITERATION_COMPLETED,
    ) -> "EngineBuilder":
        """Add optimizer parameter logging to TensorBoard."""
        return self.with_handler(
            Events.STARTED,
            lambda engine: tb_logger.attach(
                engine,
                log_handler=OptimizerParamsHandler(optimizer, param_name=param_name),
                event_name=event,
            ),
        )

    def build(self) -> Engine:
        """Build and return the configured engine."""
        engine = Engine(self._step_function)
        for key, value in self._state_kwargs.items():
            setattr(engine.state, key, value)
        for name, metric in self._metrics.items():
            metric.attach(engine, name)
        for event, handler, args, kwargs in self._event_handlers:
            engine.add_event_handler(event, handler, *args, **kwargs)
        return engine
