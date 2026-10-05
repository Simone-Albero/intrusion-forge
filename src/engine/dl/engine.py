import torch
from ignite.engine import Engine
from torch.nn.utils import clip_grad_norm_

from src.engine.dl.data.batch import Batch, ensure_batch
from src.engine.dl.loss.base import BaseLoss
from src.engine.dl.model.base import BaseModel, ModelOutput


def _forward_and_loss(
    model: BaseModel,
    batch: Batch,
    loss_fn: BaseLoss | None = None,
) -> tuple[ModelOutput, torch.Tensor | None]:
    """Forward pass and, when a loss is given, its value."""
    output = model(*batch.features)
    if loss_fn is None:
        return output, None
    return output, loss_fn(*model.for_loss(output, *batch.labels))


def train_step(engine: Engine, batch: Batch) -> dict[str, float]:
    """Single training step: forward, loss, backward, optimizer."""
    state = engine.state
    model, optimizer, scheduler, loss_fn, device = (
        state.model,
        state.optimizer,
        state.scheduler,
        state.loss_fn,
        state.device,
    )

    model.train()
    batch = ensure_batch(batch).to(device, non_blocking=True)

    optimizer.zero_grad()
    _, loss = _forward_and_loss(model, batch, loss_fn)
    loss.backward()
    grad_norm = float(clip_grad_norm_(model.parameters(), max_norm=state.max_grad_norm))
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return {"loss": loss.item(), "grad_norm": grad_norm}


def eval_step(engine: Engine, batch: Batch) -> dict[str, float]:
    """Single evaluation step returning the batch loss."""
    state = engine.state
    model, loss_fn, device = state.model, state.loss_fn, state.device

    model.eval()
    batch = ensure_batch(batch).to(device, non_blocking=True)

    with torch.no_grad():
        _, loss = _forward_and_loss(model, batch, loss_fn)

    if loss is None:
        raise ValueError("eval_step requires a loss_fn in the engine state.")
    return {"loss": loss.item()}
