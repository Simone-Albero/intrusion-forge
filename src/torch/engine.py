import torch
from torch.nn.utils import clip_grad_norm_

from ignite.engine import Engine

from .data.batch import Batch, ensure_batch
from .model.base import BaseModel, ModelOutput
from .loss.base import BaseLoss


def _forward_and_loss(
    model: BaseModel,
    batch: Batch,
    loss_fn: BaseLoss | None = None,
) -> tuple[ModelOutput, torch.Tensor | None]:
    """Forward pass with optional loss computation."""
    output = model(*batch.features)
    if loss_fn is None:
        return output, None
    return output, loss_fn(*model.for_loss(output, *batch.labels))


def train_step(engine: Engine, batch: Batch) -> dict[str, float]:
    s = engine.state
    model, optimizer, scheduler, loss_fn, device = (
        s.model,
        s.optimizer,
        s.scheduler,
        s.loss_fn,
        s.device,
    )

    model.train()
    batch = ensure_batch(batch).to(device, non_blocking=True)

    optimizer.zero_grad()
    _, loss = _forward_and_loss(model, batch, loss_fn)
    loss.backward()
    grad_norm = float(clip_grad_norm_(model.parameters(), max_norm=s.max_grad_norm))
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return {"loss": loss.item(), "grad_norm": grad_norm}


def eval_step(engine: Engine, batch: Batch) -> dict[str, float]:
    s = engine.state
    model, loss_fn, device = s.model, s.loss_fn, s.device

    model.eval()
    batch = ensure_batch(batch).to(device, non_blocking=True)

    with torch.no_grad():
        _, loss = _forward_and_loss(model, batch, loss_fn)

    return {"loss": loss.item()}


def test_step(engine: Engine, batch: Batch) -> dict:
    s = engine.state
    model, device = s.model, s.device
    loss_fn = getattr(s, "loss_fn", None)

    model.eval()
    batch = ensure_batch(batch).to(device, non_blocking=True)

    with torch.no_grad():
        output, loss = _forward_and_loss(model, batch, loss_fn)

    return {
        "output": output,
        "loss": loss if loss is not None else torch.tensor(0.0),
        "y_true": batch.labels[0] if len(batch.labels) == 1 else batch.labels,
    }
