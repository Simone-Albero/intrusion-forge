from typing import Dict, Optional, Tuple, List

import torch
from torch.nn.utils import clip_grad_norm_

from ignite.engine import Engine

from .data.batch import Batch, ensure_batch
from .model.base import BaseModel, ModelOutput
from .loss.base import BaseLoss


def _forward_and_loss(
    model: BaseModel,
    batch: Batch,
    loss_fn: Optional[BaseLoss] = None,
) -> Tuple[ModelOutput, Optional[torch.Tensor]]:
    """Forward pass with optional loss computation."""
    outputs = model(*batch.features)

    if loss_fn is None or batch.targets is None:
        return outputs, None

    loss = loss_fn(*model.for_loss(outputs, *batch.targets))
    return outputs, loss


def train_step(engine: Engine, batch: Batch) -> Dict[str, float]:
    model, optimizer, scheduler, loss_fn, device = (
        engine.state.model,
        engine.state.optimizer,
        engine.state.scheduler,
        engine.state.loss_fn,
        engine.state.device,
    )

    model.train()
    batch = ensure_batch(batch).to(device, non_blocking=True)

    optimizer.zero_grad()
    _, loss = _forward_and_loss(model, batch, loss_fn)
    loss.backward()

    grad_norm = float(
        clip_grad_norm_(model.parameters(), max_norm=engine.state.max_grad_norm)
    )

    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return {
        "loss": loss.item(),
        "grad_norm": grad_norm,
    }


def eval_step(engine: Engine, batch: Batch) -> Dict[str, float]:
    model, loss_fn, device = (
        engine.state.model,
        engine.state.loss_fn,
        engine.state.device,
    )

    model.eval()
    batch = ensure_batch(batch).to(device, non_blocking=True)

    with torch.no_grad():
        _, loss = _forward_and_loss(model, batch, loss_fn)

    return {
        "loss": loss.item(),
    }


def test_step(engine: Engine, batch: Batch) -> Dict[str, torch.Tensor]:
    model, device, loss_fn = (
        engine.state.model,
        engine.state.device,
        engine.state.loss_fn if hasattr(engine.state, "loss_fn") else None,
    )

    model.eval()
    batch = ensure_batch(batch).to(device, non_blocking=True)

    with torch.no_grad():
        output = model(*batch.features)
        if loss_fn is not None:
            loss = loss_fn(*model.for_loss(output, *batch.targets))

    return {
        "input": torch.cat(batch.features, dim=1),
        "output": output,
        "loss": loss if loss_fn is not None else torch.tensor(0.0),
        "y_true": batch.targets[0] if len(batch.targets) == 1 else torch.tensor([]),
    }


def ignore_classes(
    output: torch.Tensor,
    y_true: torch.Tensor,
    ignore_classes: Optional[List[int]] = None,
) -> Dict[str, torch.Tensor]:
    if ignore_classes is not None:
        mask = ~torch.isin(y_true, torch.tensor(ignore_classes, device=y_true.device))

        return output[mask], y_true[mask]

    return output, y_true
