from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F

from .base import BaseLoss
from .classification import CrossEntropyLoss
from . import LossFactory


@LossFactory.register()
class SupervisedContrastiveLoss(BaseLoss):

    def __init__(
        self,
        reduction: str = "mean",
        ignore_index: int = -1,
        temperature: float = 0.07,
        device: Optional[torch.device] = torch.device("cpu"),
    ) -> None:
        """Initialize supervised contrastive loss.

        Args:
            reduction: How to reduce the loss ('mean', 'sum', 'none')
            ignore_index: Index to ignore in loss calculation
            temperature: Temperature scaling factor
        """
        super().__init__(reduction)
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.device = device

    def forward(self, z: Tensor, target: Tensor, clusters: Tensor) -> Tensor:
        batch_size = z.size(0)

        # keep only valid samples
        valid = clusters != self.ignore_index
        if valid.sum() < 2:
            return z.sum() * 0.0  # preserves graph

        z_valid = F.normalize(z[valid], dim=1, eps=1e-8)
        target_valid = target[valid]
        n = z_valid.size(0)

        # cosine sim via dot product
        sim = (z_valid @ z_valid.T) / self.temperature

        # mask self similarities
        self_mask = torch.eye(n, dtype=torch.bool, device=self.device)
        sim = sim.masked_fill(self_mask, -1e9)

        # positives: same target, excluding self
        pos_mask = (target_valid[:, None] == target_valid[None, :]) & (~self_mask)

        # anchors with at least one positive
        pos_count = pos_mask.sum(dim=1)
        valid_anchors = pos_count > 0
        if valid_anchors.sum() == 0:
            return z.sum() * 0.0

        log_probs = F.log_softmax(sim, dim=1)

        loss_per_valid = -(log_probs * pos_mask).sum(dim=1) / pos_count.clamp_min(1)
        loss_per_valid = loss_per_valid[valid_anchors]

        # If reduction is 'none', pad back to original batch size
        if self.reduction == "none":
            loss_per_sample = torch.zeros(batch_size, device=z.device, dtype=z.dtype)
            valid_indices = torch.where(valid)[0]
            anchor_indices = valid_indices[valid_anchors]
            loss_per_sample[anchor_indices] = loss_per_valid
            return loss_per_sample

        return self._reduce(loss_per_valid)


@LossFactory.register()
class JointSupConCELoss(BaseLoss):
    """Joint supervised contrastive and cross-entropy loss."""

    def __init__(
        self,
        lam: float = 1.0,
        reduction: str = "mean",
        ignore_index: int = -1,
        temperature: float = 0.07,
        label_smoothing: float = 0.0,
        class_weight: Optional[Tensor | list[float]] = None,
        device: Optional[torch.device] = torch.device("cpu"),
    ) -> None:
        super().__init__(reduction)
        self.supcon_loss = SupervisedContrastiveLoss(
            reduction="none",
            ignore_index=ignore_index,
            temperature=temperature,
            device=device,
        )
        self.ce_loss = CrossEntropyLoss(
            reduction="none",
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            class_weight=class_weight,
            device=device,
        )
        self.lam = lam

    def forward(
        self, z: Tensor, logits: Tensor, target: Tensor, clusters: Tensor
    ) -> Tensor:
        supcon_loss = self.supcon_loss(z, target, clusters)
        ce_loss = self.ce_loss(logits, target)

        loss = self.lam * supcon_loss + (1 - self.lam) * ce_loss
        return self._reduce(loss)
