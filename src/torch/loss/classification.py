from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F

from .base import BaseLoss
from . import LossFactory


@LossFactory.register()
class ClassificationLoss(BaseLoss):
    """Cross-entropy loss for classification with hard labels."""

    def __init__(
        self,
        reduction: str = "mean",
        ignore_index: int = -1,
        label_smoothing: float = 0.0,
        class_weight: Optional[Tensor | list[float]] = None,
        device: Optional[torch.device] = torch.device("cpu"),
    ) -> None:
        """Initialize classification loss.

        Args:
            reduction: How to reduce the loss ('mean', 'sum', 'none')
            ignore_index: Index to ignore in loss calculation
            label_smoothing: Label smoothing factor [0, 1)
            class_weight: Per-class weights for imbalanced datasets
        """
        super().__init__(reduction)

        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError("label_smoothing must be in [0, 1)")

        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

        if class_weight is not None:
            if not isinstance(class_weight, torch.Tensor):
                class_weight = torch.tensor(list(class_weight), dtype=torch.float32).to(
                    device=device
                )
            self.register_buffer("class_weight", class_weight)
        else:
            self.class_weight = None

    def forward(
        self,
        x: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Compute cross-entropy loss.

        Args:
            x: Logits tensor [batch_size, num_classes]
            target: Target class indices [batch_size]

        Returns:
            Loss tensor (scalar or per-sample based on reduction)
        """
        class_weight = self.class_weight

        loss = F.cross_entropy(
            x,
            target,
            weight=class_weight,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )

        valid_mask = target != self.ignore_index
        loss = loss[valid_mask]
        return self._reduce(loss)


@LossFactory.register()
class FocalLoss(BaseLoss):
    """Focal Loss for addressing class imbalance.

    Focal Loss down-weights easy examples and focuses on hard negatives.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        class_weight: Optional[Tensor | list[float]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -1,
        label_smoothing: float = 0.0,
        device: Optional[torch.device] = torch.device("cpu"),
    ) -> None:
        """Initialize focal loss.

        Args:
            class_weight: Per-class weights for balancing [num_classes]. If None, no weighting.
            gamma: Focusing parameter (gamma >= 0). gamma=0 is equivalent to CE.
            reduction: How to reduce the loss ('mean', 'sum', 'none')
            ignore_index: Index to ignore in loss calculation
            label_smoothing: Label smoothing factor [0, 1)
        """
        super().__init__(reduction)

        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")

        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError("label_smoothing must be in [0, 1)")

        self.gamma = gamma
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

        if class_weight is not None:
            if not isinstance(class_weight, torch.Tensor):
                class_weight = torch.tensor(list(class_weight), dtype=torch.float32).to(
                    device=device
                )
            self.register_buffer("class_weight", class_weight)
        else:
            self.class_weight = None

    def forward(
        self,
        x: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Compute focal loss.

        Args:
            x: Logits tensor [batch_size, num_classes]
            target: Target class indices [batch_size]

        Returns:
            Loss tensor (scalar or per-sample based on reduction)
        """
        import torch

        ce_loss = F.cross_entropy(
            x,
            target,
            reduction="none",
            label_smoothing=self.label_smoothing,
            ignore_index=self.ignore_index,
        )

        p = torch.softmax(x, dim=1)
        p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)

        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * ce_loss

        if self.class_weight is not None:
            alpha_t = self.class_weight.gather(0, target)
            loss = alpha_t * loss

        valid_mask = target != self.ignore_index
        loss = loss[valid_mask]

        return self._reduce(loss)
