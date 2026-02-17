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
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__(reduction)
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.device = device

    def forward(self, z: Tensor, target: Tensor) -> Tensor:
        """
        Returns:
          - reduction='none': per-original-sample vector [B], zeros for invalid samples and for anchors w/ no positives
          - otherwise: reduced scalar
        """
        B = z.size(0)
        device = self.device if self.device is not None else z.device

        valid = target != self.ignore_index
        if valid.sum() < 2:
            out = z.sum() * 0.0
            if self.reduction == "none":
                return torch.zeros(B, device=device, dtype=z.dtype) + out
            return out

        z_v = F.normalize(z[valid], dim=1, eps=1e-8)
        t_v = target[valid]
        n = z_v.size(0)

        sim = (z_v @ z_v.T) / self.temperature

        self_mask = torch.eye(n, dtype=torch.bool, device=device)
        sim = sim.masked_fill(self_mask, torch.finfo(sim.dtype).min)

        pos_mask = (t_v[:, None] == t_v[None, :]) & (~self_mask)
        pos_count = pos_mask.sum(dim=1)  # [n]
        valid_anchors = pos_count > 0

        if valid_anchors.sum() == 0:
            out = z.sum() * 0.0
            if self.reduction == "none":
                return torch.zeros(B, device=device, dtype=z.dtype) + out
            return out

        log_probs = F.log_softmax(sim, dim=1)  # [n, n]

        # per anchor loss
        loss_v = torch.zeros(n, device=device, dtype=z.dtype)
        loss_v[valid_anchors] = -(
            (log_probs * pos_mask).sum(dim=1)[valid_anchors]
            / pos_count[valid_anchors].to(z.dtype)
        )

        if self.reduction == "none":
            out = torch.zeros(B, device=device, dtype=z.dtype)
            valid_idx = torch.where(valid)[0]  # indices into original batch
            out[valid_idx] = loss_v
            return out

        # reduce only over valid anchors (ignore zeros from anchors w/ no positives)
        return self._reduce(loss_v[valid_anchors])


@LossFactory.register()
class JointSupConCELoss(BaseLoss):
    """Joint supervised contrastive and cross-entropy loss."""

    def __init__(
        self,
        lam_supcon: float = 0.2,
        reduction: str = "mean",
        ignore_index: int = -1,
        temperature: float = 0.07,
        margin: float = 0.2,
        label_smoothing: float = 0.0,
        class_weight: Optional[Tensor | list[float]] = None,
        device: Optional[torch.device] = torch.device("cpu"),
    ) -> None:
        super().__init__(reduction)
        self.supcon = SupervisedContrastiveLoss(
            reduction="none",
            ignore_index=ignore_index,
            temperature=temperature,
            device=device,
        )
        self.ce = CrossEntropyLoss(
            reduction="none",
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            class_weight=class_weight,
            device=device,
        )
        self.lam_supcon = lam_supcon
        self.ignore_index = ignore_index

    def forward(
        self,
        contrastive_logits: Tensor,
        logits: Tensor,
        target: Tensor,
        clusters: Tensor,
    ) -> Tensor:
        """
        Compute joint loss: CE + λ_supcon * SupCon

        Note: Embeddings are normalized inside individual loss functions.
        Separate lambda values allow balancing losses with different scales.
        """
        # All losses return [B] with reduction="none"
        supcon_c = self.supcon(
            contrastive_logits, clusters
        )  # [B], internally normalizes
        ce = self.ce(logits, target)  # [B]

        # Debug: monitor individual loss scales
        print(
            f"CE: {ce.mean().item():.4f}, SupCon: {supcon_c.mean().item():.4f} -> {supcon_c.mean().item() * self.lam_supcon:.4f}"
        )

        # Combine with separate weights for each contrastive loss
        loss = ce + self.lam_supcon * supcon_c

        return self._reduce(loss)
