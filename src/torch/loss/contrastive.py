import torch
import torch.nn.functional as F
from torch import Tensor

from . import LossFactory
from .base import BaseLoss
from .classification import CrossEntropyLoss


def _zero_out(z: Tensor, B: int, reduction: str) -> Tensor:
    """Zero loss tensor that keeps the autograd graph alive."""
    zero = z.sum() * 0.0
    if reduction == "none":
        return torch.zeros(B, device=z.device, dtype=z.dtype) + zero
    return zero


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
        B = z.size(0)
        valid = target != self.ignore_index
        if valid.sum() < 2:
            return _zero_out(z, B, self.reduction)

        z_v = F.normalize(z[valid], dim=1, eps=1e-8)
        t_v = target[valid]
        n = z_v.size(0)

        sim = (z_v @ z_v.T) / self.temperature
        self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(self_mask, torch.finfo(sim.dtype).min)

        pos_mask = (t_v[:, None] == t_v[None, :]) & (~self_mask)
        pos_count = pos_mask.sum(dim=1)
        valid_anchors = pos_count > 0

        if valid_anchors.sum() == 0:
            return _zero_out(z, B, self.reduction)

        log_probs = F.log_softmax(sim, dim=1)
        loss_v = torch.zeros(n, device=z.device, dtype=z.dtype)
        loss_v[valid_anchors] = -(
            (log_probs * pos_mask).sum(dim=1)[valid_anchors]
            / pos_count[valid_anchors].to(z.dtype)
        )

        if self.reduction == "none":
            out = torch.zeros(B, device=z.device, dtype=z.dtype)
            out[torch.where(valid)[0]] = loss_v
            return out

        return self._reduce(loss_v[valid_anchors])


@LossFactory.register()
class RivalRepulsionLoss(BaseLoss):
    """Centroid-based rival repulsion with softplus margin penalty."""

    def __init__(
        self,
        cluster_rivals: dict[int, list[int]] | None = None,
        reduction: str = "mean",
        device: torch.device | None = None,
        ignore_index: int = -1,
        margin: float = 0.7,
        beta: float = 20.0,
        detach_centroids: bool = False,
    ) -> None:
        super().__init__(reduction)
        self.cluster_rivals = cluster_rivals or {}
        self.device = device or torch.device("cpu")
        self.ignore_index = ignore_index
        self.margin = margin
        self.beta = beta
        self.detach_centroids = detach_centroids

    def forward(self, z: Tensor, clusters: Tensor) -> Tensor:
        B = z.size(0)

        if not self.cluster_rivals:
            return _zero_out(z, B, self.reduction)

        valid = clusters != self.ignore_index
        if valid.sum() < 2:
            return _zero_out(z, B, self.reduction)

        z_v = F.normalize(z[valid], dim=1, eps=1e-8)
        c_v = clusters[valid].to(z.device)
        uniq = torch.unique(c_v)

        if uniq.numel() < 2:
            return _zero_out(z, B, self.reduction)

        centroids = torch.stack(
            [F.normalize(z_v[c_v == cid].mean(dim=0), dim=0, eps=1e-8) for cid in uniq]
        )
        if self.detach_centroids:
            centroids = centroids.detach()

        idx = {int(cid): i for i, cid in enumerate(uniq.tolist())}
        pi, pj = [], []
        for a, rivals in self.cluster_rivals.items():
            if a not in idx:
                continue
            for b in rivals:
                if int(b) in idx:
                    pi.append(idx[a])
                    pj.append(idx[int(b)])

        if not pi:
            return _zero_out(z, B, self.reduction)

        pi = torch.tensor(pi, device=z.device, dtype=torch.long)
        pj = torch.tensor(pj, device=z.device, dtype=torch.long)
        cos = (centroids[pi] * centroids[pj]).sum(dim=1)
        loss_pairs = F.softplus(self.beta * (cos - self.margin)) / self.beta

        if self.reduction == "none":
            out = torch.zeros(B, device=z.device, dtype=z.dtype)
            out[valid] = loss_pairs.mean()
            return out

        return loss_pairs.mean() if self.reduction == "mean" else loss_pairs.sum()


@LossFactory.register()
class JointSupConCELoss(BaseLoss):
    """Joint supervised contrastive and cross-entropy loss."""

    def __init__(
        self,
        lam_supcon: float = 0.2,
        reduction: str = "mean",
        ignore_index: int = -1,
        temperature: float = 0.07,
        label_smoothing: float = 0.0,
        class_weight: Tensor | list[float] | None = None,
        device: torch.device = torch.device("cpu"),
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
        self.rival_repulsion = RivalRepulsionLoss(reduction="mean", device=device)
        self.lam_supcon = lam_supcon
        self.ignore_index = ignore_index

    def update_cluster_rivals(self, cluster_rivals: dict[int, list[int]]) -> None:
        self.rival_repulsion.cluster_rivals = cluster_rivals

    def forward(
        self,
        contrastive_logits: Tensor,
        logits: Tensor,
        target: Tensor,
        clusters: Tensor,
    ) -> Tensor:
        supcon_c = self.supcon(contrastive_logits, clusters)
        ce = self.ce(logits, target)
        rival_loss = self.rival_repulsion(contrastive_logits, clusters)
        loss = ce + self.lam_supcon * supcon_c + 3 * rival_loss
        return self._reduce(loss)
