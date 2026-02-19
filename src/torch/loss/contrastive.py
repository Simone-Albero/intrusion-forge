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
class RivalRepulsionLoss(BaseLoss):
    """
    Centroid-based rival repulsion:
    - compute centroids per cluster present in batch (excluding ignore_index)
    - penalize rival pairs with a margin on cosine similarity

    Loss per pair: softplus(beta * (cos - margin))
    (strong gradients when cos is high; stable; no exp explosion)
    """

    def __init__(
        self,
        cluster_rivals: Optional[dict[int, list[int]]] = None,
        reduction: str = "mean",
        device: Optional[torch.device] = None,
        ignore_index: int = -1,
        margin: float = 0.7,
        beta: float = 20.0,
        detach_centroids: bool = False,  # if True: centroids are stop-grad targets
    ) -> None:
        super().__init__(reduction)
        self.cluster_rivals = cluster_rivals if cluster_rivals is not None else {}
        self.device = device if device is not None else torch.device("cpu")
        self.ignore_index = ignore_index
        self.margin = margin
        self.beta = beta
        self.detach_centroids = detach_centroids

    def forward(self, z: Tensor, clusters: Tensor) -> Tensor:
        B = z.size(0)
        device = self.device if self.device is not None else z.device

        if not self.cluster_rivals:
            out = z.sum() * 0.0
            if self.reduction == "none":
                return torch.zeros(B, device=device, dtype=z.dtype) + out
            return out

        # valid mask (ignore ambiguous/unlabeled)
        valid = clusters != self.ignore_index
        if valid.sum() < 2:
            out = z.sum() * 0.0
            if self.reduction == "none":
                return torch.zeros(B, device=device, dtype=z.dtype) + out
            return out

        z_v = F.normalize(z[valid], dim=1, eps=1e-8)
        c_v = clusters[valid].to(device)

        # unique clusters in batch
        uniq = torch.unique(c_v)
        if uniq.numel() < 2:
            out = z.sum() * 0.0
            if self.reduction == "none":
                return torch.zeros(B, device=device, dtype=z.dtype) + out
            return out

        # centroids [K, D]
        centroids = []
        for cid in uniq.tolist():
            m = c_v == cid
            mu = z_v[m].mean(dim=0)
            mu = F.normalize(mu, dim=0, eps=1e-8)
            centroids.append(mu)
        centroids = torch.stack(centroids, dim=0)  # [K, D]

        if self.detach_centroids:
            centroids = centroids.detach()

        idx = {int(cid): i for i, cid in enumerate(uniq.tolist())}

        # build rival pair indices (only pairs present in this batch)
        pi, pj = [], []
        for a, rivals in self.cluster_rivals.items():
            if a not in idx:
                continue
            ia = idx[a]
            for b in rivals:
                b = int(b)
                if b in idx:
                    ib = idx[b]
                    pi.append(ia)
                    pj.append(ib)

        if len(pi) == 0:
            out = z.sum() * 0.0
            if self.reduction == "none":
                return torch.zeros(B, device=device, dtype=z.dtype) + out
            return out

        pi = torch.tensor(pi, device=device, dtype=torch.long)
        pj = torch.tensor(pj, device=device, dtype=torch.long)

        # cosine similarity between centroid pairs
        cos = (centroids[pi] * centroids[pj]).sum(dim=1)  # [P]

        # softplus margin penalty: always stable, strong near high cos
        loss_pairs = F.softplus(self.beta * (cos - self.margin)) / self.beta  # [P]

        # reduction
        if self.reduction == "none":
            # return [B] vector: distribute pair-loss to samples of involved clusters (optional)
            # simplest: return per-sample zeros except valid, with avg pair loss broadcast
            out = torch.zeros(B, device=device, dtype=z.dtype)
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
        self.rival_repulsion = RivalRepulsionLoss(reduction="mean", device=device)

        self.lam_supcon = lam_supcon
        self.ignore_index = ignore_index

    def update_cluster_rivals(self, cluster_rivals: dict[int, list[int]]) -> None:
        """Update rival clusters for the RivalRepulsionLoss."""
        self.rival_repulsion.cluster_rivals = cluster_rivals

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
        rival_loss = self.rival_repulsion(contrastive_logits, clusters)  # [B]

        # Debug: monitor individual loss scales
        print(
            f"CE: {ce.mean().item():.4f}, SupCon: {supcon_c.mean().item():.4f} -> {supcon_c.mean().item() * self.lam_supcon:.4f}, Rival: {rival_loss.mean().item():.4f}"
        )

        # Combine with separate weights for each contrastive loss
        loss = ce + self.lam_supcon * supcon_c + 3 * rival_loss

        return self._reduce(loss)
