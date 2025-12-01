from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F

from .base import BaseLoss
from . import LossFactory


@LossFactory.register()
class SupervisedContrastiveLoss(BaseLoss):

    def __init__(
        self,
        reduction: str = "mean",
        ignore_index: int = -1,
        temperature: float = 0.07,
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

    def forward(
        self,
        z: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Compute supervised contrastive loss.

        Args:
            z: Embedding tensor [batch_size, embedding_dim]
            target: Target class indices [batch_size]

        Returns:
            Loss tensor (scalar or per-sample based on reduction)
        """
        # Normalize embeddings (with epsilon to avoid division by zero)
        z = F.normalize(z, dim=1, eps=1e-8)
        batch_size = z.shape[0]

        # Compute similarity matrix
        similarity = (z @ z.T) / self.temperature

        # Mask self-similarities to large negative value (not -inf to avoid log issues)
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=z.device)
        similarity = similarity.masked_fill(self_mask, -1e9)

        # Create positive mask based on target labels
        positive_mask = target.unsqueeze(0) == target.unsqueeze(1)
        positive_mask = positive_mask.masked_fill(self_mask, False)

        # Filter out ignored indices
        valid_mask = target != self.ignore_index
        positive_mask = (
            positive_mask & valid_mask.unsqueeze(0) & valid_mask.unsqueeze(1)
        )

        # Count the number of positive pairs for each sample
        num_positives = positive_mask.sum(dim=1)

        # Only compute loss for samples with at least one positive pair
        valid_samples = num_positives > 0
        if not valid_samples.any():
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        # Compute log probabilities
        log_probs = F.log_softmax(similarity, dim=1)

        # Sum log probabilities over positive pairs
        log_probs_positive = (log_probs * positive_mask).sum(dim=1)

        # Average over positive pairs (add epsilon to avoid division by zero)
        loss = -log_probs_positive[valid_samples] / (
            num_positives[valid_samples] + 1e-8
        )

        return self._reduce(loss)


@LossFactory.register()
class NtXentLoss(BaseLoss):

    def __init__(
        self,
        reduction: str = "mean",
        temperature: float = 0.07,
    ) -> None:
        """Initialize NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

        An unsupervised contrastive loss for self-supervised learning.

        Args:
            reduction: How to reduce the loss ('mean', 'sum', 'none')
            temperature: Temperature scaling factor
        """
        super().__init__(reduction)
        self.temperature = temperature

    def forward(
        self,
        z1: Tensor,
        z2: Tensor,
        *args,
    ) -> Tensor:
        """Compute NT-Xent loss.

        Args:
            z1: First set of embeddings [batch_size, embedding_dim]
            z2: Second set of embeddings [batch_size, embedding_dim]

        Returns:
            Loss tensor (scalar or per-sample based on reduction)
        """
        # Normalize embeddings (with epsilon to avoid division by zero)
        z1 = F.normalize(z1, dim=1, eps=1e-8)
        z2 = F.normalize(z2, dim=1, eps=1e-8)

        batch_size = z1.shape[0]

        # Concatenate both views
        z = torch.cat([z1, z2], dim=0)  # [2*batch_size, embedding_dim]

        # Compute similarity matrix
        similarity = (z @ z.T) / self.temperature

        # Mask self-similarities to large negative value (not -inf to avoid log issues)
        self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        similarity = similarity.masked_fill(self_mask, -1e9)

        # Create positive mask for augmented pairs
        positive_mask = torch.zeros_like(similarity, dtype=torch.bool)
        for i in range(batch_size):
            positive_mask[i, i + batch_size] = True
            positive_mask[i + batch_size, i] = True

        # Compute log probabilities and sum over positives
        log_probs = F.log_softmax(similarity, dim=1)
        log_probs_positive = (log_probs * positive_mask).sum(dim=1)

        # Each sample has exactly one positive pair
        loss = -log_probs_positive

        return self._reduce(loss)
