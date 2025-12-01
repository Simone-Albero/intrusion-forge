from typing import Sequence, Optional

import torch
from torch import nn, Tensor


class EmbeddingModule(nn.Module):
    """Feature embedding for categorical variables with padding/unknown support."""

    def __init__(
        self,
        num_features: Optional[int],
        cardinalities: Optional[Sequence[int]],
        max_emb_dim: int = 50,
    ):
        """Initialize feature embedding.

        Note:
            Index 0 is reserved for padding/unknown values and initialized to zeros.
        """
        super().__init__()
        self.embedding_layers = nn.ModuleList()
        self.embedding_dims = []
        if num_features is None and cardinalities is None:
            raise ValueError("Either num_features or cardinalities must be provided.")

        if cardinalities is None:
            cardinalities = [max_emb_dim] * num_features  # Default cardinality

        for card in cardinalities:
            dim = min(max_emb_dim, int(card**0.5))
            embedding_layer = nn.Embedding(card + 1, dim, padding_idx=0)
            self.embedding_layers.append(embedding_layer)
            self.embedding_dims.append(dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Categorical features [batch_size, num_categorical_features]
               Values with 0 are treated as padding/unknown

        Returns:
            Embedded features [batch_size, sum(embedding_dims)]
        """
        embedded_features = []
        for i, embedding_layer in enumerate(self.embedding_layers):
            embedded_features.append(embedding_layer(x[:, i]))
        return torch.cat(embedded_features, dim=1)
