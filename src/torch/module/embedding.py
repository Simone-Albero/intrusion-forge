from collections.abc import Sequence

import torch
from torch import Tensor, nn


class EmbeddingModule(nn.Module):
    """Categorical feature embedding with padding/unknown support (index 0 reserved)."""

    def __init__(
        self,
        num_features: int | None,
        cardinalities: Sequence[int] | None,
        max_emb_dim: int = 50,
    ) -> None:
        super().__init__()
        if num_features is None and cardinalities is None:
            raise ValueError("Either num_features or cardinalities must be provided.")
        if cardinalities is None:
            cardinalities = [max_emb_dim] * num_features
        self.embedding_dims = [min(max_emb_dim, int(c**0.5)) for c in cardinalities]
        self.embedding_layers = nn.ModuleList(
            nn.Embedding(c + 1, d, padding_idx=0)
            for c, d in zip(cardinalities, self.embedding_dims)
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat(
            [layer(x[:, i]) for i, layer in enumerate(self.embedding_layers)], dim=1
        )
