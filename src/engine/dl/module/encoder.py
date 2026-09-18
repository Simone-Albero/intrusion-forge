from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn

from ..module.embedding import EmbeddingModule
from ..module.mlp import MLPModule


class TabularEncoderModule(nn.Module):
    """Unified encoder for numerical and categorical features."""

    def __init__(
        self,
        num_numerical_features: int,
        out_features: int,
        *,
        cardinalities: Sequence[int] = (),
        max_emb_dim: int = 50,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Callable[[int], nn.Module] | None = nn.BatchNorm1d,
    ) -> None:
        super().__init__()
        self.embedding = EmbeddingModule(cardinalities=cardinalities, max_emb_dim=max_emb_dim)
        total = num_numerical_features + sum(self.embedding.embedding_dims)
        self.mlp = MLPModule(
            total,
            out_features,
            hidden_dims=hidden_dims,
            activation=activation,
            norm_layer=norm_layer,
            dropout=dropout,
        )

    def forward(self, x_numerical: Tensor, x_categorical: Tensor) -> Tensor:
        """Encode the concatenated numerical and embedded categorical blocks."""
        return self.mlp(torch.cat([x_numerical, self.embedding(x_categorical)], dim=1))
