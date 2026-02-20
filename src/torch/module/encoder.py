from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn

from ..module.embedding import EmbeddingModule
from ..module.mlp import MLPModule


class NumericalEncoderModule(nn.Module):
    """Encoder for numerical features using MLP."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Callable[[int], nn.Module] | None = nn.BatchNorm1d,
    ) -> None:
        super().__init__()
        self.mlp = MLPModule(
            in_features, out_features, hidden_dims, activation, norm_layer, dropout
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class CategoricalEncoderModule(nn.Module):
    """Encoder for categorical features using Embedding + MLP."""

    def __init__(
        self,
        out_features: int,
        num_features: int | None = None,
        cardinalities: Sequence[int] | None = None,
        max_emb_dim: int = 50,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Callable[[int], nn.Module] | None = nn.BatchNorm1d,
    ) -> None:
        super().__init__()
        self.embedding = EmbeddingModule(num_features, cardinalities, max_emb_dim)
        self.mlp = MLPModule(
            sum(self.embedding.embedding_dims),
            out_features,
            hidden_dims,
            activation,
            norm_layer,
            dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(self.embedding(x))


class TabularEncoderModule(nn.Module):
    """Unified encoder for numerical and categorical features."""

    def __init__(
        self,
        num_numerical_features: int,
        out_features: int,
        num_categorical_features: int | None = None,
        cardinalities: Sequence[int] | None = None,
        max_emb_dim: int = 50,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Callable[[int], nn.Module] | None = nn.BatchNorm1d,
    ) -> None:
        super().__init__()
        self.num_numerical_features = num_numerical_features
        self.embedding = EmbeddingModule(
            num_categorical_features, cardinalities, max_emb_dim
        )
        total = num_numerical_features + sum(self.embedding.embedding_dims)
        self.mlp = MLPModule(
            total, out_features, hidden_dims, activation, norm_layer, dropout
        )

    def forward(self, x_numerical: Tensor, x_categorical: Tensor) -> Tensor:
        return self.mlp(torch.cat([x_numerical, self.embedding(x_categorical)], dim=1))
