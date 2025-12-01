from typing import Sequence, Optional, Callable

import torch
from torch import nn

from ..module.mlp import MLPModule
from ..module.embedding import EmbeddingModule


class NumericalEncoderModule(nn.Module):
    """Encoder for numerical features using MLP."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims)

        self.mlp = MLPModule(
            in_features=in_features,
            out_features=out_features,
            hidden_dims=hidden_dims,
            activation=activation,
            norm_layer=norm_layer,
            dropout=dropout,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape [batch_size, num_numerical_features]
        Returns:
            Tensor of shape [batch_size, out_features]
        """
        return self.mlp(x)


class CategoricalEncoderModule(nn.Module):
    """Encoder for categorical features using Embedding and MLP."""

    def __init__(
        self,
        out_features: int,
        num_features: Optional[int] = None,
        cardinalities: Optional[Sequence[int]] = None,
        max_emb_dim: int = 50,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims)

        self.embedding = EmbeddingModule(
            num_features, cardinalities, max_emb_dim=max_emb_dim
        )

        embedding_features = sum(self.embedding.embedding_dims)

        self.mlp = MLPModule(
            in_features=embedding_features,
            out_features=out_features,
            hidden_dims=hidden_dims,
            activation=activation,
            norm_layer=norm_layer,
            dropout=dropout,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape [batch_size, num_categorical_features]

        Returns:
            Tensor of shape [batch_size, out_features]
        """
        embedded = self.embedding(x)
        return self.mlp(embedded)


class TabularEncoderModule(nn.Module):
    """Unified encoder for numerical and categorical features."""

    def __init__(
        self,
        num_numerical_features: int,
        out_features: int,
        num_categorical_features: Optional[int] = None,
        cardinalities: Optional[Sequence[int]] = None,
        max_emb_dim: int = 50,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.num_numerical_features = num_numerical_features

        self.embedding = EmbeddingModule(
            num_categorical_features, cardinalities, max_emb_dim=max_emb_dim
        )

        embedding_features = sum(self.embedding.embedding_dims)
        total_features = num_numerical_features + embedding_features

        self.mlp = MLPModule(
            in_features=total_features,
            out_features=out_features,
            hidden_dims=hidden_dims,
            activation=activation,
            norm_layer=norm_layer,
            dropout=dropout,
            bias=bias,
        )

    def forward(
        self, x_numerical: torch.Tensor, x_categorical: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x_numerical: Tensor of shape [batch_size, num_numerical_features]
            x_categorical: Tensor of shape [batch_size, num_categorical_features]

        Returns:
            Tensor of shape [batch_size, out_features]
        """
        embedded = self.embedding(x_categorical)
        combined = torch.cat([x_numerical, embedded], dim=1)
        return self.mlp(combined)
