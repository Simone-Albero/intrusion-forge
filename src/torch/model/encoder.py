from typing import Sequence, Optional, Callable, Tuple

import torch
from torch import nn

from .base import BaseModel, ModelOutput
from ..module.encoder import (
    NumericalEncoderModule,
    CategoricalEncoderModule,
    TabularEncoderModule,
)
from . import ModelFactory


@ModelFactory.register()
class NumericalEncoder(BaseModel):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Optional[Callable[[int], nn.Module]] = nn.BatchNorm1d,
        bias: bool = True,
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims)

        self.encoder_module = NumericalEncoderModule(
            in_features=in_features,
            out_features=out_features,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            norm_layer=norm_layer,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Forward pass.
        Args:
            x: Input tensor
        Returns:
            Model output with 'z'
        """
        z = self.encoder_module(x)
        return ModelOutput(z=z)

    def for_loss(
        self,
        output: ModelOutput,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns z and target for loss computation."""
        return output["z"], target


@ModelFactory.register()
class CategoricalEncoder(BaseModel):
    def __init__(
        self,
        out_features: int,
        num_features: Optional[int] = None,
        cardinalities: Optional[Sequence[int]] = None,
        max_emb_dim: int = 50,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Optional[Callable[[int], nn.Module]] = nn.BatchNorm1d,
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims)

        self.encoder_module = CategoricalEncoderModule(
            out_features=out_features,
            num_features=num_features,
            cardinalities=cardinalities,
            max_emb_dim=max_emb_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            norm_layer=norm_layer,
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Forward pass.
        Args:
            x: Input tensor
        Returns:
            Model output with 'z'
        """
        z = self.encoder_module(x)
        return ModelOutput(z=z)

    def for_loss(
        self,
        output: ModelOutput,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns z and target for loss computation."""
        return output["z"], target


@ModelFactory.register()
class TabularEncoder(BaseModel):

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
        norm_layer: Optional[Callable[[int], nn.Module]] = nn.BatchNorm1d,
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims)

        self.encoder_module = TabularEncoderModule(
            num_numerical_features=num_numerical_features,
            out_features=out_features,
            num_categorical_features=num_categorical_features,
            cardinalities=cardinalities,
            max_emb_dim=max_emb_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            norm_layer=norm_layer,
        )

    def forward(
        self, x_numerical: torch.Tensor, x_categorical: torch.Tensor
    ) -> ModelOutput:
        """Forward pass.
        Args:
            x_numerical: Tensor of shape [batch_size, num_numerical_features]
            x_categorical: Tensor of shape [batch_size, num_categorical_features]
        Returns:
            Model output with 'z'
        """
        z = self.encoder_module(x_numerical, x_categorical)
        return ModelOutput(z=z)

    def for_loss(
        self,
        output: ModelOutput,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns z and target for loss computation."""
        return output["z"], target
