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
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
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
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        bias: bool = True,
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
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        bias: bool = True,
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
            bias=bias,
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


@ModelFactory.register()
class ContrastiveTabularEncoder(BaseModel):

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
        noise_factor: float = 0.1,
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
            bias=bias,
        )
        self.noise_factor = noise_factor

    def _augment(
        self,
        x_numerical: torch.Tensor,
        x_categorical: torch.Tensor,
        noise_factor: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Data augmentation for contrastive learning.
        Args:
            x_numerical: Tensor of shape [batch_size, num_numerical_features]
            x_categorical: Tensor of shape [batch_size, num_categorical_features]
        Returns:
            Augmented numerical and categorical tensors.
        """
        # Gaussian noise to numerical features
        noise = torch.randn_like(x_numerical) * noise_factor
        x_numerical_aug = x_numerical + noise

        # Randomly mask some categorical features
        x_categorical_aug = x_categorical.clone()
        if x_categorical.numel() > 0:
            mask = torch.rand_like(x_categorical.float()) < noise_factor
            x_categorical_aug[mask] = 0  # Assuming 0 is the 'unknown' category

        return x_numerical_aug, x_categorical_aug

    def forward(
        self, x_numerical: torch.Tensor, x_categorical: torch.Tensor
    ) -> ModelOutput:
        """Forward pass with contrastive augmentations.
        Args:
            x_numerical: Tensor of shape [batch_size, num_numerical_features]
            x_categorical: Tensor of shape [batch_size, num_categorical_features]
        Returns:
            Model output with 'z1' and 'z2'
        """
        x1_numerical, x1_categorical = self._augment(
            x_numerical, x_categorical, self.noise_factor
        )
        x2_numerical, x2_categorical = self._augment(
            x_numerical, x_categorical, self.noise_factor
        )
        z1 = self.encoder_module(x1_numerical, x1_categorical)
        z2 = self.encoder_module(x2_numerical, x2_categorical)

        return ModelOutput(
            z1=z1,
            z2=z2,
        )

    def for_loss(
        self,
        output: ModelOutput,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns augmented z and target for loss computation."""

        return (
            output["z1"],
            output["z2"],
            target,
        )
