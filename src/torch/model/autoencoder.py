from typing import Sequence, Optional, Callable, Tuple

import torch
from torch import nn

from .base import BaseModel, ModelOutput
from ..module.encoder import (
    NumericalEncoderModule,
    CategoricalEncoderModule,
    TabularEncoderModule,
)
from ..module.decoder import (
    NumericalDecoderModule,
    CategoricalDecoderModule,
    TabularDecoderModule,
)
from . import ModelFactory


class ComposableAutoencoder(BaseModel):
    """Composable Autoencoder with custom encoder and decoder modules."""

    def __init__(
        self,
        encoder_module: nn.Module,
        decoder_module: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder_module = encoder_module
        self.decoder_module = decoder_module

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Forward pass.
        Args:
            x: Input tensor
        Returns:
            Model output with 'x_recon' and 'z'
        """
        z = self.encoder_module(x)
        x_recon = self.decoder_module(z)
        return ModelOutput(x_recon=x_recon, z=z)

    def for_loss(
        self,
        output: ModelOutput,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns reconstructed features and targets for loss computation."""
        return output["x_recon"], target


class ComposableTabularAutoencoder(BaseModel):
    """Composable Tabular Autoencoder with custom encoder and decoder modules."""

    def __init__(
        self,
        encoder_module: nn.Module,
        decoder_module: nn.Module,
        noise_factor: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder_module = encoder_module
        self.decoder_module = decoder_module
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
        """Forward pass.
        Args:
            x_numerical: Tensor of shape [batch_size, num_numerical_features]
            x_categorical: Tensor of shape [batch_size, num_categorical_features]
        Returns:
            Model output with 'x_numerical_recon', 'x_categorical_recon', and 'z'
        """
        if self.noise_factor > 0:
            x_numerical, x_categorical = self._augment(
                x_numerical, x_categorical, self.noise_factor
            )

        z = self.encoder_module(x_numerical, x_categorical)
        x_numerical_recon, x_categorical_recon = self.decoder_module(z)
        return ModelOutput(
            x_numerical_recon=x_numerical_recon,
            x_categorical_recon=x_categorical_recon,
            z=z,
        )

    def for_loss(
        self,
        output: ModelOutput,
        target_numerical: torch.Tensor,
        target_categorical: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns reconstructed features and targets for loss computation."""
        return (
            output["x_numerical_recon"],
            output["x_categorical_recon"],
            target_numerical,
            target_categorical,
        )


@ModelFactory.register()
class NumericalAutoencoder(ComposableAutoencoder):

    def __init__(
        self,
        in_features: int,
        latent_dim: int,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Optional[Callable[[int], nn.Module]] = nn.BatchNorm1d,
    ) -> None:
        hidden_dims = list(hidden_dims)

        encoder_module = NumericalEncoderModule(
            in_features=in_features,
            out_features=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            norm_layer=norm_layer,
        )

        decoder_module = NumericalDecoderModule(
            in_features=latent_dim,
            out_features=in_features,
            hidden_dims=hidden_dims[::-1],
            dropout=dropout,
            activation=activation,
            norm_layer=norm_layer,
        )

        super().__init__(encoder_module=encoder_module, decoder_module=decoder_module)


@ModelFactory.register()
class CategoricalAutoencoder(ComposableAutoencoder):
    def __init__(
        self,
        latent_dim: int,
        num_features: Optional[int] = None,
        cardinalities: Optional[Sequence[int]] = None,
        max_emb_dim: int = 50,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Optional[Callable[[int], nn.Module]] = nn.BatchNorm1d,
    ) -> None:
        hidden_dims = list(hidden_dims)

        encoder_module = CategoricalEncoderModule(
            out_features=latent_dim,
            num_features=num_features,
            cardinalities=cardinalities,
            max_emb_dim=max_emb_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            norm_layer=norm_layer,
        )

        decoder_module = CategoricalDecoderModule(
            in_features=latent_dim,
            num_features=num_features,
            cardinalities=cardinalities,
            max_emb_dim=max_emb_dim,
            hidden_dims=hidden_dims[::-1],
            dropout=dropout,
            activation=activation,
            norm_layer=norm_layer,
        )

        super().__init__(encoder_module=encoder_module, decoder_module=decoder_module)


@ModelFactory.register()
class TabularAutoencoder(ComposableTabularAutoencoder):

    def __init__(
        self,
        num_numerical_features: int,
        latent_dim: int,
        num_categorical_features: Optional[int] = None,
        cardinalities: Optional[Sequence[int]] = None,
        max_emb_dim: int = 50,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Optional[Callable[[int], nn.Module]] = nn.BatchNorm1d,
        noise_factor: float = 0.0,
    ) -> None:
        hidden_dims = list(hidden_dims)

        encoder_module = TabularEncoderModule(
            num_numerical_features=num_numerical_features,
            out_features=latent_dim,
            num_categorical_features=num_categorical_features,
            cardinalities=cardinalities,
            max_emb_dim=max_emb_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            norm_layer=norm_layer,
        )

        decoder_module = TabularDecoderModule(
            in_features=latent_dim,
            num_numerical_features=num_numerical_features,
            num_categorical_features=num_categorical_features,
            cardinalities=cardinalities,
            max_emb_dim=max_emb_dim,
            hidden_dims=hidden_dims[::-1],
            dropout=dropout,
            activation=activation,
            norm_layer=norm_layer,
        )

        super().__init__(
            encoder_module=encoder_module,
            decoder_module=decoder_module,
            noise_factor=noise_factor,
        )
