from typing import Sequence, Optional, Callable, Tuple

import torch
from torch import nn

from .base import BaseModel, ModelOutput
from ..module.mlp import MLPModule
from . import ModelFactory


class ComposableContrastiveClassifier(BaseModel):
    """Composable contrastive classifier with encoder and head modules."""

    def __init__(
        self,
        encoder_module: nn.Module,
        classification_head_module: nn.Module,
        contrastive_head_module: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder_module = encoder_module
        self.classification_head_module = classification_head_module
        self.contrastive_head_module = contrastive_head_module

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Model output with 'logits' and 'z'
        """
        z = self.encoder_module(x)
        logits = self.classification_head_module(z)
        contrastive_logits = self.contrastive_head_module(z)

        return ModelOutput(logits=logits, z=z, contrastive_logits=contrastive_logits)

    def for_loss(
        self,
        output: ModelOutput,
        target: torch.Tensor,
        *args,
    ) -> Tuple[torch.Tensor, ...]:
        """Returns data for loss computation.

        For standard losses: returns (logits, target)
        For contrastive losses: returns (z, target, clusters, ...)
        """
        if args:
            return (
                output["contrastive_logits"],
                output["logits"],
                target,
                *args,
            )
        return (output["logits"], target)


class ComposableTabularContrastiveClassifier(BaseModel):
    """Composable contrastive classifier for tabular data with encoder and head modules."""

    def __init__(
        self,
        encoder_module: nn.Module,
        classification_head_module: nn.Module,
        contrastive_head_module: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder_module = encoder_module
        self.classification_head_module = classification_head_module
        self.contrastive_head_module = contrastive_head_module

    def forward(
        self, x_numerical: torch.Tensor, x_categorical: torch.Tensor
    ) -> ModelOutput:
        """Forward pass.

        Args:
            x_numerical: Tensor of shape [batch_size, num_numerical_features]
            x_categorical: Tensor of shape [batch_size, num_categorical_features]

        Returns:
            Model output with 'logits' and 'z'
        """
        z = self.encoder_module(x_numerical, x_categorical)
        logits = self.classification_head_module(z)
        contrastive_logits = self.contrastive_head_module(z)

        return ModelOutput(logits=logits, z=z, contrastive_logits=contrastive_logits)

    def for_loss(
        self,
        output: ModelOutput,
        target: torch.Tensor,
        *args,
    ) -> Tuple[torch.Tensor, ...]:
        """Returns data for loss computation.

        For standard losses: returns (logits, target)
        For contrastive losses: returns (z, target, clusters, ...)
        """
        if args:
            return (
                output["contrastive_logits"],
                output["logits"],
                target,
                *args,
            )
        return (output["logits"], target)


@ModelFactory.register()
class NumericalContrastiveClassifier(ComposableContrastiveClassifier):
    """Classifier for numerical features."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dims: Sequence[int],
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Optional[Callable[[int], nn.Module]] = nn.LayerNorm,
        bias: bool = True,
    ) -> None:
        from ..module.encoder import NumericalEncoderModule

        encoder_module = NumericalEncoderModule(
            in_features=in_features,
            out_features=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
            dropout=dropout,
            activation=activation,
            norm_layer=norm_layer,
        )
        bottleneck_dim = hidden_dims[-1]

        classification_head_module = MLPModule(
            in_features=bottleneck_dim,
            out_features=num_classes,
            hidden_dims=[bottleneck_dim, bottleneck_dim // 2, bottleneck_dim // 2],
            dropout=dropout,
            activation=activation,
            norm_layer=norm_layer,
        )
        contrastive_head_module = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim // 2, bias=bias),
            nn.LayerNorm(bottleneck_dim // 2),
        )

        super().__init__(
            encoder_module=encoder_module,
            classification_head_module=classification_head_module,
            contrastive_head_module=contrastive_head_module,
        )
