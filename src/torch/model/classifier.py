from typing import Sequence, Optional, Callable, Tuple

import torch
from torch import nn

from .base import BaseModel, ModelOutput
from ..module.encoder import TabularEncoderModule
from . import ModelFactory


class ComposableTabularClassifier(BaseModel):
    """Composable classifier for tabular data with encoder and head modules."""

    def __init__(
        self,
        encoder_module: nn.Module,
        head_module: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder_module = encoder_module
        self.head_module = head_module

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
        logits = self.head_module(z)

        return ModelOutput(logits=logits, z=z)

    def for_loss(
        self,
        output: ModelOutput,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns logits and target for loss computation."""
        return output["logits"], target


@ModelFactory.register()
class TabularClassifier(ComposableTabularClassifier):
    """Classifier for mixed tabular data (numerical + categorical features)."""

    def __init__(
        self,
        num_numerical_features: int,
        num_classes: int,
        hidden_dims: Sequence[int],
        num_categorical_features: Optional[int] = None,
        cardinalities: Optional[Sequence[int]] = None,
        max_emb_dim: int = 50,
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        bias: bool = True,
    ) -> None:
        encoder_module = TabularEncoderModule(
            num_numerical_features=num_numerical_features,
            out_features=hidden_dims[-1],
            num_categorical_features=num_categorical_features,
            cardinalities=cardinalities,
            max_emb_dim=max_emb_dim,
            hidden_dims=hidden_dims[:-1],
            dropout=dropout,
            activation=activation,
            norm_layer=norm_layer,
        )
        head_module = nn.Linear(hidden_dims[-1], num_classes, bias=bias)

        super().__init__(
            encoder_module=encoder_module,
            head_module=head_module,
        )
