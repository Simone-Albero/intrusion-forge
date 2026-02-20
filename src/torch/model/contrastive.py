from __future__ import annotations

from collections.abc import Callable, Sequence

from torch import Tensor, nn

from . import ModelFactory
from .base import BaseModel, ModelOutput
from ..module.encoder import NumericalEncoderModule
from ..module.mlp import MLPModule


class ComposableContrastiveClassifier(BaseModel):
    """Contrastive classifier with separate encoder, classification, and contrastive heads."""

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

    def _encode_and_project(self, z: Tensor) -> ModelOutput:
        return ModelOutput(
            logits=self.classification_head_module(z),
            z=z,
            contrastive_logits=self.contrastive_head_module(z),
        )

    def forward(self, x: Tensor) -> ModelOutput:
        return self._encode_and_project(self.encoder_module(x))

    def for_loss(self, output: ModelOutput, target: Tensor, *args) -> tuple[Tensor, ...]:
        if args:
            return (output["contrastive_logits"], output["logits"], target, *args)
        return (output["logits"], target)


class ComposableTabularContrastiveClassifier(ComposableContrastiveClassifier):
    """Contrastive classifier for tabular (numerical + categorical) input."""

    def forward(self, x_numerical: Tensor, x_categorical: Tensor) -> ModelOutput:
        return self._encode_and_project(self.encoder_module(x_numerical, x_categorical))


@ModelFactory.register()
class NumericalContrastiveClassifier(ComposableContrastiveClassifier):
    """Contrastive classifier for numerical features."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dims: Sequence[int],
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Callable[[int], nn.Module] | None = nn.LayerNorm,
    ) -> None:
        d = hidden_dims[-1]
        super().__init__(
            encoder_module=NumericalEncoderModule(in_features, d, hidden_dims[:-1], dropout, activation, norm_layer),
            classification_head_module=MLPModule(d, num_classes, [d, d // 2, d // 2], activation, norm_layer, dropout),
            contrastive_head_module=nn.Sequential(nn.Linear(d, d // 2), nn.LayerNorm(d // 2)),
        )
