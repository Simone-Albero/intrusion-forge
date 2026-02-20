from collections.abc import Callable, Sequence

from torch import Tensor, nn

from . import ModelFactory
from .base import BaseModel, ModelOutput
from ..module.encoder import (
    CategoricalEncoderModule,
    NumericalEncoderModule,
    TabularEncoderModule,
)


class ComposableClassifier(BaseModel):
    """Classifier with encoder and linear head."""

    def __init__(self, encoder_module: nn.Module, head_module: nn.Module) -> None:
        super().__init__()
        self.encoder_module = encoder_module
        self.head_module = head_module

    def forward(self, x: Tensor) -> ModelOutput:
        z = self.encoder_module(x)
        return ModelOutput(logits=self.head_module(z), z=z)

    def for_loss(
        self, output: ModelOutput, target: Tensor, *args
    ) -> tuple[Tensor, ...]:
        return (output["logits"], target, *args)


class ComposableTabularClassifier(ComposableClassifier):
    """Classifier for tabular (numerical + categorical) input."""

    def forward(self, x_numerical: Tensor, x_categorical: Tensor) -> ModelOutput:
        z = self.encoder_module(x_numerical, x_categorical)
        return ModelOutput(logits=self.head_module(z), z=z)


@ModelFactory.register()
class NumericalClassifier(ComposableClassifier):
    """Classifier for numerical features."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dims: Sequence[int],
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Callable[[int], nn.Module] | None = nn.BatchNorm1d,
        bias: bool = True,
    ) -> None:
        super().__init__(
            encoder_module=NumericalEncoderModule(
                in_features,
                hidden_dims[-1],
                hidden_dims[:-1],
                dropout,
                activation,
                norm_layer,
            ),
            head_module=nn.Linear(hidden_dims[-1], num_classes, bias=bias),
        )


@ModelFactory.register()
class CategoricalClassifier(ComposableClassifier):
    """Classifier for categorical features."""

    def __init__(
        self,
        num_classes: int,
        hidden_dims: Sequence[int],
        num_features: int | None = None,
        cardinalities: Sequence[int] | None = None,
        max_emb_dim: int = 50,
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Callable[[int], nn.Module] | None = nn.BatchNorm1d,
        bias: bool = True,
    ) -> None:
        super().__init__(
            encoder_module=CategoricalEncoderModule(
                hidden_dims[-1],
                num_features,
                cardinalities,
                max_emb_dim,
                hidden_dims[:-1],
                dropout,
                activation,
                norm_layer,
            ),
            head_module=nn.Linear(hidden_dims[-1], num_classes, bias=bias),
        )


@ModelFactory.register()
class TabularClassifier(ComposableTabularClassifier):
    """Classifier for mixed tabular data (numerical + categorical)."""

    def __init__(
        self,
        num_numerical_features: int,
        num_classes: int,
        hidden_dims: Sequence[int],
        num_categorical_features: int | None = None,
        cardinalities: Sequence[int] | None = None,
        max_emb_dim: int = 50,
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Callable[[int], nn.Module] | None = nn.BatchNorm1d,
        bias: bool = True,
    ) -> None:
        super().__init__(
            encoder_module=TabularEncoderModule(
                num_numerical_features,
                hidden_dims[-1],
                num_categorical_features,
                cardinalities,
                max_emb_dim,
                hidden_dims[:-1],
                dropout,
                activation,
                norm_layer,
            ),
            head_module=nn.Linear(hidden_dims[-1], num_classes, bias=bias),
        )
