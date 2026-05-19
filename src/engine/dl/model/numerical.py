from collections.abc import Callable, Sequence

from torch import nn

from src.engine.dl.module.encoder import NumericalEncoderModule
from . import DLClassifierFactory
from .base import ComposableClassifier


@DLClassifierFactory.register("numerical")
class NumericalClassifier(ComposableClassifier):
    """Classifier for numerical features."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dims: Sequence[int],
        *,
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Callable[[int], nn.Module] | None = nn.BatchNorm1d,
        bias: bool = True,
    ) -> None:
        super().__init__(
            encoder_module=NumericalEncoderModule(
                in_features,
                hidden_dims[-1],
                hidden_dims=hidden_dims[:-1],
                dropout=dropout,
                activation=activation,
                norm_layer=norm_layer,
            ),
            head_module=nn.Linear(hidden_dims[-1], num_classes, bias=bias),
        )
