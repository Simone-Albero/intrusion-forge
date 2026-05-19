from collections.abc import Callable, Sequence

from torch import nn

from src.engine.dl.module.encoder import TabularEncoderModule
from . import DLClassifierFactory
from .base import ComposableTabularClassifier


@DLClassifierFactory.register("tabular")
class TabularClassifier(ComposableTabularClassifier):
    """Classifier for mixed tabular data (numerical + categorical)."""

    def __init__(
        self,
        num_numerical_features: int,
        num_classes: int,
        hidden_dims: Sequence[int],
        *,
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
                num_categorical_features=num_categorical_features,
                cardinalities=cardinalities,
                max_emb_dim=max_emb_dim,
                hidden_dims=hidden_dims[:-1],
                dropout=dropout,
                activation=activation,
                norm_layer=norm_layer,
            ),
            head_module=nn.Linear(hidden_dims[-1], num_classes, bias=bias),
        )
