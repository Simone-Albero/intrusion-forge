from collections.abc import Callable, Sequence

from torch import nn

from src.engine.dl.module.encoder import CategoricalEncoderModule
from . import DLClassifierFactory
from .base import ComposableClassifier


@DLClassifierFactory.register("categorical")
class CategoricalClassifier(ComposableClassifier):
    """Classifier for categorical features."""

    def __init__(
        self,
        num_classes: int,
        hidden_dims: Sequence[int],
        *,
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
                num_features=num_features,
                cardinalities=cardinalities,
                max_emb_dim=max_emb_dim,
                hidden_dims=hidden_dims[:-1],
                dropout=dropout,
                activation=activation,
                norm_layer=norm_layer,
            ),
            head_module=nn.Linear(hidden_dims[-1], num_classes, bias=bias),
        )
