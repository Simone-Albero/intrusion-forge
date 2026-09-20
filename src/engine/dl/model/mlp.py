from collections.abc import Callable, Sequence

from torch import nn

from src.engine.dl.module.encoder import TabularEncoderModule

from . import DLClassifierFactory
from .base import ComposableClassifier


@DLClassifierFactory.register("mlp")
class MLPClassifier(ComposableClassifier):
    """Multi-layer perceptron over numerical features and categorical embeddings."""

    def __init__(
        self,
        num_numerical_features: int,
        num_classes: int,
        hidden_dims: Sequence[int],
        *,
        cardinalities: Sequence[int] = (),
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
                cardinalities=cardinalities,
                max_emb_dim=max_emb_dim,
                hidden_dims=hidden_dims[:-1],
                dropout=dropout,
                activation=activation,
                norm_layer=norm_layer,
            ),
            head_module=nn.Linear(hidden_dims[-1], num_classes, bias=bias),
        )
