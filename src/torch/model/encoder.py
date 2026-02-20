from collections.abc import Callable, Sequence

from torch import Tensor, nn

from . import ModelFactory
from .base import BaseModel, ModelOutput
from ..module.encoder import (
    CategoricalEncoderModule,
    NumericalEncoderModule,
    TabularEncoderModule,
)


@ModelFactory.register()
class NumericalEncoder(BaseModel):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Callable[[int], nn.Module] | None = nn.BatchNorm1d,
    ) -> None:
        super().__init__()
        self.encoder_module = NumericalEncoderModule(
            in_features, out_features, hidden_dims, dropout, activation, norm_layer
        )

    def forward(self, x: Tensor) -> ModelOutput:
        return ModelOutput(z=self.encoder_module(x))

    def for_loss(self, output: ModelOutput, target: Tensor) -> tuple[Tensor, Tensor]:
        return output["z"], target


@ModelFactory.register()
class CategoricalEncoder(BaseModel):

    def __init__(
        self,
        out_features: int,
        num_features: int | None = None,
        cardinalities: Sequence[int] | None = None,
        max_emb_dim: int = 50,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Callable[[int], nn.Module] | None = nn.BatchNorm1d,
    ) -> None:
        super().__init__()
        self.encoder_module = CategoricalEncoderModule(
            out_features,
            num_features,
            cardinalities,
            max_emb_dim,
            hidden_dims,
            dropout,
            activation,
            norm_layer,
        )

    def forward(self, x: Tensor) -> ModelOutput:
        return ModelOutput(z=self.encoder_module(x))

    def for_loss(self, output: ModelOutput, target: Tensor) -> tuple[Tensor, Tensor]:
        return output["z"], target


@ModelFactory.register()
class TabularEncoder(BaseModel):

    def __init__(
        self,
        num_numerical_features: int,
        out_features: int,
        num_categorical_features: int | None = None,
        cardinalities: Sequence[int] | None = None,
        max_emb_dim: int = 50,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Callable[[int], nn.Module] | None = nn.BatchNorm1d,
    ) -> None:
        super().__init__()
        self.encoder_module = TabularEncoderModule(
            num_numerical_features,
            out_features,
            num_categorical_features,
            cardinalities,
            max_emb_dim,
            hidden_dims,
            dropout,
            activation,
            norm_layer,
        )

    def forward(self, x_numerical: Tensor, x_categorical: Tensor) -> ModelOutput:
        return ModelOutput(z=self.encoder_module(x_numerical, x_categorical))

    def for_loss(self, output: ModelOutput, target: Tensor) -> tuple[Tensor, Tensor]:
        return output["z"], target
