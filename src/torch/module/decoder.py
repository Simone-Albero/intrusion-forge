from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn

from ..module.mlp import MLPModule


def _build_trunk(
    in_features: int,
    hidden_dims: Sequence[int],
    activation: Callable[[], nn.Module],
    norm_layer: Callable[[int], nn.Module] | None,
    dropout: float,
) -> tuple[nn.Module, int]:
    """Return (trunk, trunk_out_features)."""
    if hidden_dims:
        return (
            MLPModule(
                in_features,
                hidden_dims[-1],
                hidden_dims[:-1],
                activation,
                norm_layer,
                dropout,
            ),
            hidden_dims[-1],
        )
    return nn.Identity(), in_features


def _pad_cat_logits(logits: list[Tensor], cardinalities: list[int]) -> Tensor:
    """Pad each logit tensor to max_cardinality and stack -> [B, F, max_card]."""
    max_card = max(cardinalities)
    padded = []
    for logit, card in zip(logits, cardinalities):
        if card < max_card:
            pad = torch.full(
                (logit.size(0), max_card - card),
                float("-inf"),
                device=logit.device,
                dtype=logit.dtype,
            )
            logit = torch.cat([logit, pad], dim=1)
        padded.append(logit)
    return torch.stack(padded, dim=1)


class NumericalDecoderModule(nn.Module):
    """Decoder for numerical features using MLP."""

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
        self.mlp = MLPModule(
            in_features, out_features, hidden_dims, activation, norm_layer, dropout
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class CategoricalDecoderModule(nn.Module):
    """Decoder for categorical features using MLP trunk + per-feature classification heads."""

    def __init__(
        self,
        in_features: int,
        num_features: int | None = None,
        cardinalities: Sequence[int] | None = None,
        max_emb_dim: int = 50,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Callable[[int], nn.Module] | None = nn.BatchNorm1d,
    ) -> None:
        super().__init__()
        self.cardinalities = list(cardinalities or [max_emb_dim] * num_features)
        self.trunk, trunk_out = _build_trunk(
            in_features, list(hidden_dims), activation, norm_layer, dropout
        )
        self.heads = nn.ModuleList(nn.Linear(trunk_out, c) for c in self.cardinalities)

    def forward(self, x: Tensor) -> Tensor:
        features = self.trunk(x)
        return _pad_cat_logits(
            [head(features) for head in self.heads], self.cardinalities
        )


class TabularDecoderModule(nn.Module):
    """Unified decoder for numerical and categorical features."""

    def __init__(
        self,
        in_features: int,
        num_numerical_features: int,
        num_categorical_features: int | None = None,
        cardinalities: Sequence[int] | None = None,
        max_emb_dim: int = 50,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Callable[[int], nn.Module] | None = nn.BatchNorm1d,
    ) -> None:
        super().__init__()
        self.cardinalities = list(
            cardinalities or [max_emb_dim] * num_categorical_features
        )
        self.trunk, trunk_out = _build_trunk(
            in_features, list(hidden_dims), activation, norm_layer, dropout
        )
        self.numerical_head = nn.Linear(trunk_out, num_numerical_features)
        self.categorical_heads = nn.ModuleList(
            nn.Linear(trunk_out, c) for c in self.cardinalities
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.trunk(x)
        return (
            self.numerical_head(features),
            _pad_cat_logits(
                [head(features) for head in self.categorical_heads], self.cardinalities
            ),
        )
