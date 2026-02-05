from typing import Sequence, Optional, Callable, Tuple

import torch
from torch import nn

from ..module.mlp import MLPModule


class NumericalDecoderModule(nn.Module):
    """Decoder for numerical features using MLP."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Optional[Callable[[int], nn.Module]] = nn.BatchNorm1d,
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims)

        self.mlp = MLPModule(
            in_features=in_features,
            out_features=out_features,
            hidden_dims=hidden_dims,
            activation=activation,
            norm_layer=norm_layer,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape [batch_size, latent_dim]

        Returns:
            Tensor of shape [batch_size, out_features]
        """
        return self.mlp(x)


class CategoricalDecoderModule(nn.Module):
    """Decoder for categorical features using MLP and classification heads."""

    def __init__(
        self,
        in_features: int,
        num_features: Optional[int] = None,
        cardinalities: Optional[Sequence[int]] = None,
        max_emb_dim: int = 50,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Optional[Callable[[int], nn.Module]] = nn.BatchNorm1d,
    ) -> None:
        super().__init__()

        self.cardinalities = cardinalities or [max_emb_dim] * num_features
        hidden_dims = list(hidden_dims)

        # Common decoder trunk
        if hidden_dims:
            self.trunk = MLPModule(
                in_features=in_features,
                out_features=hidden_dims[-1],
                hidden_dims=hidden_dims[:-1],
                activation=activation,
                norm_layer=norm_layer,
                dropout=dropout,
            )
            trunk_out_features = hidden_dims[-1]
        else:
            self.trunk = nn.Identity()
            trunk_out_features = in_features

        # Separate classification head for each categorical feature
        self.heads = nn.ModuleList(
            [
                nn.Linear(trunk_out_features, cardinality)
                for cardinality in self.cardinalities
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape [batch_size, latent_dim]

        Returns:
            Tensor of shape [batch_size, num_features, max_cardinality] for reconstructed categorical logits
        """
        # Pass through common trunk
        features = self.trunk(x)

        # Generate logits for each categorical feature
        logits = [head(features) for head in self.heads]

        # Pad to max cardinality if needed
        max_cardinality = max(self.cardinalities)
        padded_logits = []
        for logit, cardinality in zip(logits, self.cardinalities):
            if cardinality < max_cardinality:
                padding = torch.full(
                    (logit.size(0), max_cardinality - cardinality),
                    float("-inf"),
                    device=logit.device,
                    dtype=logit.dtype,
                )
                logit = torch.cat([logit, padding], dim=1)
            padded_logits.append(logit)

        return torch.stack(padded_logits, dim=1)


class TabularDecoderModule(nn.Module):
    """Unified decoder for numerical and categorical features."""

    def __init__(
        self,
        in_features: int,
        num_numerical_features: int,
        num_categorical_features: Optional[int] = None,
        cardinalities: Optional[Sequence[int]] = None,
        max_emb_dim: int = 50,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Optional[Callable[[int], nn.Module]] = nn.BatchNorm1d,
    ) -> None:
        super().__init__()
        self.num_numerical_features = num_numerical_features

        self.cardinalities = cardinalities or [max_emb_dim] * num_categorical_features
        hidden_dims = list(hidden_dims)

        # Common decoder trunk
        if hidden_dims:
            self.trunk = MLPModule(
                in_features=in_features,
                out_features=hidden_dims[-1],
                hidden_dims=hidden_dims[:-1],
                activation=activation,
                norm_layer=norm_layer,
                dropout=dropout,
            )
            trunk_out_features = hidden_dims[-1]
        else:
            self.trunk = nn.Identity()
            trunk_out_features = in_features

        # Numeric features head
        self.numerical_head = nn.Linear(trunk_out_features, num_numerical_features)

        # Separate classification head for each categorical feature
        self.categorical_heads = nn.ModuleList(
            [
                nn.Linear(trunk_out_features, cardinality)
                for cardinality in self.cardinalities
            ]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Tensor of shape [batch_size, latent_dim]

        Returns:
            Tuple containing:
                - Tensor of shape [batch_size, num_numerical_features] for reconstructed numerical features
                - Tensor of shape [batch_size, num_categorical_features, max_cardinality] for reconstructed categorical logits
        """

        features = self.trunk(x)

        # Generate numerical reconstructions
        numerical_output = self.numerical_head(features)

        # Generate logits for each categorical feature
        categorical_logits = [head(features) for head in self.categorical_heads]

        # Pad to max cardinality if needed
        max_cardinality = max(self.cardinalities)
        padded_logits = []
        for logit, cardinality in zip(categorical_logits, self.cardinalities):
            if cardinality < max_cardinality:
                padding = torch.full(
                    (logit.size(0), max_cardinality - cardinality),
                    float("-inf"),
                    device=logit.device,
                    dtype=logit.dtype,
                )
                logit = torch.cat([logit, padding], dim=1)
            padded_logits.append(logit)

        categorical_output = torch.stack(padded_logits, dim=1)

        return numerical_output, categorical_output
