from typing import Callable, Optional, Sequence

import torch
from torch import nn


class MLPModule(nn.Module):
    """Reusable MLP module (feedforward network)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dims: Sequence[int] = (),
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        """Initialize an MLP module.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            hidden_dims: Sizes of hidden layers (empty = single linear).
            activation: Activation constructor (e.g. nn.ReLU).
            norm_layer: Optional normalization layer constructor (e.g. nn.BatchNorm1d).
            dropout: Dropout probability applied after activations.
            bias: Whether to include bias in linear layers.
        """
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = in_features

        # Hidden layers
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim

        # Output layer
        layers.append(nn.Linear(prev_dim, out_features, bias=bias))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier init for Linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [batch_size, features].

        Returns:
            Output tensor of shape [batch_size, out_features].
        """
        if not torch.is_tensor(x):
            raise TypeError("Expected input as torch.Tensor")
        if x.dim() != 2:
            raise ValueError("Expected 2D tensor [batch_size, features]")

        return self.net(x)
