from typing import Tuple, Dict, Sequence

from torch import nn, Tensor
import torch
import numpy as np


class ModelOutput(dict):
    """Model output container. Extends dict to enforce Tensor values."""

    def __init__(self, data: Dict[str, Tensor] = None, **kwargs):
        if data is None:
            data = kwargs
        elif kwargs:
            raise ValueError("Cannot use both 'data' argument and keyword arguments")

        for key, value in data.items():
            if not isinstance(value, Tensor):
                raise TypeError(
                    f"ModelOutput value for key '{key}' must be a Tensor, got {type(value)}"
                )
        super().__init__(data)

    def __setitem__(self, key: str, value: Tensor):
        if not isinstance(value, Tensor):
            raise TypeError(
                f"ModelOutput value for key '{key}' must be a Tensor, got {type(value)}"
            )
        super().__setitem__(key, value)

    def detach(self) -> "ModelOutput":
        detached = {}
        for k, v in self.items():
            detached[k] = v.detach()
        return ModelOutput(detached)

    def to(self, device: torch.device, non_blocking: bool = True) -> "ModelOutput":
        moved = {}
        for k, v in self.items():
            moved[k] = v.to(device, non_blocking=non_blocking)
        return ModelOutput(moved)

    def numpy(self) -> Dict[str, np.ndarray]:
        numpy_dict = {}
        for k, v in self.items():
            numpy_dict[k] = v.cpu().numpy()
        return numpy_dict


def cat_model_outputs(
    outputs: Sequence[ModelOutput], dim: int = 0
) -> Dict[str, Tensor]:
    """Concatenate a sequence of ModelOutput along a specified dimension.

    Args:
        outputs: Sequence of ModelOutput instances to concatenate.
        dim: Dimension along which to concatenate.

    Returns:
        A single ModelOutput with concatenated tensors.
    """
    if not outputs:
        raise ValueError("The outputs sequence is empty.")

    concatenated = {}
    keys = outputs[0].keys()
    for key in keys:
        concatenated[key] = torch.cat([output[key] for output in outputs], dim=dim)

    return ModelOutput(concatenated)


class BaseModel(nn.Module):
    """
    Base class for models. Defines the interface for forward and loss_inputs methods.
    """

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Forward pass. Must be implemented by subclasses.
        Args:
            x: Input tensor of shape [batch_size, num_features].
        Returns:
            ModelOutput: NamedTuple, expected to contain at least 'logits'.
        """
        raise NotImplementedError

    def for_loss(
        self,
        output: ModelOutput,
        target: torch.Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Prepares arguments for the loss function. Default: pred=output['logits'].
        Override if your model/loss requires different fields.
        """
        return output.logits, target
