from dataclasses import dataclass
from typing import Optional, Any, Dict, List

import torch

from .dataset import Sample


def _map_tensors(x: Any, fn) -> Any:
    """
    Apply a function to all tensors in a nested structure.
    """
    if torch.is_tensor(x):
        return fn(x)
    if isinstance(x, (list, tuple)):
        t = type(x)
        return t(_map_tensors(v, fn) for v in x)
    return x


@dataclass(frozen=True)
class Batch:
    """
    Container for model features and ground truth.
    Features and Labels are always lists of tensors.
    """

    features: List[torch.Tensor]
    labels: List[torch.Tensor]

    def to(self, device: torch.device, non_blocking: bool = True) -> "Batch":
        """
        Move all tensors in the batch to the specified device.
        """
        return Batch(
            features=[t.to(device, non_blocking=non_blocking) for t in self.features],
            labels=[t.to(device, non_blocking=non_blocking) for t in self.labels],
        )

    def detach(self) -> "Batch":
        """
        Detach all tensors in the batch from the computation graph.
        """
        return Batch(
            features=[t.detach() for t in self.features],
            labels=[t.detach() for t in self.labels],
        )

    def as_dict(self) -> Dict[str, Any]:
        """
        Return the batch as a dictionary.
        """
        return {"features": self.features, "labels": self.labels}

    def __getitem__(self, key: str) -> Any:
        """
        Get an item from the batch by key ('features' or 'labels').
        """
        if key == "features":
            return self.features
        if key == "labels":
            return self.labels
        raise KeyError(key)

    def __iter__(self):
        """
        Iterate over batch keys ('features', 'labels').
        """
        yield "features"
        yield "labels"


def ensure_batch(x: Batch | Sample) -> Batch:
    """
    Ensure the input is a Batch instance.
    """
    if isinstance(x, Batch):
        return x

    return default_collate(x)


def _stack_samples(samples_list: List[Sample]) -> List[torch.Tensor]:
    """
    Stack features or labels from a list of samples.
    Args:
        samples_list: List of samples from the dataset
    Returns:
        List of stacked tensors, one per feature/label group
    """

    features, labels = zip(*samples_list)
    stacked_features = [torch.stack(tensors, dim=0) for tensors in zip(*features)]
    stacked_labels = [torch.stack(tensors, dim=0) for tensors in zip(*labels)]
    return stacked_features, stacked_labels


def default_collate(samples_list: List[Sample]) -> Batch:
    """
    Collate a list of samples into a Batch.
    Handles both single-tensor and multi-tensor features.
    """
    features, labels = _stack_samples(samples_list)

    return Batch(
        features=features,
        labels=labels,
    )
