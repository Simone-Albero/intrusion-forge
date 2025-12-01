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
    Features and Targets are always lists of tensors.
    """

    features: List[torch.Tensor]
    targets: Optional[List[torch.Tensor]] = None

    def to(self, device: torch.device, non_blocking: bool = True) -> "Batch":
        """
        Move all tensors in the batch to the specified device.
        """
        return Batch(
            features=[t.to(device, non_blocking=non_blocking) for t in self.features],
            targets=(
                None
                if self.targets is None
                else [t.to(device, non_blocking=non_blocking) for t in self.targets]
            ),
        )

    def detach(self) -> "Batch":
        """
        Detach all tensors in the batch from the computation graph.
        """
        return Batch(
            features=[t.detach() for t in self.features],
            targets=(
                None if self.targets is None else [t.detach() for t in self.targets]
            ),
        )

    def as_dict(self) -> Dict[str, Any]:
        """
        Return the batch as a dictionary.
        """
        return {"samples": self.features, "ground_truth": self.targets}

    def __getitem__(self, key: str) -> Any:
        """
        Get an item from the batch by key ('samples' or 'ground_truth').
        """
        if key == "samples":
            return self.features
        if key == "ground_truth":
            return self.targets
        raise KeyError(key)

    def __iter__(self):
        """
        Iterate over batch keys ('samples', 'ground_truth').
        """
        yield "samples"
        yield "ground_truth"


def ensure_batch(x: Batch | Sample) -> Batch:
    """
    Ensure the input is a Batch instance.
    """
    if isinstance(x, Batch):
        return x

    return default_collate(x)


def _stack_samples(samples_list: List[Sample], index: int) -> List[torch.Tensor]:
    """
    Stack a group of tensors from samples.

    Args:
        samples_list: List of samples from the dataset
        index: Index to extract (0 for features, 1 for labels)

    Returns:
        List of stacked tensors, one per feature/label group
    """
    first_group = samples_list[0][index]

    # Case 1: Group is a list of tensors (e.g., MixedTabularDataset)
    if isinstance(first_group, list):
        num_groups = len(first_group)
        return [
            torch.stack([s[index][i] for s in samples_list], dim=0)
            for i in range(num_groups)
        ]

    # Case 2: Group is a single tensor (e.g., TensorDataset)
    return [torch.stack([s[index] for s in samples_list], dim=0)]


def default_collate(samples_list: List[Sample]) -> Batch:
    """
    Collate a list of samples into a Batch.
    Handles both single-tensor and multi-tensor features.
    """
    # Stack features
    samples = _stack_samples(samples_list, index=0)

    # Stack targets if present
    targets = None
    if samples_list[0][1] is not None:
        targets = _stack_samples(samples_list, index=1)

    return Batch(
        features=samples,
        targets=targets,
    )
