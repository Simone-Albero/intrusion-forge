from dataclasses import dataclass

import torch

from .dataset import Sample


@dataclass(frozen=True)
class Batch:
    """Immutable container for model inputs and ground-truth labels."""

    features: list[torch.Tensor]
    labels: list[torch.Tensor]

    def to(self, device: torch.device, *, non_blocking: bool = True) -> "Batch":
        """Move every tensor to `device`."""
        return Batch(
            features=[t.to(device, non_blocking=non_blocking) for t in self.features],
            labels=[t.to(device, non_blocking=non_blocking) for t in self.labels],
        )


def ensure_batch(x: Batch | Sample) -> Batch:
    """Return x as a Batch, collating it first if necessary."""
    return x if isinstance(x, Batch) else default_collate(x)


def default_collate(samples: list[Sample]) -> Batch:
    """Collate a list of (features, labels) samples into a Batch."""
    features, labels = zip(*samples)
    return Batch(
        features=[torch.stack(ts) for ts in zip(*features)],
        labels=[torch.stack(ts) for ts in zip(*labels)],
    )
