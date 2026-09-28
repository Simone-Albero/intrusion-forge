from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class KdnReference:
    """Uniform sample of the train split, in the clustering space, kDN counts neighbours in."""

    X: np.ndarray
    y: np.ndarray
    ids: np.ndarray


def compute_kdn(
    X: np.ndarray, *, ids: np.ndarray, label: object, reference: KdnReference, k: int
) -> np.ndarray:
    """Share of each row's k nearest reference rows, itself excluded, of another class."""
    effective_k = min(k, reference.X.shape[0] - 1)
    if effective_k < 1:
        raise ValueError(
            f"compute_kdn: reference of {reference.X.shape[0]} row(s) leaves no "
            "neighbour once a row's own copy is excluded."
        )
    nn = NearestNeighbors(n_neighbors=effective_k + 1).fit(reference.X)
    _, neighbor_pos = nn.kneighbors(X)
    neighbor_ids = reference.ids[neighbor_pos]
    neighbor_labels = reference.y[neighbor_pos]

    self_mask = neighbor_ids == ids[:, np.newaxis]
    has_self = self_mask.any(axis=1)
    # Drop the row's own reference copy when it has one, else the farthest neighbour,
    # so every row keeps exactly `effective_k` votes.
    drop = np.where(has_self, self_mask.argmax(axis=1), effective_k)
    keep = np.ones((len(X), effective_k + 1), dtype=bool)
    keep[np.arange(len(X)), drop] = False

    kept_labels = neighbor_labels[keep].reshape(len(X), effective_k)
    return (kept_labels != label).sum(axis=1) / effective_k
