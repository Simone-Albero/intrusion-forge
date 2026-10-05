from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class KdnNodes:
    """The graph's nodes (a uniform sample of train) in the clustering space, where kDN
    counts neighbours."""

    X: np.ndarray
    y: np.ndarray
    row_ids: np.ndarray


def compute_kdn(
    X: np.ndarray, *, row_ids: np.ndarray, label: object, nodes: KdnNodes, k: int
) -> np.ndarray:
    """Share of each row's k nearest nodes, itself excluded, of another class."""
    effective_k = min(k, nodes.X.shape[0] - 1)
    if effective_k < 1:
        raise ValueError(
            f"compute_kdn: {nodes.X.shape[0]} node(s) leave no "
            "neighbour once a row's own copy is excluded."
        )
    index = NearestNeighbors(n_neighbors=effective_k + 1).fit(nodes.X)
    _, neighbor_positions = index.kneighbors(X)
    neighbor_ids = nodes.row_ids[neighbor_positions]
    neighbor_labels = nodes.y[neighbor_positions]

    self_mask = neighbor_ids == row_ids[:, np.newaxis]
    has_self = self_mask.any(axis=1)
    # Drop the row's own node when it has one, else the farthest neighbour,
    # so every row keeps exactly `effective_k` votes.
    drop = np.where(has_self, self_mask.argmax(axis=1), effective_k)
    keep = np.ones((len(X), effective_k + 1), dtype=bool)
    keep[np.arange(len(X)), drop] = False

    kept_labels = neighbor_labels[keep].reshape(len(X), effective_k)
    return (kept_labels != label).sum(axis=1) / effective_k
