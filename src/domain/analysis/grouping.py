import numpy as np


class RowGroups:
    """The rows of every distinct label, found with one sort instead of a mask per label.

    Each group's rows stay in their original order, so a reduction gives exactly what
    `values[labels == g]` would, at a cost of n log n plus one slice per group.
    """

    def __init__(self, labels: np.ndarray) -> None:
        self._order = np.argsort(labels, kind="stable")
        self.ids, self._starts = np.unique(labels[self._order], return_index=True)
        self._ends = np.append(self._starts[1:], len(labels))

    @property
    def sizes(self) -> np.ndarray:
        """Rows per group, in ascending label order."""
        return self._ends - self._starts

    def members(self, index: int) -> np.ndarray:
        """The rows of the `index`-th group, in their original order."""
        return self._order[self._starts[index] : self._ends[index]]

    def reduce(self, values: np.ndarray, fn=np.mean) -> np.ndarray:
        """One value per group, in ascending label order."""
        ordered = np.asarray(values)[self._order]
        return np.array([fn(ordered[s:e]) for s, e in zip(self._starts, self._ends)])

    def first(self, values: np.ndarray) -> np.ndarray:
        """The first row's value of every group."""
        return np.asarray(values)[self._order][self._starts]

    def spread(self, per_group: np.ndarray) -> np.ndarray:
        """Give every row the value of its group."""
        out = np.empty(len(self._order), dtype=per_group.dtype)
        out[self._order] = np.repeat(per_group, self.sizes)
        return out
