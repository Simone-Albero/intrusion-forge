from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MeasuredSample:
    """Rows sampled from each population, with their nearest graph nodes."""

    X: np.ndarray
    population: np.ndarray
    neighbors: np.ndarray
    neighbor_dist: np.ndarray
