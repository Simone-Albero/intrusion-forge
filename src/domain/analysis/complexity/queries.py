from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Queries:
    """Rows sampled from each population, with their nearest reference points."""

    X: np.ndarray
    population: np.ndarray
    nbs: np.ndarray
    nb_dist: np.ndarray
