import numpy as np

from src.domain.analysis.grouping import RowGroups


def mcp_risk(y_proba: np.ndarray) -> np.ndarray:
    """1 - max predicted class probability, per sample."""
    return 1.0 - y_proba.max(axis=1)


def atc_threshold(confidence: np.ndarray, correct: np.ndarray) -> float:
    """ATC cut: the confidence below which as many samples fall as were misjudged."""
    conf = np.sort(np.asarray(confidence, dtype=float))
    n_errors = int((~np.asarray(correct).astype(bool)).sum())
    # Only distinct values are cuts: `confidence < cut` cannot split a run of ties, so a
    # cut placed inside one would flag fewer samples than it was chosen for.
    cuts = np.append(np.unique(conf), np.inf)
    below = np.searchsorted(conf, cuts, side="left")
    return float(cuts[np.argmin(np.abs(below - n_errors))])


def atc_region_risk(
    confidence: np.ndarray, region: np.ndarray, *, threshold: float
) -> np.ndarray:
    """Region-level ATC: the share of a region below the threshold, per sample."""
    below = (np.asarray(confidence, dtype=float) < threshold).astype(float)
    groups = RowGroups(np.asarray(region))
    return groups.spread(groups.reduce(below))
