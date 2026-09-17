import numpy as np


def mcp_risk(y_proba: np.ndarray) -> np.ndarray:
    """1 - max predicted class probability, per sample."""
    return 1.0 - y_proba.max(axis=1)


def atc_threshold(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Average Thresholded Confidence cut: where samples below it match the misclassified count."""
    order = np.argsort(np.asarray(confidence, dtype=float), kind="stable")
    conf_sorted = np.asarray(confidence, dtype=float)[order]
    correct_sorted = np.asarray(correct).astype(bool)[order]
    fp = float((~correct_sorted).sum())
    fn = 0.0
    best_gap, thr = abs(fp - fn), conf_sorted[0]
    for i in range(conf_sorted.size):
        if correct_sorted[i]:
            fn += 1
        else:
            fp -= 1
        if abs(fp - fn) < best_gap:
            best_gap, thr = abs(fp - fn), conf_sorted[i]
    return float(thr)


def atc_cluster_risk(
    confidence: np.ndarray,
    correct: np.ndarray,
    cluster: np.ndarray,
) -> np.ndarray:
    """Cluster-level ATC: the share of a cluster below the global threshold, per sample."""
    confidence = np.asarray(confidence, dtype=float)
    below = (confidence < atc_threshold(confidence, correct)).astype(float)
    cluster = np.asarray(cluster)
    out = np.empty_like(below)
    for c in np.unique(cluster):
        m = cluster == c
        out[m] = below[m].mean()
    return out
