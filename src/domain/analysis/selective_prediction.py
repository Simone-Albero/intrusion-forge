import numpy as np


def mcp_risk(y_proba: np.ndarray) -> np.ndarray:
    """1 - max predicted class probability, per sample."""
    return 1.0 - y_proba.max(axis=1)


def risk_coverage_curve(
    score: np.ndarray,
    failure_rate: np.ndarray,
    support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Support-weighted coverage vs accuracy, admitting clusters in ascending `score`."""
    score = np.asarray(score, dtype=float)
    failure_rate = np.asarray(failure_rate, dtype=float)
    support = np.asarray(support, dtype=float)
    if score.size == 0 or support.sum() <= 0:
        return np.array([]), np.array([])

    order = np.argsort(score, kind="stable")
    s = support[order]
    correct = (1.0 - failure_rate[order]) * s
    cum_support = np.cumsum(s)
    coverage = cum_support / cum_support[-1]
    accuracy = np.cumsum(correct) / cum_support
    return coverage, accuracy


def oracle_benefit_recovered(
    score: np.ndarray,
    actual: np.ndarray,
    support: np.ndarray,
    *,
    coverage_target: float = 0.8,
) -> float:
    """Fraction of the oracle's accuracy gain over random that `score` recovers at `coverage_target`."""
    score = np.asarray(score, dtype=float)
    actual = np.asarray(actual, dtype=float)
    support = np.asarray(support, dtype=float)
    total = support.sum()
    if score.size == 0 or total <= 0:
        return float("nan")

    global_accuracy = float(1.0 - (actual * support).sum() / total)
    cov_p, acc_p = risk_coverage_curve(score, actual, support)
    cov_o, acc_o = risk_coverage_curve(actual, actual, support)
    at_target_p = float(np.interp(coverage_target, cov_p, acc_p))
    at_target_o = float(np.interp(coverage_target, cov_o, acc_o))

    gain = at_target_o - global_accuracy
    return (at_target_p - global_accuracy) / gain if gain > 1e-9 else float("nan")


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
