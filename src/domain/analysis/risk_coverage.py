import numpy as np


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
