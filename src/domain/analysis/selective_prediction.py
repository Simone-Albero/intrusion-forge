import numpy as np
from scipy.stats import spearmanr


def mcp_risk(y_proba: np.ndarray) -> np.ndarray:
    """1 - max predicted class probability, per sample."""
    return 1.0 - y_proba.max(axis=1)


def margin_risk(y_proba: np.ndarray) -> np.ndarray:
    """1 - the gap between the top-2 predicted class probabilities, per sample."""
    top2 = np.sort(y_proba, axis=1)[:, -2:]
    return 1.0 - (top2[:, 1] - top2[:, 0])


def entropy_risk(y_proba: np.ndarray) -> np.ndarray:
    """Predictive entropy, per sample: -sum(p * log(p))."""
    p = np.clip(y_proba, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def risk_coverage_curve(
    score: np.ndarray,
    failure_rate: np.ndarray,
    support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Coverage vs accuracy as clusters are admitted in ascending `score` order.

    Both axes are support-weighted:
        coverage = Σ support(admitted) / Σ support
        accuracy = 1 − Σ(failure_rate·support, admitted) / Σ(support, admitted)
    """
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


def macro_recall_curve(
    score: np.ndarray,
    failure_rate: np.ndarray,
    support: np.ndarray,
    cluster_class: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Coverage vs macro-averaged recall on the retained set (ascending `score`).

    Each cluster is intra-class, so its `1 − failure_rate` is its class's recall
    contribution; recall is pooled per class over admitted clusters, then averaged
    over the classes still present.
    """
    score = np.asarray(score, dtype=float)
    failure_rate = np.asarray(failure_rate, dtype=float)
    support = np.asarray(support, dtype=float)
    cluster_class = np.asarray(cluster_class)
    if score.size == 0 or support.sum() <= 0:
        return np.array([]), np.array([])

    order = np.argsort(score, kind="stable")
    s = support[order]
    correct = (1.0 - failure_rate[order]) * s
    cls = cluster_class[order]

    coverage = np.cumsum(s) / s.sum()
    classes = np.unique(cls)
    cum_correct = {c: np.cumsum(np.where(cls == c, correct, 0.0)) for c in classes}
    cum_support = {c: np.cumsum(np.where(cls == c, s, 0.0)) for c in classes}

    macro_recall = np.empty(coverage.size, dtype=float)
    for i in range(coverage.size):
        recalls = [
            cum_correct[c][i] / cum_support[c][i]
            for c in classes
            if cum_support[c][i] > 0
        ]
        macro_recall[i] = float(np.mean(recalls)) if recalls else np.nan
    return coverage, macro_recall


def _curve_summary(
    cov_predictor: np.ndarray,
    val_predictor: np.ndarray,
    cov_oracle: np.ndarray,
    val_oracle: np.ndarray,
    coverage_target: float,
) -> dict:
    """AURC over coverage ∈ [0, 1] plus each curve's value at `coverage_target`.

    np.interp clamps the low-coverage tail to the best-cluster value."""
    grid = np.linspace(0.0, 1.0, 101)
    return {
        "aurc_predictor": float(np.trapezoid(np.interp(grid, cov_predictor, val_predictor), grid)),
        "aurc_oracle": float(np.trapezoid(np.interp(grid, cov_oracle, val_oracle), grid)),
        "at_target_predictor": float(np.interp(coverage_target, cov_predictor, val_predictor)),
        "at_target_oracle": float(np.interp(coverage_target, cov_oracle, val_oracle)),
    }


def _recovered(value_predictor: float, value_oracle: float, baseline: float) -> float:
    """Fraction of the oracle's gain over Random the predictor captures; NaN when the
    oracle leaves no positive headroom (legitimate on the macro-recall axis, where the
    error-ranked oracle need not dominate)."""
    gain = value_oracle - baseline
    return float((value_predictor - baseline) / gain) if gain > 1e-9 else float("nan")


def selective_prediction_metrics(
    predicted: np.ndarray,
    actual: np.ndarray,
    support: np.ndarray,
    *,
    coverage_target: float = 0.8,
) -> dict:
    """Scalar risk–coverage summary (pooled accuracy) for the failure regressor.

    Three rankings of the same (actual rate, support): predictor (by predicted rate,
    label-free), oracle (by true rate), random (flat at global accuracy).
    """
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    support = np.asarray(support, dtype=float)
    total = support.sum()
    if predicted.size == 0 or total <= 0:
        return {}

    global_accuracy = float(1.0 - (actual * support).sum() / total)
    cov_p, acc_p = risk_coverage_curve(predicted, actual, support)
    cov_o, acc_o = risk_coverage_curve(actual, actual, support)
    s = _curve_summary(cov_p, acc_p, cov_o, acc_o, coverage_target)

    return {
        "coverage_target": coverage_target,
        "global_accuracy": global_accuracy,
        "aurc_predictor": s["aurc_predictor"],
        "aurc_oracle": s["aurc_oracle"],
        "aurc_random": global_accuracy,
        "acc_at_target_predictor": s["at_target_predictor"],
        "acc_at_target_oracle": s["at_target_oracle"],
        "lift_over_random": s["at_target_predictor"] - global_accuracy,
        "oracle_benefit_recovered": _recovered(
            s["at_target_predictor"], s["at_target_oracle"], global_accuracy
        ),
    }


def bootstrap_compare(
    scores: dict[str, np.ndarray],
    actual: np.ndarray,
    support: np.ndarray,
    *,
    coverage_target: float = 0.8,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    random_state: int = 0,
) -> dict:
    """Cluster-level bootstrap CIs for spearman/lift/oracle-recovery per score, plus
    paired-difference significance of every non-reference score against the first
    (reference) entry in `scores`.

    Resamples clusters with replacement `n_resamples` times, reusing the same
    resampled indices across all scores within an iteration (paired bootstrap) so
    the difference distributions reflect paired variation, not independent noise.
    This quantifies sampling variability of the existing OOF clusters, not the
    variability of re-clustering/re-training with a different seed.
    """
    names = list(scores.keys())
    reference = names[0]
    actual = np.asarray(actual, dtype=float)
    support = np.asarray(support, dtype=float)
    n = actual.size
    rng = np.random.default_rng(random_state)

    spearman_draws = {name: np.full(n_resamples, np.nan) for name in names}
    lift_draws = {name: np.full(n_resamples, np.nan) for name in names}
    oracle_draws = {name: np.full(n_resamples, np.nan) for name in names}

    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        a, s = actual[idx], support[idx]
        total = s.sum()
        if total <= 0:
            continue
        global_accuracy = 1.0 - (a * s).sum() / total
        cov_o, acc_o = risk_coverage_curve(a, a, s)

        for name in names:
            sc = scores[name][idx]
            if np.std(a) > 0 and np.std(sc) > 0:
                spearman_draws[name][b] = spearmanr(sc, a).statistic
            cov_p, acc_p = risk_coverage_curve(sc, a, s)
            summary = _curve_summary(cov_p, acc_p, cov_o, acc_o, coverage_target)
            lift_draws[name][b] = summary["at_target_predictor"] - global_accuracy
            oracle_draws[name][b] = _recovered(
                summary["at_target_predictor"], summary["at_target_oracle"], global_accuracy
            )

    def _ci(draws: np.ndarray) -> dict:
        lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return {"mean": float(np.nanmean(draws)), "ci_low": float(lo), "ci_high": float(hi)}

    per_score = {
        name: {
            "spearman": _ci(spearman_draws[name]),
            "lift_over_random": _ci(lift_draws[name]),
            "oracle_benefit_recovered": _ci(oracle_draws[name]),
        }
        for name in names
    }

    vs_reference = {}
    for name in names:
        if name == reference:
            continue
        lift_diff = _ci(lift_draws[name] - lift_draws[reference])
        spearman_diff = _ci(spearman_draws[name] - spearman_draws[reference])
        vs_reference[name] = {
            "reference": reference,
            "lift_diff": lift_diff,
            "lift_significant": not (lift_diff["ci_low"] <= 0 <= lift_diff["ci_high"]),
            "spearman_diff": spearman_diff,
            "spearman_significant": not (
                spearman_diff["ci_low"] <= 0 <= spearman_diff["ci_high"]
            ),
        }

    return {
        "n_resamples": n_resamples,
        "alpha": alpha,
        "per_score": per_score,
        "vs_reference": vs_reference,
    }


def selective_recall_metrics(
    predicted: np.ndarray,
    actual: np.ndarray,
    support: np.ndarray,
    cluster_class: np.ndarray,
    *,
    coverage_target: float = 0.8,
) -> dict:
    """Class-balanced counterpart of `selective_prediction_metrics`: macro-recall
    on the retained set instead of pooled accuracy.

    The oracle still ranks by true *error* rate, so on this axis it need not dominate —
    a dip below Random signals that error-greedy rejection costs class balance.
    """
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    support = np.asarray(support, dtype=float)
    cluster_class = np.asarray(cluster_class)
    if predicted.size == 0 or support.sum() <= 0:
        return {}

    cov_p, rec_p = macro_recall_curve(predicted, actual, support, cluster_class)
    cov_o, rec_o = macro_recall_curve(actual, actual, support, cluster_class)
    global_macro_recall = float(rec_p[-1])
    s = _curve_summary(cov_p, rec_p, cov_o, rec_o, coverage_target)

    return {
        "coverage_target": coverage_target,
        "global_macro_recall": global_macro_recall,
        "aurc_predictor": s["aurc_predictor"],
        "aurc_oracle": s["aurc_oracle"],
        "aurc_random": global_macro_recall,
        "recall_at_target_predictor": s["at_target_predictor"],
        "recall_at_target_oracle": s["at_target_oracle"],
        "lift_over_random": s["at_target_predictor"] - global_macro_recall,
        "oracle_benefit_recovered": _recovered(
            s["at_target_predictor"], s["at_target_oracle"], global_macro_recall
        ),
    }
