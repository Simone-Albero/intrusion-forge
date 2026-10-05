import logging

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr

from src.domain.analysis.calibration import fit_platt
from src.domain.analysis.grouping import RowsBy
from src.domain.analysis.risk_coverage import oracle_benefit_recovered

logger = logging.getLogger(__name__)

RATE_BASELINES = ("atc", "mcp", "train_empiric", "val_empiric")
CALIBRATED = {"atc_cal": "atc", "mcp_cal": "mcp", "train_empiric_cal": "train_empiric"}
# Each combo averages the regressor's rate with the rate of the variant it names, a
# calibrated one wherever the raw score is not a rate.
COMBOS = {
    "atc_combo": "atc_cal",
    "mcp_combo": "mcp_cal",
    "train_empiric_combo": "train_empiric_cal",
    "val_empiric_combo": "val_empiric",
    "sample_regressor_combo": "sample_regressor",
}
VARIANTS = ("regressor", "sample_regressor", *RATE_BASELINES, *CALIBRATED, *COMBOS)

SIZE_VARIANTS = (
    "regressor",
    "sample_regressor",
    "train_empiric",
    "val_empiric",
    "mcp",
    "atc",
)
# The by-size errors add each calibrated variant beside its raw one; a positive slope
# keeps the raw one's rho, so the rho figures leave them out.
SIZE_ERROR_VARIANTS = (
    "regressor",
    "sample_regressor",
    "train_empiric",
    "train_empiric_cal",
    "val_empiric",
    "mcp",
    "mcp_cal",
    "atc",
    "atc_cal",
)
SIZE_COMBO_VARIANTS = ("regressor", *COMBOS)


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman's rho, NaN when either side is constant."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.std(x) > 1e-12 and np.std(y) > 1e-12:
        return float(spearmanr(x, y).statistic)
    return float("nan")


def atc_rate(
    confidence: np.ndarray, region: np.ndarray, *, threshold: float
) -> pd.Series:
    """Share of each region's rows whose confidence is under `threshold`."""
    by_region = RowsBy(np.asarray(region))
    below = (np.asarray(confidence, dtype=float) < threshold).astype(float)
    return pd.Series(
        by_region.reduce(below), index=pd.Index(by_region.ids, name="region")
    )


def empirical_rate(failures: pd.DataFrame, *, region_class: pd.Series) -> pd.Series:
    """Failure rate of each region; one without rows takes its class's, then the overall."""
    counts = (
        failures.set_index("region")[["n_eval", "n_error"]]
        .reindex(region_class.index)
        .fillna(0)
    )
    by_class = counts.groupby(region_class).sum()
    class_rate = by_class["n_error"] / by_class["n_eval"].replace(0, np.nan)
    overall = counts["n_error"].sum() / counts["n_eval"].sum()
    fallback = region_class.map(class_rate).fillna(overall)
    return (counts["n_error"] / counts["n_eval"].replace(0, np.nan)).fillna(fallback)


def calibrate(
    score: pd.Series, *, val_score: pd.Series, val_rate: pd.Series
) -> tuple[pd.Series, float, float]:
    """`score` mapped onto the failure rate's scale by a Platt fit of val's regions."""
    intercept, slope = fit_platt(val_score.to_numpy(), val_rate.to_numpy())
    return (
        expit(intercept + slope * score),
        intercept,
        slope,
    )


def combine(regressor: pd.Series, other: pd.Series) -> pd.Series:
    """Average of the regressor's prediction and another variant's."""
    return (regressor + other) / 2.0


def score_variants(
    rates: dict[str, pd.Series],
    observed: pd.Series,
    *,
    region: np.ndarray,
    failure: np.ndarray,
    row_scores: dict[str, np.ndarray],
) -> list[dict]:
    """Rho and MSE of each variant's region rates, and its oracle benefit over the rows."""
    # A row's score is its region's rate unless `row_scores` gives the variant's own; a
    # combo of such a variant averages the regressor's rate with them row by row.
    by_region = RowsBy(region)
    scores = {
        name: by_region.spread(rate.loc[by_region.ids].to_numpy(dtype=float))
        for name, rate in rates.items()
    }
    for combo, partner in COMBOS.items():
        if partner in row_scores:
            scores[combo] = combine(scores["regressor"], row_scores[partner])
    scores.update(row_scores)

    support = np.ones(failure.size)
    table = []
    for name in VARIANTS:
        rate = rates[name].loc[observed.index].to_numpy(dtype=float)
        table.append(
            {
                "variant": name,
                "spearman": rank_correlation(rate, observed),
                "mse": float(np.mean((rate - observed.to_numpy(dtype=float)) ** 2)),
                # A NaN score, a baseline val could not calibrate, ranks no row.
                "oracle_benefit_recovered": (
                    oracle_benefit_recovered(scores[name], failure, support)
                    if np.isfinite(scores[name]).all()
                    else float("nan")
                ),
            }
        )
    return table


def _mean_and_se(values: np.ndarray) -> tuple[float, float]:
    se = float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else np.nan
    return float(values.mean()), se


def error_by_region_size(
    observed: pd.Series,
    predictions: dict[str, pd.Series],
    *,
    size: pd.Series,
    n_bins: int,
) -> list[dict]:
    """Per bin of region size, each prediction's squared and signed error and its rho."""
    regions = observed.index
    if len(regions) < n_bins:
        return []
    order = np.lexsort((regions.to_numpy(), size.loc[regions].to_numpy()))
    size_bin_of = np.empty(len(regions), dtype=int)
    size_bin_of[order] = np.arange(len(regions)) * n_bins // len(regions)

    rate = observed.to_numpy(dtype=float)
    rows = []
    for size_bin in range(n_bins):
        in_bin = size_bin_of == size_bin
        sizes = size.loc[regions[in_bin]]
        for variant, predicted in predictions.items():
            value = predicted.loc[regions[in_bin]].to_numpy(dtype=float)
            error = value - rate[in_bin]
            mse, mse_se = _mean_and_se(error**2)
            bias, bias_se = _mean_and_se(error)
            rows.append(
                {
                    "size_bin": size_bin,
                    "variant": variant,
                    "n_regions": int(in_bin.sum()),
                    "size_min": int(sizes.min()),
                    "size_max": int(sizes.max()),
                    "spearman": rank_correlation(value, rate[in_bin]),
                    "mse": mse,
                    "mse_se": mse_se,
                    "bias": bias,
                    "bias_se": bias_se,
                }
            )
    return rows
