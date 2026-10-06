import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.domain.analysis.baselines import rank_correlation

_SCORES = ("spearman", "mse", "r2")


def _score(predicted: pd.Series, observed: pd.Series) -> dict[str, float]:
    if predicted.empty:
        return dict.fromkeys(_SCORES, np.nan)
    return {
        "spearman": rank_correlation(predicted, observed),
        "mse": float(np.mean((predicted - observed) ** 2)),
        # sklearn scores a constant target 0.0, a number where there is none.
        "r2": (
            float(r2_score(observed, predicted)) if observed.nunique() > 1 else np.nan
        ),
    }


def _error_rate(regions: pd.DataFrame) -> float:
    return float(regions["n_error"].sum() / regions["n_eval"].sum())


def score_transfer(
    source_regions: pd.DataFrame, target_regions: dict[str, pd.DataFrame]
) -> list[dict]:
    """Score the source's regressor against each target's own, on the target's rates."""
    used_regions = source_regions.index[source_regions["used"]]
    source_predicted = source_regions.loc[used_regions, "predicted_rate"]
    source_rate = source_regions.loc[used_regions, "failure_rate"]
    rows = []
    for classifier, target in target_regions.items():
        target_rate = target.loc[used_regions, "failure_rate"]
        target_used = target[target["used"]]
        own_scores = _score(target_used["predicted_rate"], target_used["failure_rate"])
        transfer_scores = _score(source_predicted, target_rate)
        rows.append(
            {
                "classifier": classifier,
                **{f"own_{key}": own_scores[key] for key in _SCORES},
                **{f"transfer_{key}": transfer_scores[key] for key in _SCORES},
                "error_rate_delta": _error_rate(target.loc[used_regions])
                - _error_rate(source_regions.loc[used_regions]),
                "failure_rate_mae": float(np.mean(np.abs(target_rate - source_rate))),
                "failure_rate_spearman": rank_correlation(target_rate, source_rate),
            }
        )
    return rows
