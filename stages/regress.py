import logging

import numpy as np
import pandas as pd

from src.core.config import to_container
from src.core.io import load_df, save_df
from src.core.log import setup_logger
from src.core.record import clear_dir, write_record
from src.core.utils import flush_timing, load_from_json, save_to_json, timed
from src.domain.analysis.baselines import (
    CALIBRATED,
    COMBOS,
    SIZE_ERROR_VARIANTS,
    atc_rate,
    calibrate,
    combine,
    empirical_rate,
    error_by_region_size,
    score_variants,
)
from src.domain.analysis.classification import region_failures
from src.domain.analysis.confidence import atc_threshold, mcp_risk
from src.domain.analysis.failure import is_failure
from src.domain.analysis.failure_regressor import (
    fit_failure_regressor,
    join_region_summary,
)
from src.domain.analysis.sample_regressor import fit_sample_regressor
from stages import (
    load_cli_config,
    load_split,
    paths_from_cfg,
    stage_config,
    upstream_ids,
)

setup_logger()
logger = logging.getLogger(__name__)

SIZE_BINS = 5


def _evaluated_rows(paths, split: str) -> pd.DataFrame:
    """The rows of a split with their region, true class and what the classifier said."""
    labels = load_split(paths, split, columns=["label"])["label"].to_numpy()
    only = [("split", "==", split)]
    regions = load_df(paths.of("regions") / "assignments.parquet", filters=only)
    predictions = load_df(paths.of("classify") / "predictions.parquet", filters=only)
    regions = regions.sort_values("row")
    predictions = predictions.sort_values("row")
    proba_columns = [c for c in predictions.columns if c.startswith("proba_")]
    if not proba_columns:
        raise ValueError(
            "classify's predictions hold no class probabilities: re-run "
            "`make classify FORCE=1`."
        )
    if not len(labels) == len(regions) == len(predictions):
        raise ValueError(
            f"{split} has {len(labels)} rows, {len(regions)} regions and "
            f"{len(predictions)} predictions: re-run the stages that disagree."
        )
    return pd.DataFrame(
        {
            "region": regions["region"].to_numpy(),
            "nearest_region": regions["nearest_region"].to_numpy(),
            "y_true": labels,
            "y_pred": predictions["y_pred"].to_numpy(),
            "mcp_risk": mcp_risk(predictions[proba_columns].to_numpy(dtype=np.float64)),
            "in_fit": predictions["in_fit"].to_numpy(dtype=bool),
        }
    )


def _failures(rows: pd.DataFrame) -> pd.DataFrame:
    return region_failures(
        rows["region"].to_numpy(),
        rows["y_true"].to_numpy(),
        rows["y_pred"].to_numpy(),
        rows["mcp_risk"].to_numpy(),
    )


def _region_table(
    summary: pd.DataFrame,
    failures: pd.DataFrame,
    *,
    train: pd.DataFrame,
    fit_failures: pd.DataFrame,
    val_failures: pd.DataFrame,
    held_out: pd.DataFrame,
) -> pd.DataFrame:
    table = summary[["class_id", "n_eval", "failure_rate", "mcp_risk"]].copy()
    table["n_train"] = (
        train["region"].value_counts().reindex(table.index).fillna(0).astype(int)
    )
    for name, counted in (("fit", fit_failures), ("val", val_failures)):
        by_region = counted.set_index("region").reindex(table.index)
        table[f"n_{name}"] = by_region["n_eval"].fillna(0).astype(int)
        table[f"{name}_failure_rate"] = by_region["failure_rate"]
    table["n_error"] = (
        failures.set_index("region")["n_error"]
        .reindex(table.index)
        .fillna(0)
        .astype(int)
    )
    table["predicted_rate"] = held_out["predicted_rate"].reindex(table.index)
    table["used"] = table.index.isin(held_out.index)
    return table.reset_index()[
        [
            "region",
            "class_id",
            "n_eval",
            "n_error",
            "failure_rate",
            "mcp_risk",
            "n_train",
            "n_fit",
            "fit_failure_rate",
            "n_val",
            "val_failure_rate",
            "predicted_rate",
            "used",
        ]
    ]


def _atc(val: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.Series, pd.Series, float]:
    val_confidence = 1.0 - val["mcp_risk"].to_numpy()
    threshold = atc_threshold(
        val_confidence, ~is_failure(val["y_true"].to_numpy(), val["y_pred"].to_numpy())
    )
    return (
        atc_rate(
            1.0 - test["mcp_risk"].to_numpy(), test["region"], threshold=threshold
        ),
        atc_rate(val_confidence, val["region"], threshold=threshold),
        threshold,
    )


def _mcp(
    failures: pd.DataFrame, val_failures: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    return (
        failures.set_index("region")["mcp_risk"],
        val_failures.set_index("region")["mcp_risk"],
    )


def _sample_regressor(
    cfg, paths, scored_rows: pd.DataFrame, held_out: pd.DataFrame
) -> tuple[pd.Series, np.ndarray]:
    """Each region's mean predicted chance of failure, and each row's, from the row's own
    features, never the region's geometry."""
    meta = load_from_json(paths.of("split") / "meta.json")
    regressor_cfg = cfg.failure_regressor
    features = load_split(paths, "test", columns=meta["num_cols"] + meta["cat_cols"])
    row_positions = scored_rows.index.to_numpy()
    predicted = fit_sample_regressor(
        features.iloc[row_positions],
        is_failure(
            scored_rows["y_true"].to_numpy(), scored_rows["y_pred"].to_numpy()
        ).astype(float),
        scored_rows["region"].to_numpy(),
        fold_of_region=held_out["fold"],
        name=regressor_cfg.primary,
        # No search runs here, so the cores the search's candidates would use are free.
        params={
            **to_container(regressor_cfg.models)[regressor_cfg.primary]["params"],
            "n_jobs": -1,
        },
        max_rows=regressor_cfg.max_sample_rows,
        random_state=cfg.seed,
    )
    rate = pd.Series(predicted).groupby(scored_rows["region"].to_numpy()).mean()
    rate.index.name = "region"
    return rate, predicted


def _calibrated(
    rates: dict[str, pd.Series],
    val_scores: dict[str, pd.Series],
    *,
    val_rate: pd.Series,
) -> tuple[dict[str, pd.Series], list[dict]]:
    """Each raw rate mapped by a fit on val's regions, so no test row sets its scale."""
    # A val without failures, or with nothing else, has no scale to fit: the calibrated
    # variants stay NaN instead of stopping the stage over a fact of the data.
    can_calibrate = 0.0 < val_rate.mean() < 1.0
    if not can_calibrate:
        logger.warning(
            "Val's regions fail at a rate of %s: no calibrated baselines.",
            val_rate.mean(),
        )
    calibrated, calibration = {}, []
    for name, raw in CALIBRATED.items():
        rate = pd.Series(np.nan, index=rates[raw].index)
        intercept, slope = np.nan, np.nan
        if can_calibrate:
            rate, intercept, slope = calibrate(
                rates[raw], val_score=val_scores[raw], val_rate=val_rate
            )
            if slope == 0.0 and val_scores[raw].nunique() > 1:
                logger.warning(
                    "%s does not rise with val's failures: its calibration is constant.",
                    raw,
                )
        calibrated[name] = rate
        calibration.append({"variant": name, "intercept": intercept, "slope": slope})
    return calibrated, calibration


def _score_baselines(
    cfg,
    paths,
    *,
    summary: pd.DataFrame,
    held_out: pd.DataFrame,
    region_table: pd.DataFrame,
    test: pd.DataFrame,
    val: pd.DataFrame,
    failures: pd.DataFrame,
    fit_failures: pd.DataFrame,
    val_failures: pd.DataFrame,
) -> dict:
    scored_regions = held_out.index
    region_class = summary["class_id"]
    atc, val_atc, threshold = _atc(val, test)
    mcp, val_mcp = _mcp(failures, val_failures)
    train_empiric = empirical_rate(fit_failures, region_class=region_class)
    val_empiric = empirical_rate(val_failures, region_class=region_class)
    scored_rows = test[test["region"].isin(scored_regions)]
    sample_regressor_rate, row_predictions = _sample_regressor(
        cfg, paths, scored_rows, held_out
    )

    rates = {
        "regressor": held_out["predicted_rate"],
        "sample_regressor": sample_regressor_rate,
        "atc": atc,
        "mcp": mcp,
        "train_empiric": train_empiric,
        "val_empiric": val_empiric,
    }
    rates = {name: rate.loc[scored_regions] for name, rate in rates.items()}
    val_regions = val_failures["region"]
    val_rate = val_failures.set_index("region")["failure_rate"]
    calibrated, calibration = _calibrated(
        rates,
        {
            "atc": val_atc.loc[val_regions],
            "mcp": val_mcp,
            "train_empiric": train_empiric.loc[val_regions],
        },
        val_rate=val_rate,
    )
    rates |= calibrated
    rates |= {
        combo: combine(rates["regressor"], rates[partner])
        for combo, partner in COMBOS.items()
    }

    observed = summary.loc[scored_regions, "failure_rate"]
    report = {
        "n_eval": len(scored_rows),
        "n_regions": len(scored_regions),
        "baselines": score_variants(
            rates,
            observed,
            region=scored_rows["region"].to_numpy(),
            failure=is_failure(
                scored_rows["y_true"].to_numpy(), scored_rows["y_pred"].to_numpy()
            ).astype(float),
            row_scores={"sample_regressor": row_predictions},
        ),
        "atc_threshold": threshold,
        "calibration": calibration,
        "n_regions_val": len(val_regions),
        "n_val": len(val),
        "n_regions_without_fit": int(
            (region_table["used"] & (region_table["n_fit"] == 0)).sum()
        ),
        "n_regions_without_val": int(
            (region_table["used"] & (region_table["n_val"] == 0)).sum()
        ),
        "error_by_size": error_by_region_size(
            observed,
            {name: rates[name] for name in SIZE_ERROR_VARIANTS},
            size=region_table.set_index("region")["n_train"],
            n_bins=SIZE_BINS,
        ),
    }
    if not report["error_by_size"]:
        logger.warning(
            "Fewer than %d scored regions: no error-by-size table.", SIZE_BINS
        )
    logger.info(
        "Baselines (%d test rows in %d scored regions):",
        report["n_eval"],
        report["n_regions"],
    )
    for row in report["baselines"]:
        logger.info(
            "  %-24s rho=%7.4f  mse=%.5f  oracle benefit=%7.4f",
            row["variant"],
            row["spearman"],
            row["mse"],
            row["oracle_benefit_recovered"],
        )
    return report


@timed
def regress(cfg, paths) -> tuple[pd.DataFrame, dict, dict | None]:
    """Fit the failure regressor on the regions' descriptors and failure rates, and score
    it against the baselines."""
    train, val, test = (
        _evaluated_rows(paths, split) for split in ("train", "val", "test")
    )
    failures = _failures(test)
    # Never gets a column the stage adds: the regressor takes every numeric one as a feature.
    summary = join_region_summary(
        load_df(paths.of("complexity") / "regions.parquet"),
        load_df(paths.of("complexity") / "classes.parquet"),
        failures,
    )
    regressor_cfg = cfg.failure_regressor
    results, held_out = fit_failure_regressor(
        summary,
        models=to_container(regressor_cfg.models),
        primary=regressor_cfg.primary,
        n_outer_splits=regressor_cfg.n_outer_splits,
        n_inner_splits=regressor_cfg.n_inner_splits,
        n_iter=regressor_cfg.n_iter,
        min_eval_support=regressor_cfg.min_eval_support,
        random_state=cfg.seed,
    )
    # Counted in the region a test row like it is routed to, not in the cluster its own
    # class drew it into: the target holds every class's rows that land there.
    fit_rows = train[train["in_fit"]]
    fit_failures = _failures(fit_rows.assign(region=fit_rows["nearest_region"]))
    val_failures = _failures(val)
    table = _region_table(
        summary,
        failures,
        train=train,
        fit_failures=fit_failures,
        val_failures=val_failures,
        held_out=held_out,
    )
    if results.get("skipped"):
        logger.info("Baselines skipped: the failure regressor was.")
        return table, results, None
    baselines = _score_baselines(
        cfg,
        paths,
        summary=summary,
        held_out=held_out,
        region_table=table,
        test=test,
        val=val,
        failures=failures,
        fit_failures=fit_failures,
        val_failures=val_failures,
    )
    return table, results, baselines


def main() -> None:
    """Entry point for the regress stage."""
    cfg = load_cli_config()
    paths = paths_from_cfg(cfg)
    stage_dir = paths.of("regress")
    config = stage_config(cfg, "regress")
    inputs = upstream_ids(cfg, paths, "regress")

    clear_dir(stage_dir)
    table, results, baselines = regress(cfg, paths)
    save_df(table, stage_dir / "regions.parquet")
    save_to_json(results, stage_dir / "results.json")
    if baselines is not None:
        save_to_json(baselines, stage_dir / "baselines.json")
    flush_timing(stage_dir / "timing.json")
    write_record(stage_dir, config=config, inputs=inputs)


if __name__ == "__main__":
    main()
