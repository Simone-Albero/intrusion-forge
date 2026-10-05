import logging

import numpy as np
import pandas as pd
from scipy.special import expit

from src.core.config import to_container
from src.core.io import load_df, save_df
from src.core.log import setup_logger
from src.core.record import clear_dir, write_record
from src.core.utils import flush_timing, save_to_json, timed
from src.domain.analysis.calibration import fit_platt
from src.domain.analysis.classification import (
    empirical_region_rate,
    region_failures,
)
from src.domain.analysis.confidence import atc_threshold
from src.domain.analysis.failure import is_failure
from src.domain.analysis.failure_regressor import (
    CALIBRATED_VARIANTS,
    error_by_region_size,
    fit_failure_regressor,
    instance_baselines,
    join_region_summary,
)
from stages import (
    load_cli_config,
    load_split,
    paths_from_cfg,
    stage_config,
    upstream_ids,
)

setup_logger()
logger = logging.getLogger(__name__)

# Bumped when the code changes what a config builds: older records never match.
SCHEMA = 6

# Bins of region size the error is reported over.
SIZE_BINS = 5


def _evaluated_rows(paths, split: str) -> pd.DataFrame:
    """The rows of a split with their region, true class and what the classifier said."""
    labels = load_split(paths, split, columns=["label"])["label"].to_numpy()
    only = [("split", "==", split)]
    regions = load_df(paths.of("regions") / "assignments.parquet", filters=only)
    predictions = load_df(paths.of("classify") / "predictions.parquet", filters=only)
    regions = regions.sort_values("row")
    predictions = predictions.sort_values("row")
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
            "mcp_risk": predictions["mcp_risk"].to_numpy(dtype=np.float64),
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


@timed
def regress(cfg, paths) -> tuple[pd.DataFrame, dict, dict | None]:
    """Fit the failure regressor on the regions' descriptors and failure rates."""
    train = _evaluated_rows(paths, "train")
    test = _evaluated_rows(paths, "test")
    val = _evaluated_rows(paths, "val")
    failures = _failures(test)
    summary = join_region_summary(
        load_df(paths.of("complexity") / "regions.parquet"),
        load_df(paths.of("complexity") / "classes.parquet"),
        failures,
    )
    fr = cfg.failure_regressor
    results, predicted_rate = fit_failure_regressor(
        summary,
        models=to_container(fr.models),
        primary=fr.primary,
        n_outer_splits=fr.n_outer_splits,
        n_inner_splits=fr.n_inner_splits,
        n_iter=fr.n_iter,
        min_eval_support=fr.min_eval_support,
        random_state=cfg.seed,
    )
    # Never added to `summary`: the regressor takes every numeric column of it as a feature.
    # Counted in the region a test row like it is routed to, not in the cluster its own
    # class drew it into: the target holds every class's rows that land there.
    fit_rows = train[train["in_fit"]]
    fit_failures = _failures(fit_rows.assign(region=fit_rows["nearest_region"]))
    val_failures = _failures(val)
    train_rate = empirical_region_rate(fit_failures, region_class=summary["class_id"])
    val_rate = empirical_region_rate(val_failures, region_class=summary["class_id"])
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
    table["predicted_rate"] = predicted_rate.reindex(table.index)
    table["used"] = table.index.isin(predicted_rate.index)
    scored = predicted_rate.index
    n_without_fit = int((table.loc[scored, "n_fit"] == 0).sum())
    n_without_val = int((table.loc[scored, "n_val"] == 0).sum())
    table = table.reset_index()[
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
    if results.get("skipped"):
        logger.info("Instance-level baselines skipped: the failure regressor was.")
        return table, results, None
    # ATC cuts confidence where as many rows fall below as were misjudged: chosen on val,
    # so the test rows it is scored on never pick their own cut.
    threshold = atc_threshold(
        1.0 - val["mcp_risk"].to_numpy(),
        ~is_failure(val["y_true"].to_numpy(), val["y_pred"].to_numpy()),
    )

    def atc_rate(rows: pd.DataFrame) -> pd.Series:
        return ((1.0 - rows["mcp_risk"]) < threshold).groupby(rows["region"]).mean()

    rates = {
        "region": predicted_rate,
        "train_rate_region": train_rate.loc[scored],
        "val_rate_region": val_rate.loc[scored],
        "mcp_region": summary.loc[scored, "mcp_risk"],
        "atc_region": atc_rate(test).loc[scored],
    }
    # Each score on val's regions, fitted to val's failures there, then mapped on test's
    # regions: no test row chooses its own scale.
    val_regions = val_failures.set_index("region")
    val_scores = {
        "mcp_region": val_regions["mcp_risk"],
        "atc_region": atc_rate(val).loc[val_regions.index],
        "train_rate_region": train_rate.loc[val_regions.index],
    }
    # A val without failures, or with nothing else, has no scale to fit: the calibrated
    # variants stay NaN instead of stopping the stage over a fact of the data.
    val_failure_rate = val_regions["failure_rate"].mean()
    can_calibrate = 0.0 < val_failure_rate < 1.0
    if not can_calibrate:
        logger.warning(
            "Val's regions fail at a rate of %s: no calibrated baselines.",
            val_failure_rate,
        )
    calibration = []
    for name, raw in CALIBRATED_VARIANTS.items():
        intercept, slope = np.nan, np.nan
        if can_calibrate:
            intercept, slope = fit_platt(
                val_scores[raw].to_numpy(), val_regions["failure_rate"].to_numpy()
            )
        if slope == 0.0 and val_scores[raw].nunique() > 1:
            logger.warning(
                "%s does not rise with val's failures: its calibration is constant.",
                raw,
            )
        rates[name] = expit(intercept + slope * rates[raw])
        calibration.append({"variant": name, "intercept": intercept, "slope": slope})
    baselines = {
        **instance_baselines(
            test,
            predicted_rate,
            atc_threshold=threshold,
            train_rate=train_rate,
            val_rate=val_rate,
            calibrated={name: rates[name] for name in CALIBRATED_VARIANTS},
        ),
        "atc_threshold": threshold,
        "calibration": calibration,
        "n_regions_val": len(val_regions),
        "n_val": len(val),
        "n_regions_without_fit": n_without_fit,
        "n_regions_without_val": n_without_val,
        "error_by_size": error_by_region_size(
            summary.loc[scored, "failure_rate"],
            rates,
            size=table.set_index("region")["n_train"],
            n_bins=SIZE_BINS,
        ),
    }
    if not baselines["error_by_size"]:
        logger.warning(
            "Fewer than %d scored regions: no error-by-size table.", SIZE_BINS
        )
    logger.info(
        "Instance-level baselines (%d rows in %d scored regions).",
        baselines["n_eval"],
        baselines["n_regions"],
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
    write_record(stage_dir, schema=SCHEMA, config=config, inputs=inputs)


if __name__ == "__main__":
    main()
