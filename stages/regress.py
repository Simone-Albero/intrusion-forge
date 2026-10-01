import logging

import numpy as np
import pandas as pd

from src.core.config import to_container
from src.core.io import load_df, save_df
from src.core.log import setup_logger
from src.core.record import clear_dir, write_record
from src.core.utils import flush_timing, save_to_json, timed
from src.domain.analysis.classification import region_failures
from src.domain.analysis.confidence import atc_threshold
from src.domain.analysis.failure import is_failure
from src.domain.analysis.failure_regressor import (
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
SCHEMA = 2


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
            "y_true": labels,
            "y_pred": predictions["y_pred"].to_numpy(),
            "mcp_risk": predictions["mcp_risk"].to_numpy(dtype=np.float64),
        }
    )


@timed
def regress(cfg, paths) -> tuple[pd.DataFrame, dict, dict | None]:
    """Fit the failure regressor on the regions' descriptors and failure rates."""
    test = _evaluated_rows(paths, "test")
    failures = region_failures(
        test["region"].to_numpy(),
        test["y_true"].to_numpy(),
        test["y_pred"].to_numpy(),
        test["mcp_risk"].to_numpy(),
    )
    summary = join_region_summary(
        load_df(paths.of("complexity") / "regions.parquet"),
        load_df(paths.of("complexity") / "classes.parquet"),
        failures,
    )
    fr = cfg.failure_regressor
    results, predicted_rate = fit_failure_regressor(
        summary,
        param_grid=to_container(fr.param_grid),
        n_outer_splits=fr.n_outer_splits,
        n_inner_splits=fr.n_inner_splits,
        n_iter=fr.n_iter,
        min_eval_support=fr.min_eval_support,
        random_state=cfg.seed,
    )
    table = summary[["class_id", "n_eval", "failure_rate", "mcp_risk"]].copy()
    table["n_error"] = (
        failures.set_index("region")["n_error"]
        .reindex(table.index)
        .fillna(0)
        .astype(int)
    )
    table["predicted_rate"] = predicted_rate.reindex(table.index)
    table["used"] = table.index.isin(predicted_rate.index)
    table = table.reset_index()[
        [
            "region",
            "class_id",
            "n_eval",
            "n_error",
            "failure_rate",
            "mcp_risk",
            "predicted_rate",
            "used",
        ]
    ]
    if results.get("skipped"):
        logger.info("Instance-level baselines skipped: the failure regressor was.")
        return table, results, None
    # ATC cuts confidence where as many rows fall below as were misjudged: chosen on val,
    # so the test rows it is scored on never pick their own cut.
    val = _evaluated_rows(paths, "val")
    threshold = atc_threshold(
        1.0 - val["mcp_risk"].to_numpy(),
        ~is_failure(val["y_true"].to_numpy(), val["y_pred"].to_numpy()),
    )
    baselines = {
        **instance_baselines(test, predicted_rate, atc_threshold=threshold),
        "atc_threshold": threshold,
        "n_val": len(val),
    }
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
