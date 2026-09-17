import logging
import sys
from pathlib import Path

from pipelines import paths_from_cfg
from src.core.config import load_config, save_config, to_container
from src.core.io import load_df
from src.core.log import JSONSubscriber, LogBundle, LogDispatcher, setup_logger
from src.core.utils import flush_timing, load_from_json
from src.domain.analysis.failure_regressor import (
    build_cluster_summary,
    fit_failure_regressor,
    instance_baselines,
)

setup_logger(log_file="resources/logs.txt")
logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for the failure-regressor stage."""
    cfg = load_config(
        config_path=Path(__file__).parent.parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )
    paths = paths_from_cfg(cfg)
    save_config(cfg, paths.configs / "config_composed.json")

    bus = LogDispatcher()
    bus.subscribe(JSONSubscriber(paths.outputs))

    complexity_path = paths.shared / "complexity.json"
    class_complexity_path = paths.shared / "class_complexity.json"
    for p in (complexity_path, class_complexity_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing complexity artifact at {p}. Run `make complexity` first."
            )
    complexity = load_from_json(complexity_path)
    class_complexity = load_from_json(class_complexity_path)
    predictions = load_from_json(paths.outputs / "analysis/predictions/clusters.json")

    cluster_summary = build_cluster_summary(
        complexity,
        class_complexity,
        predictions,
    )
    bus.publish(LogBundle.from_dict({"json/analysis/cluster_summary": cluster_summary}))
    logger.info("Cluster summary published.")

    results = fit_failure_regressor(
        cluster_summary,
        to_container(cfg.failure_regressor.param_grid),
        n_outer_splits=cfg.failure_regressor.n_outer_splits,
        n_inner_splits=cfg.failure_regressor.n_inner_splits,
        min_test_support=cfg.failure_regressor.min_test_support,
        random_state=cfg.seed,
    )
    bus.publish(
        LogBundle.from_dict({"json/analysis/failure_regressor_results": results})
    )

    dump_path = paths.outputs / "analysis/predictions/oof_samples.parquet"
    if (
        not results.get("skipped")
        and results.get("oof_predicted_rate")
        and dump_path.exists()
    ):
        instance = instance_baselines(load_df(dump_path), results["oof_predicted_rate"])
        bus.publish(LogBundle.from_dict({"json/analysis/instance_baselines": instance}))
        logger.info(
            "Instance-level baselines published (%d test samples, %d clusters).",
            instance["n_test"],
            instance["n_clusters"],
        )
    else:
        logger.info(
            "Instance-level baselines skipped (no per-sample dump at %s).", dump_path
        )

    flush_timing(paths.outputs / "timing.json")


if __name__ == "__main__":
    main()
