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
from src.domain.data.digest import digest_frames, digest_regions

setup_logger()
logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for the failure-regressor stage."""
    cfg = load_config(
        config_path=Path(__file__).parent.parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )
    paths = paths_from_cfg(cfg)
    bus = LogDispatcher()
    bus.subscribe(JSONSubscriber(paths.outputs))

    # Each upstream stage writes its record last: without one, its outputs may disagree.
    for stage, record in (
        ("prepare", paths.shared / "prepare_fingerprint.json"),
        ("complexity", paths.shared / "complexity_fingerprint.json"),
    ):
        if not record.exists():
            raise FileNotFoundError(f"Missing {record}: run `make {stage}` first.")
    complexity_path = paths.shared / "complexity.json"
    class_complexity_path = paths.shared / "class_complexity.json"
    dump_path = paths.outputs / "analysis/predictions/oof_samples.parquet"
    if not dump_path.exists():
        raise FileNotFoundError(f"Missing {dump_path}: re-run `make classify`.")
    complexity = load_from_json(complexity_path)
    if any("class_id" not in row for row in complexity):
        raise ValueError(
            f"{complexity_path} predates the current artifact format: "
            "re-run `make complexity`."
        )
    class_complexity = load_from_json(class_complexity_path)
    predictions_path = paths.outputs / "analysis/predictions/clusters.json"
    predictions = load_from_json(predictions_path)
    if any("n_eval" not in row for row in predictions["clusters"]):
        raise ValueError(
            f"{predictions_path} predates the current artifact format: "
            "re-run `make classify`."
        )
    # Descriptors and failure rates are joined by cluster id, so both must come from the
    # data prepare holds now: recompute the digests each stage recorded of what it read.
    # A digest a stage never recorded counts as stale.
    splits = {
        split: load_df(paths.processed_data / f"{split}.{cfg.data.extension}")
        for split in ("train", "val", "test")
    }
    if any("routed_cluster" not in df.columns for df in splits.values()):
        raise ValueError(
            f"{paths.processed_data} predates `routed_cluster`: re-run `make prepare`."
        )
    columns = (
        list(cfg.data.num_cols)
        + list(cfg.data.cat_cols)
        + [f"encoded_{cfg.data.label_col}"]
    )
    regions = digest_regions(splits["train"])
    current = {
        "complexity": (digest_frames({"train": splits["train"]}, columns), regions),
        "classify": (
            digest_frames(splits, columns),
            digest_frames(splits, ["routed_cluster"]),
        ),
    }
    complexity_record = load_from_json(paths.shared / "complexity_fingerprint.json")
    recorded = {
        "complexity": (
            complexity_record.get("data_digest"),
            complexity_record.get("regions_digest"),
        ),
        "classify": (predictions.get("data_digest"), predictions.get("routed_digest")),
    }
    stale = [stage for stage in current if recorded[stage] != current[stage]]
    if stale:
        raise ValueError(
            f"The prepared data changed since {' and '.join(stale)} last ran: re-run "
            + " and ".join(f"`make {stage}`" for stage in stale)
            + "."
        )

    cluster_summary = build_cluster_summary(
        complexity,
        class_complexity,
        predictions,
    )
    bus.publish(LogBundle.from_dict({"json/analysis/cluster_summary": cluster_summary}))
    logger.info("Cluster summary published.")

    results = fit_failure_regressor(
        cluster_summary,
        param_grid=to_container(cfg.failure_regressor.param_grid),
        n_outer_splits=cfg.failure_regressor.n_outer_splits,
        n_inner_splits=cfg.failure_regressor.n_inner_splits,
        n_iter=cfg.failure_regressor.n_iter,
        min_eval_support=cfg.failure_regressor.min_eval_support,
        random_state=cfg.seed,
    )
    bus.publish(
        LogBundle.from_dict({"json/analysis/failure_regressor_results": results})
    )

    if results.get("skipped"):
        # An earlier run's baselines would otherwise pair with these skipped results.
        (paths.outputs / "analysis/instance_baselines.json").unlink(missing_ok=True)
        logger.info("Instance-level baselines skipped: the failure regressor was.")
    else:
        instance = instance_baselines(load_df(dump_path), results["oof_predicted_rate"])
        bus.publish(LogBundle.from_dict({"json/analysis/instance_baselines": instance}))
        logger.info(
            "Instance-level baselines published (%d rows in %d scored clusters).",
            instance["n_eval"],
            instance["n_clusters"],
        )

    flush_timing(paths.outputs / "timing.json")
    save_config(cfg, paths.configs / "config_composed_regress.json")


if __name__ == "__main__":
    main()
