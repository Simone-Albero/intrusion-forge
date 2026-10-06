import logging
from pathlib import Path

from src.core.io import load_df
from src.core.log import setup_logger
from src.core.paths import RunPaths
from src.core.record import clear_dir, read_record
from src.core.utils import first_difference, load_from_json, save_to_json
from src.domain.analysis.transfer import score_transfer
from stages import load_cli_config, paths_from_cfg, regressed_runs

setup_logger()
logger = logging.getLogger(__name__)


def _load_runs(dataset_dir: Path) -> dict[str, dict]:
    """Each regressed classifier's config, results and regions, raising when stale."""
    runs: dict[str, dict] = {}
    for paths in regressed_runs(dataset_dir):
        regress_dir = paths.of("regress")
        runs[paths.classifier.name] = {
            "config": read_record(regress_dir)["config"],
            "results": load_from_json(regress_dir / "results.json"),
            "regions": load_df(regress_dir / "regions.parquet").set_index("region"),
        }
    return runs


def transfer(paths: RunPaths) -> dict:
    """Score the best failure regressor on every other classifier's failure rates."""
    runs = _load_runs(paths.dataset)
    if len(runs) < 2:
        raise ValueError(
            f"{paths.dataset} holds {len(runs)} classifier(s) with a regress stage: "
            "transfer needs at least two."
        )
    reference, *others = runs
    for name in others:
        changed = first_difference(runs[reference]["config"], runs[name]["config"])
        if changed is not None:
            raise ValueError(
                f"regress ran with another {changed} for {reference} and {name}: "
                "re-run `make regress` for both."
            )
    # A skipped regressor has no score, and a constant prediction a null one.
    ranked = [
        name for name in runs if runs[name]["results"].get("spearman") is not None
    ]
    if not ranked:
        raise ValueError(
            "No failure regressor has a Spearman score: nothing to transfer."
        )
    source = max(ranked, key=lambda name: runs[name]["results"]["spearman"])
    target_scores = score_transfer(
        runs[source]["regions"],
        {name: run["regions"] for name, run in runs.items() if name != source},
    )
    source_results = runs[source]["results"]
    logger.info(
        "Transfer from %s (rho %.4f, %d regions):",
        source,
        source_results["spearman"],
        source_results["n_regions_used"],
    )
    for row in target_scores:
        logger.info(
            "  %-24s own rho=%7.4f  transfer rho=%7.4f  error rate delta=%7.4f",
            row["classifier"],
            row["own_spearman"],
            row["transfer_spearman"],
            row["error_rate_delta"],
        )
    return {
        "source": source,
        "source_spearman": source_results["spearman"],
        "source_error_rate": source_results["global_error_rate"],
        "n_regions_used": source_results["n_regions_used"],
        "targets": target_scores,
    }


def main() -> None:
    """Entry point for the transfer stage."""
    cfg = load_cli_config()
    paths = paths_from_cfg(cfg)
    results = transfer(paths)
    stage_dir = paths.of("transfer")
    clear_dir(stage_dir)
    save_to_json(results, stage_dir / "results.json")


if __name__ == "__main__":
    main()
