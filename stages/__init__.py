import sys
from pathlib import Path

import pandas as pd

from src.core.config import load_config, select_config
from src.core.io import load_df
from src.core.paths import RunPaths
from src.core.record import read_record
from src.core.utils import first_difference, load_from_json
from src.domain.data.space import Space

SPLITS = ("train", "val", "test")

# The config a stage's output depends on, by dotted key: what its record holds.
CONFIG_KEYS = {
    "split": ("data", "seed"),
    "graph": ("graph", "space", "distance", "seed"),
    "regions": ("clustering", "distance", "seed"),
    "complexity": ("complexity", "distance", "seed"),
    "classify": (
        "classifier",
        "loss",
        "optimizer",
        "scheduler",
        "fit",
        "grid_search",
        "seed",
    ),
    "regress": ("failure_regressor", "seed"),
    "render": ("figure_format",),
}

# The stages whose outputs each stage reads.
READS = {
    "split": (),
    "graph": ("split",),
    "regions": ("split", "graph"),
    "complexity": ("split", "graph", "regions"),
    "classify": ("split",),
    "regress": ("split", "regions", "complexity", "classify"),
    "render": ("split", "complexity", "regress"),
}


def load_cli_config():
    """Compose the config from the command line, as every stage's entry point does."""
    cfg = load_config(
        config_path=Path(__file__).parent.parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )
    if cfg.distance not in ("euclidean", "cosine"):
        raise ValueError(
            f"Unknown distance: {cfg.distance!r}. Valid: 'euclidean', 'cosine'."
        )
    return cfg


def paths_from_cfg(cfg) -> RunPaths:
    """Where each stage of this run writes."""
    base = Path(cfg.path.out_base_path)
    return RunPaths(dataset=base, classifier=base / cfg.classifier.name)


def stage_config(cfg, stage: str) -> dict:
    """The config `stage` depends on, as its record holds it."""
    config = select_config(cfg, CONFIG_KEYS[stage])
    if stage == "classify":
        # Where and how fast a model trains, not on what: reusing a model trained on
        # another device or with other parallelism is the point, even where float
        # rounding differs.
        config["fit"].pop("device")
        if config["classifier"]["params"] is not None:
            config["classifier"]["params"].pop("n_jobs", None)
        for loop in ("training", "validation"):
            for key in ("num_workers", "pin_memory"):
                config["fit"][loop]["dataloader"].pop(key)
    return config


def upstream_ids(cfg, paths: RunPaths, stage: str) -> dict[str, str]:
    """Ids of the stages `stage` reads; raises when one is missing or stale."""
    # Stale: its recorded config differs from the current one, or it was built from
    # another version of a stage than the one on disk.
    ids = {}
    for name in READS[stage]:
        record = read_record(paths.of(name))
        changed = first_difference(record["config"], stage_config(cfg, name))
        if changed is not None:
            raise ValueError(
                f"{name}: {changed} differs from what `make {name}` last ran with: "
                f"re-run `make {name}`."
            )
        ids[name] = record["id"]
    for name in READS[stage]:
        for source, source_id in read_record(paths.of(name))["inputs"].items():
            if read_record(paths.of(source))["id"] != source_id:
                raise ValueError(
                    f"{name} was built from another {source} than the one on disk: "
                    f"re-run `make {name}`."
                )
    return ids


def load_split(
    paths: RunPaths, split: str, *, columns: list[str] | None = None
) -> pd.DataFrame:
    """One split as split wrote it, optionally restricted to `columns`."""
    return load_df(paths.of("split") / f"{split}.parquet", columns=columns)


def load_space(paths: RunPaths, meta: dict) -> Space:
    """The space graph fitted on train."""
    return Space.from_record(
        load_from_json(paths.of("graph") / "space.json"), num_cols=meta["num_cols"]
    )
