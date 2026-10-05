import logging

import numpy as np

from src.core.io import save_arrays
from src.core.log import setup_logger
from src.core.record import clear_dir, is_current, write_record
from src.core.utils import flush_timing, load_from_json, save_to_json
from src.domain.analysis.complexity import build_reference
from src.domain.clustering.base import subsample_indices
from src.domain.data.space import Space
from stages import (
    load_cli_config,
    load_split,
    paths_from_cfg,
    stage_config,
    upstream_ids,
)

setup_logger()
logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for the graph stage."""
    cfg = load_cli_config()
    paths = paths_from_cfg(cfg)
    stage_dir = paths.of("graph")
    config = stage_config(cfg, "graph")
    inputs = upstream_ids(cfg, paths, "graph")
    if is_current(stage_dir, config=config, inputs=inputs, force=cfg.force):
        return

    clear_dir(stage_dir)
    meta = load_from_json(paths.of("split") / "meta.json")
    train = load_split(paths, "train")
    rows = np.sort(
        subsample_indices(
            len(train), max_samples=cfg.graph.max_samples, random_state=cfg.seed
        )
    )
    space = Space.fit(
        train,
        num_cols=meta["num_cols"],
        cat_cols=meta["cat_cols"],
        top_k=cfg.space.top_k,
        cat_cost=cfg.space.cat_cost,
    )
    reference = build_reference(
        space.embed(train.iloc[rows]), k=cfg.graph.k, metric=cfg.distance
    )
    save_to_json(space.to_record(), stage_dir / "space.json")
    save_arrays(
        {
            "rows": rows,
            "knn_idx": reference.knn_idx,
            "knn_dist": reference.knn_dist,
            "mst": reference.mst_edges,
        },
        stage_dir / "graph.npz",
    )
    flush_timing(stage_dir / "timing.json")
    write_record(stage_dir, config=config, inputs=inputs)


if __name__ == "__main__":
    main()
