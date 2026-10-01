import logging

import numpy as np
import pandas as pd

from src.core.io import load_arrays, load_df, save_df
from src.core.log import setup_logger
from src.core.record import clear_dir, is_current, write_record
from src.core.utils import flush_timing, load_from_json, timed
from src.domain.analysis.complexity import (
    Queries,
    Reference,
    analysis_centroids,
    compute_population_complexity,
    query_neighbors,
)
from src.domain.analysis.grouping import RowGroups
from src.domain.data.space import Space
from stages import (
    load_cli_config,
    load_space,
    load_split,
    paths_from_cfg,
    stage_config,
    upstream_ids,
)

setup_logger()
logger = logging.getLogger(__name__)

# Bumped when the code changes what a config builds: older records never match.
SCHEMA = 4
OUTPUTS = ("regions.parquet", "classes.parquet")


def _sample_queries(
    cfg,
    *,
    space: Space,
    train: pd.DataFrame,
    groups: RowGroups,
    population: np.ndarray,
    reference: Reference,
    reference_rows: np.ndarray,
    cap: int | None,
) -> Queries:
    """Up to `cap` rows of every population, in train order, with their nearest
    reference points."""
    rng = np.random.default_rng(cfg.seed)
    picked = []
    for i in range(len(groups.ids)):
        members = groups.members(i)
        if cap is not None and len(members) > cap:
            members = rng.choice(members, size=cap, replace=False)
        picked.append(members)
    rows = np.sort(np.concatenate(picked))
    if np.array_equal(rows, reference_rows):
        # Every reference point is a query: its neighbours are the graph's own.
        return Queries(
            reference.X, population[rows], reference.knn_idx, reference.knn_dist
        )
    X = space.embed(train.iloc[rows])
    nbs, nb_dist = query_neighbors(
        reference.X, reference_rows, X, rows, k=cfg.graph.k, metric=cfg.distance
    )
    return Queries(X, population[rows], nbs, nb_dist)


@timed
def measure_populations(
    cfg,
    *,
    space: Space,
    train: pd.DataFrame,
    population: np.ndarray,
    reference: Reference,
    reference_rows: np.ndarray,
    population_to_class: dict[str, int],
    centroids: dict[str, list[float]] | None,
    cap: int | None,
) -> dict[str, dict[str, float | None]]:
    """Complexity measures of every population, keyed by its id."""
    groups = RowGroups(population)
    queries = _sample_queries(
        cfg,
        space=space,
        train=train,
        groups=groups,
        population=population,
        reference=reference,
        reference_rows=reference_rows,
        cap=cap,
    )
    cx = cfg.complexity
    return compute_population_complexity(
        reference,
        population[reference_rows],
        queries,
        population_to_class=population_to_class,
        centroids=centroids
        or analysis_centroids(queries.X, queries.population, metric=cfg.distance),
        sizes={int(i): int(n) for i, n in zip(groups.ids, groups.sizes)},
        top_k_clusters=cx.top_k_clusters,
        metric=cfg.distance,
        silhouette_max_samples=cx.silhouette_max_samples,
        silhouette_min_per_cluster=cx.silhouette_min_per_cluster,
        random_state=cfg.seed,
    )


def main() -> None:
    """Entry point for the complexity stage."""
    cfg = load_cli_config()
    paths = paths_from_cfg(cfg)
    stage_dir = paths.of("complexity")
    config = stage_config(cfg, "complexity")
    inputs = upstream_ids(cfg, paths, "complexity")
    if is_current(
        stage_dir, OUTPUTS, schema=SCHEMA, config=config, inputs=inputs, force=cfg.force
    ):
        return

    clear_dir(stage_dir)
    space = load_space(paths, load_from_json(paths.of("split") / "meta.json"))
    train = load_split(paths, "train")
    graph = load_arrays(paths.of("graph") / "graph.npz")
    assignments = load_df(
        paths.of("regions") / "assignments.parquet", filters=[("split", "==", "train")]
    )
    region = assignments.sort_values("row")["region"].to_numpy(dtype=np.int64)
    if len(region) != len(train):
        raise ValueError(
            f"regions assigned {len(region)} train rows, split holds {len(train)}: "
            "re-run `make regions`."
        )
    label = train["label"].to_numpy(dtype=np.int64)
    rows = graph["rows"]
    reference = Reference(
        space.embed(train.iloc[rows]),
        graph["knn_idx"],
        graph["knn_dist"],
        graph["mst"],
    )

    # A reference that is the whole split makes every row a query. Past that, a
    # population is measured on a sample of its rows.
    sampled = len(train) > cfg.graph.max_samples
    cx = cfg.complexity
    centroids = load_df(paths.of("regions") / "centroids.parquet")
    coordinates = centroids.drop(columns=["region", "class_id"]).to_numpy()
    region_class = dict(zip(centroids["region"], centroids["class_id"]))

    region_measures = measure_populations(
        cfg,
        space=space,
        train=train,
        population=region,
        reference=reference,
        reference_rows=rows,
        population_to_class={str(r): int(c) for r, c in region_class.items()},
        # Exact centroids: the ones regions routes by, not a sample's.
        centroids={
            str(r): point.tolist() for r, point in zip(centroids["region"], coordinates)
        },
        cap=cx.max_queries_per_region if sampled else None,
    )
    class_measures = measure_populations(
        cfg,
        space=space,
        train=train,
        population=label,
        reference=reference,
        reference_rows=rows,
        population_to_class={str(c): int(c) for c in np.unique(label)},
        centroids=None,
        cap=cx.max_queries_per_class if sampled else None,
    )
    save_df(
        pd.DataFrame(
            [
                {"region": int(r), "class_id": int(region_class[int(r)]), **row}
                for r, row in region_measures.items()
            ]
        ),
        stage_dir / "regions.parquet",
    )
    save_df(
        pd.DataFrame(
            [{"class_id": int(c), **row} for c, row in class_measures.items()]
        ),
        stage_dir / "classes.parquet",
    )
    flush_timing(stage_dir / "timing.json")
    write_record(stage_dir, schema=SCHEMA, config=config, inputs=inputs)


if __name__ == "__main__":
    main()
