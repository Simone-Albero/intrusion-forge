import logging

import numpy as np
import pandas as pd

from src.core.io import load_arrays, load_df, save_df
from src.core.log import setup_logger
from src.core.record import clear_dir, is_current, write_record
from src.core.utils import flush_timing, load_from_json, timed
from src.domain.analysis.complexity import (
    MeasuredSample,
    TrainGraph,
    compute_population_complexity,
    nearest_neighbors,
)
from src.domain.analysis.complexity.shared import scale_for_metric
from src.domain.analysis.grouping import RowsBy
from src.domain.clustering.base import compute_centroids
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


@timed
def _draw_sample(
    cfg,
    *,
    space: Space,
    train: pd.DataFrame,
    population: np.ndarray,
    train_graph: TrainGraph,
    graph_rows: np.ndarray,
    cap: int | None,
) -> MeasuredSample:
    """Up to `cap` rows of every population, in train order, with their nearest graph
    nodes."""
    by_population = RowsBy(population)
    rng = np.random.default_rng(cfg.seed)
    picked = []
    for i in range(len(by_population.ids)):
        members = by_population.members(i)
        if cap is not None and len(members) > cap:
            members = rng.choice(members, size=cap, replace=False)
        picked.append(members)
    rows = np.sort(np.concatenate(picked))
    if np.array_equal(rows, graph_rows):
        return MeasuredSample(
            train_graph.X,
            population[rows],
            train_graph.knn_idx,
            train_graph.knn_dist,
        )
    X = space.embed(train.iloc[rows])
    neighbors, neighbor_dist = nearest_neighbors(
        train_graph.X, graph_rows, X, rows, k=cfg.graph.k, metric=cfg.distance
    )
    return MeasuredSample(X, population[rows], neighbors, neighbor_dist)


def _measure(
    cfg,
    *,
    train_graph: TrainGraph,
    graph_rows: np.ndarray,
    population: np.ndarray,
    sample: MeasuredSample,
    population_class: dict[int, int],
    centroids: dict[int, np.ndarray],
) -> dict[int, dict[str, float | None]]:
    """Complexity measures of every population, keyed by its id."""
    complexity_cfg = cfg.complexity
    ids, counts = np.unique(population, return_counts=True)
    return compute_population_complexity(
        train_graph,
        population[graph_rows],
        sample,
        population_class=population_class,
        centroids=centroids,
        sizes=dict(zip(ids.tolist(), counts.tolist())),
        top_k_clusters=complexity_cfg.top_k_clusters,
        metric=cfg.distance,
        silhouette_max_samples=complexity_cfg.silhouette_max_samples,
        silhouette_min_per_cluster=complexity_cfg.silhouette_min_per_cluster,
        random_state=cfg.seed,
    )


def main() -> None:
    """Entry point for the complexity stage."""
    cfg = load_cli_config()
    paths = paths_from_cfg(cfg)
    stage_dir = paths.of("complexity")
    config = stage_config(cfg, "complexity")
    inputs = upstream_ids(cfg, paths, "complexity")
    if is_current(stage_dir, config=config, inputs=inputs, force=cfg.force):
        return

    clear_dir(stage_dir)
    space = load_space(paths, load_from_json(paths.of("split") / "meta.json"))
    train = load_split(paths, "train")
    graph_arrays = load_arrays(paths.of("graph") / "graph.npz")
    assignments = load_df(
        paths.of("regions") / "assignments.parquet", filters=[("split", "==", "train")]
    )
    region = assignments.sort_values("row")["region"].to_numpy(dtype=np.int64)
    label = train["label"].to_numpy(dtype=np.int64)
    graph_rows = graph_arrays["rows"]
    train_graph = TrainGraph(
        space.embed(train.iloc[graph_rows]),
        graph_arrays["knn_idx"],
        graph_arrays["knn_dist"],
        graph_arrays["mst"],
    )

    graph_is_sample = len(graph_rows) < len(train)
    complexity_cfg = cfg.complexity
    centroids = load_df(paths.of("regions") / "centroids.parquet")
    centroid_coordinates = centroids.drop(columns=["region", "class_id"]).to_numpy()
    region_class = {
        int(r): int(c) for r, c in zip(centroids["region"], centroids["class_id"])
    }

    region_sample = _draw_sample(
        cfg,
        space=space,
        train=train,
        population=region,
        train_graph=train_graph,
        graph_rows=graph_rows,
        cap=complexity_cfg.max_sample_per_region if graph_is_sample else None,
    )
    region_measures = _measure(
        cfg,
        train_graph=train_graph,
        graph_rows=graph_rows,
        population=region,
        sample=region_sample,
        population_class=region_class,
        # Exact centroids: the ones regions routes by, not a sample's.
        centroids={
            int(r): point for r, point in zip(centroids["region"], centroid_coordinates)
        },
    )

    label_sample = _draw_sample(
        cfg,
        space=space,
        train=train,
        population=label,
        train_graph=train_graph,
        graph_rows=graph_rows,
        cap=complexity_cfg.max_sample_per_class if graph_is_sample else None,
    )
    class_measures = _measure(
        cfg,
        train_graph=train_graph,
        graph_rows=graph_rows,
        population=label,
        sample=label_sample,
        population_class={int(c): int(c) for c in np.unique(label)},
        centroids=compute_centroids(
            scale_for_metric(label_sample.X, cfg.distance),
            label_sample.population,
            metric=cfg.distance,
        ),
    )
    save_df(
        pd.DataFrame(
            [
                {"region": r, "class_id": region_class[r], **row}
                for r, row in region_measures.items()
            ]
        ),
        stage_dir / "regions.parquet",
    )
    save_df(
        pd.DataFrame([{"class_id": c, **row} for c, row in class_measures.items()]),
        stage_dir / "classes.parquet",
    )
    flush_timing(stage_dir / "timing.json")
    write_record(stage_dir, config=config, inputs=inputs)


if __name__ == "__main__":
    main()
