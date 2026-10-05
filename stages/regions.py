import logging

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from src.core.io import load_arrays, save_df
from src.core.log import setup_logger
from src.core.record import clear_dir, is_current, write_record
from src.core.utils import flush_timing, load_from_json, save_to_json, timed
from src.domain.analysis.complexity.shared import scale_for_metric
from src.domain.clustering import build_cluster_fn
from src.domain.clustering.base import (
    assign_nearest_centroid,
    cluster_size_balance,
    compute_centroids,
)
from src.domain.clustering.hardness import KdnReference
from src.domain.data.space import Space
from stages import (
    SPLITS,
    load_cli_config,
    load_space,
    load_split,
    paths_from_cfg,
    stage_config,
    upstream_ids,
)

setup_logger()
logger = logging.getLogger(__name__)

ROUTE_CHUNK = 50_000


def _points(cfg, space: Space, rows: pd.DataFrame) -> np.ndarray:
    """The rows in the space regions are drawn in: unit vectors under cosine."""
    return scale_for_metric(space.embed(rows), cfg.distance)


def _cluster_per_class(
    cfg,
    space: Space,
    train: pd.DataFrame,
    *,
    reference_rows: np.ndarray,
    class_names: dict[int, str],
    eval_rows_per_train_row: float,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[str, list]]:
    """Cluster each class at the granularity that least misreads its rows' hardness from
    their region's error rate, merging any undersized cluster into a survivor."""
    y_class = train["label"].to_numpy()
    classes = sorted(np.unique(y_class).tolist())
    n = len(train)
    clustering = cfg.clustering
    algorithms = OmegaConf.to_container(clustering.algorithms, resolve=True)
    max_clusters_per_class = max(2, clustering.max_regions // len(classes))
    labels = np.full(n, -1, dtype=np.int64)
    centroids: dict[int, np.ndarray] = {}
    offset = 0
    report: dict[str, list] = {"classes": [], "sweep": []}

    reference = KdnReference(
        X=_points(cfg, space, train.iloc[reference_rows]),
        y=y_class[reference_rows],
        ids=reference_rows,
    )

    for cls in tqdm(classes, desc="Clustering classes"):
        mask = y_class == cls
        if not mask.any():
            continue
        ids_cls = np.flatnonzero(mask)
        X_num_cls = _points(cfg, space, train.iloc[ids_cls])

        algo_reports: dict[str, dict] = {}
        cluster_fn = build_cluster_fn(
            algorithms=algorithms,
            max_fit_samples=clustering.max_fit_samples,
            random_state=cfg.seed,
            reporter=algo_reports.__setitem__,
            max_clusters=max_clusters_per_class,
            min_cluster_floor=clustering.min_cluster_floor,
            hardness_k=clustering.hardness_k,
            eval_rows_per_train_row=eval_rows_per_train_row,
            reference=reference,
            metric=cfg.distance,
        )
        raw_labels, n_merged_clusters, n_merged = cluster_fn(
            X_num_cls, ids=ids_cls, label=cls
        )
        n_clusters_cls = int(np.unique(raw_labels).size)
        if n_clusters_cls > max_clusters_per_class:
            raise ValueError(
                f"class {class_names[cls]!r}: {n_clusters_cls} clusters survive merging, "
                f"over its share of clustering.max_regions ({max_clusters_per_class}). "
                "Raise max_regions or tighten the clustering grid."
            )

        [algo_report] = algo_reports.values()
        best = algo_report["best"]
        identity = {"class_id": int(cls), "class_name": class_names[cls]}
        report["classes"].append(
            {
                **identity,
                "n_train": int(raw_labels.shape[0]),
                "n_clusters": n_clusters_cls,
                "loss": best["loss"],
                "size_balance": cluster_size_balance(raw_labels),
                "n_merged_clusters": n_merged_clusters,
                "n_merged": n_merged,
            }
        )
        report["sweep"].extend(
            {
                **identity,
                **{f"param_{k}": v for k, v in candidate["combo"].items()},
                "best": candidate["best"],
                "n_clusters": candidate["n_clusters"],
                "n_merged_clusters": candidate["n_merged_clusters"],
                "n_merged": candidate["n_merged"],
                "size_balance": candidate["size_balance"],
                "loss": candidate["loss"],
                "loss_within": candidate["loss_within"],
                "loss_noise": candidate["loss_noise"],
                "duration_s": candidate["duration_s"],
                "error": candidate.get("error", False),
            }
            for candidate in algo_report["sweep"]
        )

        cluster_ids = np.unique(raw_labels)
        labels[mask] = raw_labels + offset
        class_centroids = compute_centroids(X_num_cls, raw_labels, metric=cfg.distance)
        centroids.update(
            {int(cid) + offset: centroid for cid, centroid in class_centroids.items()}
        )
        offset += int(cluster_ids.max()) + 1

    return labels, centroids, report


def _route(cfg, space: Space, rows: pd.DataFrame, centroids: dict) -> np.ndarray:
    """Each row's nearest centroid, a chunk of rows at a time."""
    region = np.empty(len(rows), dtype=np.int64)
    for start in range(0, len(rows), ROUTE_CHUNK):
        chunk = rows.iloc[start : start + ROUTE_CHUNK]
        region[start : start + len(chunk)] = assign_nearest_centroid(
            space.embed(chunk), centroids, metric=cfg.distance
        )
    return region


@timed
def build_regions(
    cfg,
    splits: dict[str, pd.DataFrame],
    *,
    meta: dict,
    space: Space,
    reference_rows: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Cluster train per class, then place every split's rows in the nearest region."""
    train = splits["train"]
    y_class = train["label"].to_numpy()
    class_names = {c["class_id"]: c["class_name"] for c in meta["classes"]}

    eval_rows_per_train_row = len(splits["test"]) / len(train)
    if eval_rows_per_train_row <= 0:
        raise ValueError(
            f"eval_rows_per_train_row={eval_rows_per_train_row}: classify would "
            "evaluate every region on zero rows."
        )

    logger.info("Running per-class clustering on train (n=%d)...", len(train))
    labels, centroids, clustering_report = _cluster_per_class(
        cfg,
        space,
        train,
        reference_rows=reference_rows,
        class_names=class_names,
        eval_rows_per_train_row=eval_rows_per_train_row,
    )
    report = {
        "metric": cfg.distance,
        "algorithm": next(iter(cfg.clustering.algorithms)),
        "eval_rows_per_train_row": eval_rows_per_train_row,
        **clustering_report,
    }

    if (labels < 0).any():
        raise ValueError(f"{int((labels < 0).sum())} train rows were given no region.")
    region_ids = sorted(centroids)
    region_class = pd.Series(y_class).groupby(labels).first()
    centroid_table = pd.DataFrame(
        np.stack([centroids[r] for r in region_ids]),
        columns=space.columns(),
    )
    if not np.isfinite(centroid_table.to_numpy()).all():
        raise ValueError("A region centroid is not finite: a column of the space is.")
    centroid_table.insert(0, "class_id", region_class.loc[region_ids].to_numpy())
    centroid_table.insert(0, "region", region_ids)

    nearest = {name: _route(cfg, space, splits[name], centroids) for name in SPLITS}
    assigned = {"train": labels, "val": nearest["val"], "test": nearest["test"]}
    assignments = pd.concat(
        [
            pd.DataFrame(
                {
                    "split": name,
                    "row": np.arange(len(assigned[name]), dtype=np.int32),
                    "region": assigned[name].astype(np.int32),
                    "nearest_region": nearest[name].astype(np.int32),
                }
            )
            for name in SPLITS
        ],
        ignore_index=True,
    )
    assignments["split"] = pd.Categorical(assignments["split"], categories=SPLITS)
    logger.info("Clustering complete — %d regions", len(centroids))
    return centroid_table, assignments, report


def main() -> None:
    """Entry point for the regions stage."""
    cfg = load_cli_config()
    paths = paths_from_cfg(cfg)
    stage_dir = paths.of("regions")
    config = stage_config(cfg, "regions")
    inputs = upstream_ids(cfg, paths, "regions")
    if is_current(stage_dir, config=config, inputs=inputs, force=cfg.force):
        return

    clear_dir(stage_dir)
    meta = load_from_json(paths.of("split") / "meta.json")
    splits = {name: load_split(paths, name) for name in SPLITS}
    reference_rows = load_arrays(paths.of("graph") / "graph.npz")["rows"]
    centroids, assignments, report = build_regions(
        cfg,
        splits,
        meta=meta,
        space=load_space(paths, meta),
        reference_rows=reference_rows,
    )
    save_df(centroids, stage_dir / "centroids.parquet")
    save_df(assignments, stage_dir / "assignments.parquet")
    save_to_json(report, stage_dir / "report.json")
    flush_timing(stage_dir / "timing.json")
    write_record(stage_dir, config=config, inputs=inputs)


if __name__ == "__main__":
    main()
