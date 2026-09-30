import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

from src.core.config import load_config, save_config, to_container
from src.core.io import load_df, save_df
from src.core.log import (
    JSONSubscriber,
    LogBundle,
    LogDispatcher,
    setup_logger,
)
from src.core.utils import flush_timing, load_from_json, skip_if_unchanged, timed
from src.domain.analysis.complexity.shared import l2_normalize
from src.domain.analysis.metadata import compute_df_metadata, get_df_info
from src.domain.clustering import build_cluster_fn
from src.domain.clustering.base import (
    assign_nearest_centroid,
    cluster_size_balance,
    compute_centroids,
    subsample_indices,
)
from src.domain.clustering.hardness import KdnReference
from src.domain.data.preprocessing import (
    LogTransformer,
    TopNHashEncoder,
    build_preprocessor,
    drop_nans,
    encode_labels,
    ml_split,
    query_filter,
    rare_category_filter,
)

setup_logger()
logger = logging.getLogger(__name__)


def _cluster_per_class(
    cfg,
    X_num: np.ndarray,
    y_class: np.ndarray,
    *,
    classes: list,
    eval_rows_per_train_row: float,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[str, list]]:
    """Cluster each class at the finest granularity its region error rates would stay
    reliable at, merging any undersized cluster into a survivor."""
    n = X_num.shape[0]
    clustering = cfg.clustering
    algorithms = OmegaConf.to_container(clustering.algorithms, resolve=True)
    max_clusters_total = (
        cfg.complexity.max_complexity_samples
        // cfg.complexity.min_subsample_per_cluster
    )
    max_clusters_per_class = max(2, max_clusters_total // len(classes))
    labels = np.full(n, -1, dtype=np.int64)
    centroids: dict[int, np.ndarray] = {}
    offset = 0
    report: dict[str, list] = {"classes": [], "sweep": []}

    X_num_space = l2_normalize(X_num) if clustering.distance == "cosine" else X_num
    ref_idx = subsample_indices(
        n, max_samples=cfg.complexity.max_complexity_samples, random_state=cfg.seed
    )
    reference = KdnReference(X=X_num_space[ref_idx], y=y_class[ref_idx], ids=ref_idx)

    for cls in tqdm(classes, desc="Clustering classes"):
        mask = y_class == cls
        if not mask.any():
            continue
        X_num_cls = X_num_space[mask]
        ids_cls = np.flatnonzero(mask)

        algo_reports: dict[str, dict] = {}
        cluster_fn = build_cluster_fn(
            algorithms=algorithms,
            max_fit_samples=clustering.max_fit_samples,
            random_state=cfg.seed,
            reporter=algo_reports.__setitem__,
            max_clusters=max_clusters_per_class,
            min_cluster_floor=clustering.min_cluster_floor,
            hardness_k=clustering.hardness_k,
            reliability_target=clustering.reliability_target,
            eval_rows_per_train_row=eval_rows_per_train_row,
            reference=reference,
            metric=clustering.distance,
        )
        raw_labels, n_merged_clusters, n_merged = cluster_fn(
            X_num_cls, ids=ids_cls, label=cls
        )
        n_clusters_cls = int(np.unique(raw_labels).size)
        if n_clusters_cls > max_clusters_per_class:
            raise ValueError(
                f"class {cls!r}: {n_clusters_cls} clusters survive merging, over "
                f"max_clusters={max_clusters_per_class} for this class — the complexity "
                "stage's point budget cannot subsample this many. Raise "
                "max_complexity_samples, lower min_subsample_per_cluster, or tighten "
                "the clustering grid."
            )

        [algo_report] = algo_reports.values()
        # The winning candidate's own reliability, predicted on the scored subsample;
        # n_merged_clusters/n_merged above are the full class's, from the final merge.
        best = algo_report["best"]
        report["classes"].append(
            {
                "class_name": str(cls),
                "n_train": int(raw_labels.shape[0]),
                "n_clusters": n_clusters_cls,
                "reliability": best["reliability"],
                "size_balance": cluster_size_balance(raw_labels),
                "n_merged_clusters": n_merged_clusters,
                "n_merged": n_merged,
            }
        )
        report["sweep"].extend(
            {
                "class_name": str(cls),
                **{f"param_{k}": v for k, v in candidate["combo"].items()},
                "best": candidate["best"],
                "n_clusters": candidate["n_clusters"],
                "n_merged_clusters": candidate["n_merged_clusters"],
                "n_merged": candidate["n_merged"],
                "size_balance": candidate["size_balance"],
                "var_between": candidate["var_between"],
                "var_sampling": candidate["var_sampling"],
                "reliability": candidate["reliability"],
                "duration_s": candidate["duration_s"],
                "error": candidate.get("error", False),
            }
            for candidate in algo_report["sweep"]
        )

        cluster_ids = np.unique(raw_labels)
        labels[mask] = raw_labels + offset
        class_centroids = compute_centroids(
            X_num_cls, raw_labels, metric=clustering.distance
        )
        centroids.update(
            {int(cid) + offset: centroid for cid, centroid in class_centroids.items()}
        )
        offset += int(cluster_ids.max()) + 1

    return labels, centroids, report


@timed
def preprocess_df(
    cfg,
    df: pd.DataFrame,
    *,
    num_cols: list[str],
    cat_cols: list[str],
    label_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Preprocess dataframe: filter, encode, scale, and split."""
    logger.info(
        "Preprocessing: %d rows, %d num_cols, %d cat_cols",
        len(df),
        len(num_cols),
        len(cat_cols),
    )
    df = drop_nans(df, num_cols + cat_cols + [label_col])
    df = query_filter(df, query=cfg.data.filter_query)
    df = rare_category_filter(df, [label_col], min_count=cfg.data.min_cat_count)

    train_df, val_df, test_df = ml_split(
        df,
        train_frac=cfg.data.train_frac,
        val_frac=cfg.data.val_frac,
        test_frac=cfg.data.test_frac,
        random_state=cfg.seed,
        label_col=label_col,
    )
    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    preprocessor = build_preprocessor(
        num_cols=num_cols,
        cat_cols=cat_cols,
        num_steps=[
            ("log_transformer", LogTransformer()),
            ("scaler", RobustScaler()),
        ],
        cat_steps=[
            (
                "top_n_encoder",
                TopNHashEncoder(
                    top_n=cfg.data.top_n, hash_buckets=cfg.data.hash_buckets
                ),
            ),
        ],
    )
    logger.info("Preprocessor: %s", preprocessor)
    preprocessor.fit(train_df)
    # Only num_cols, cat_cols and the label travel past this point: no raw column the
    # pipeline never reads (IPs, ports, DNS ids, ...) survives into the parquet.
    train_df, val_df, test_df = (
        preprocessor.transform(split).assign(**{label_col: split[label_col].to_numpy()})
        for split in [train_df, val_df, test_df]
    )

    return train_df, val_df, test_df


def _cluster_splits(
    cfg,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    num_cols: list[str],
    label_col: str,
    dispatcher: LogDispatcher,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cluster train per class, then give every split `cluster` and `routed_cluster`."""
    X_num = train_df[num_cols].to_numpy(dtype=np.float64)
    y_class = train_df[label_col].to_numpy()
    all_classes = sorted(train_df[label_col].unique().tolist())

    # How many rows classify's evaluation will measure each region on, per train row it
    # holds: only the test rows are ever evaluated.
    n_train, n_test = len(train_df), len(test_df)
    eval_rows_per_train_row = n_test / n_train
    if eval_rows_per_train_row <= 0:
        raise ValueError(
            f"eval_rows_per_train_row={eval_rows_per_train_row}: classify would "
            "evaluate every region on zero rows."
        )

    logger.info("Running per-class clustering on train (n=%d)...", len(train_df))
    labels, centroids, clustering_report = _cluster_per_class(
        cfg,
        X_num,
        y_class,
        classes=all_classes,
        eval_rows_per_train_row=eval_rows_per_train_row,
    )
    report_tables = {
        "metric": cfg.clustering.distance,
        "algorithm": next(iter(cfg.clustering.algorithms)),
        "eval_rows_per_train_row": eval_rows_per_train_row,
        **clustering_report,
    }
    dispatcher.publish(LogBundle.from_dict({"json/clustering_report": report_tables}))

    # Routing never reads the label: a region drawn inside one class holds only rows of
    # that class, and its error rate would count only the mistakes made on it.
    routed: dict[str, pd.DataFrame] = {}
    for name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        split_df = split_df.copy()
        split_df["routed_cluster"] = assign_nearest_centroid(
            split_df[num_cols].to_numpy(dtype=np.float64),
            centroids,
            metric=cfg.clustering.distance,
        )
        # Only train rows were clustered; the others belong where they are routed.
        split_df["cluster"] = labels if name == "train" else split_df["routed_cluster"]
        routed[name] = split_df
    train_df, val_df, test_df = routed["train"], routed["val"], routed["test"]

    logger.info("Clustering complete — %d clusters", len(centroids))
    return train_df, val_df, test_df


def _publish_metadata(
    cfg,
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    encoded_label_col: str,
    label_mapping: dict,
    dispatcher: LogDispatcher,
) -> None:
    logger.info("Computing and saving metadata...")
    metadata = compute_df_metadata(
        {"train": train_df, "val": val_df, "test": test_df},
        label_col=encoded_label_col,
        num_cols=num_cols,
        cat_cols=cat_cols,
        benign_tag=cfg.data.benign_tag,
        label_mapping=label_mapping,
    )
    dispatcher.publish(LogBundle.from_dict({"json/df_meta": metadata}))


@timed
def prepare(cfg) -> None:
    """Preprocess, cluster and persist the train/val/test splits."""
    num_cols = list(cfg.data.num_cols)
    cat_cols = list(cfg.data.cat_cols)
    label_col = cfg.data.label_col

    raw_data_path = Path(cfg.path.raw_data)
    processed_data_path = Path(cfg.path.processed_data)
    data_logs_path = Path(cfg.path.shared)

    dispatcher = LogDispatcher()
    dispatcher.subscribe(JSONSubscriber(data_logs_path / "metadata"))

    logger.info("Loading and preprocessing data...")
    df = load_df(str(raw_data_path))
    logger.info("Raw data loaded: %d rows, %d columns", *df.shape)

    df_info = get_df_info(df, label_col=label_col)
    dispatcher.publish(LogBundle.from_dict({"json/df_info": df_info}))

    train_df, val_df, test_df = preprocess_df(
        cfg, df, num_cols=num_cols, cat_cols=cat_cols, label_col=label_col
    )
    train_df, val_df, test_df = (
        df.reset_index(drop=True) for df in [train_df, val_df, test_df]
    )

    train_df, val_df, test_df = _cluster_splits(
        cfg,
        train_df,
        val_df,
        test_df,
        num_cols=num_cols,
        label_col=label_col,
        dispatcher=dispatcher,
    )

    encoded_label_col = f"encoded_{label_col}"
    train_df, val_df, test_df, label_mapping = encode_labels(
        train_df,
        val_df,
        test_df,
        src_label_col=label_col,
        dst_label_col=encoded_label_col,
    )

    logger.info("Saving processed data...")
    for split_name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        save_df(split_df, processed_data_path / f"{split_name}.{cfg.data.extension}")

    _publish_metadata(
        cfg,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        num_cols=num_cols,
        cat_cols=cat_cols,
        encoded_label_col=encoded_label_col,
        label_mapping=label_mapping,
        dispatcher=dispatcher,
    )


def _fingerprint(cfg) -> dict:
    """The config the splits and regions are built from."""
    return {
        # Bumped when the code changes what a config builds: older records never match.
        "schema": 7,
        "data": to_container(cfg.data),
        "clustering": to_container(cfg.clustering),
        "seed": cfg.seed,
        # The per-class region budget is derived from the complexity sample cap.
        "complexity": {
            key: cfg.complexity[key]
            for key in ("max_complexity_samples", "min_subsample_per_cluster")
        },
    }


def main() -> None:
    """Entry point for the data preparation stage."""
    cfg = load_config(
        config_path=Path(__file__).parent.parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )

    if cfg.clustering.distance not in ("euclidean", "cosine"):
        raise ValueError(
            f"Unknown clustering.distance: {cfg.clustering.distance!r}. "
            "Valid: 'euclidean', 'cosine'."
        )
    ext = cfg.data.extension
    processed = Path(cfg.path.processed_data)
    shared = Path(cfg.path.shared)
    snapshot = shared / "config_composed_prepare.json"
    outputs = [processed / f"{s}.{ext}" for s in ("train", "val", "test")]
    outputs += [
        shared / f"metadata/{name}.json"
        for name in ("df_info", "df_meta", "clustering_report")
    ]
    outputs.append(snapshot)
    record = shared / "prepare_fingerprint.json"
    raw_data = Path(cfg.path.raw_data)
    fingerprint = _fingerprint(cfg)
    # Size and mtime, not a digest: re-reading a CSV of several GB only to decide
    # whether to read it would cost what the cache saves.
    if raw_data.exists():
        stat = raw_data.stat()
        fingerprint["raw_data"] = {"size": stat.st_size, "mtime": stat.st_mtime}
    elif record.exists():
        # Outputs copied without their raw CSV: only the config can still be checked.
        fingerprint["raw_data"] = load_from_json(record)["raw_data"]
        logger.warning(
            "Missing %s: the prepare cache checks the config alone.", raw_data
        )
    if skip_if_unchanged(
        outputs, record, fingerprint, force=cfg.force, stage_name="prepare"
    ):
        return
    if not raw_data.exists():
        raise FileNotFoundError(
            f"Missing {raw_data}: prepare must recompute and cannot without it."
        )

    # Dropped first: an interrupted recompute leaves new outputs under an old record.
    record.unlink(missing_ok=True)
    prepare(cfg)
    flush_timing(shared / "timing.json")
    save_config(cfg, snapshot)
    # Last: the record vouches for everything above, the snapshot included.
    bus = LogDispatcher()
    bus.subscribe(JSONSubscriber(shared))
    bus.publish(LogBundle.from_dict({"json/prepare_fingerprint": fingerprint}))


if __name__ == "__main__":
    main()
