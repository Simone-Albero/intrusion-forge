import logging
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

from src.core.io import load_df, save_df
from src.core.log import setup_logger
from src.core.record import clear_dir, is_current, write_record
from src.core.utils import (
    flush_timing,
    save_to_joblib,
    save_to_json,
    timed,
)
from src.domain.analysis.metadata import build_meta
from src.domain.data.preprocessing import (
    LogTransformer,
    TopNHashEncoder,
    build_preprocessor,
    drop_nans,
    drop_rare_classes,
    encode_labels,
    ml_split,
    query_filter,
)
from stages import SPLITS, load_cli_config, paths_from_cfg, stage_config

setup_logger()
logger = logging.getLogger(__name__)


def _load_raw(cfg, raw_path: Path) -> tuple[pd.DataFrame, int]:
    """The raw frame and its column count, reading only the columns the stage uses."""
    n_columns = len(load_df(raw_path, nrows=0).columns)
    usecols = (
        None
        if cfg.data.filter_query
        else [*cfg.data.num_cols, *cfg.data.cat_cols, cfg.data.label_col]
    )
    return load_df(raw_path, usecols=usecols), n_columns


@timed
def build_splits(
    cfg, df: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], ColumnTransformer, dict[int, str]]:
    """Filter, split and preprocess the raw frame; labels become integer class ids."""
    num_cols, cat_cols = list(cfg.data.num_cols), list(cfg.data.cat_cols)
    label_col = cfg.data.label_col
    logger.info(
        "Preprocessing: %d rows, %d num_cols, %d cat_cols",
        len(df),
        len(num_cols),
        len(cat_cols),
    )
    df = drop_nans(df, num_cols + cat_cols + [label_col])
    df = query_filter(df, query=cfg.data.filter_query)
    df = drop_rare_classes(df, label_col, min_count=cfg.data.min_class_count)

    parts = ml_split(
        df,
        train_frac=cfg.data.train_frac,
        val_frac=cfg.data.val_frac,
        test_frac=cfg.data.test_frac,
        random_state=cfg.seed,
        label_col=label_col,
    )
    logger.info(
        "Split sizes — train: %d, val: %d, test: %d", *(len(part) for part in parts)
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
    preprocessor.fit(parts[0])

    frames = [
        preprocessor.transform(part)
        .assign(**{label_col: part[label_col].to_numpy()})
        .reset_index(drop=True)
        for part in parts
    ]
    *frames, label_mapping = encode_labels(
        *frames, src_label_col=label_col, dst_label_col="label"
    )
    keep = num_cols + cat_cols + ["label"]
    return (
        {name: frame[keep] for name, frame in zip(SPLITS, frames)},
        preprocessor,
        label_mapping,
    )


def main() -> None:
    """Entry point for the split stage."""
    cfg = load_cli_config()
    paths = paths_from_cfg(cfg)
    stage_dir = paths.of("split")
    raw_path = Path(cfg.path.raw_data)
    config = stage_config(cfg, "split")
    if is_current(stage_dir, config=config, inputs={}, force=cfg.force):
        return

    logger.info("Loading %s ...", raw_path)
    raw, n_raw_columns = _load_raw(cfg, raw_path)
    clear_dir(stage_dir)
    n_raw_rows = len(raw)
    logger.info("Raw data loaded: %d rows, %d columns", n_raw_rows, n_raw_columns)
    raw_classes = raw[cfg.data.label_col].value_counts()

    splits, preprocessor, label_mapping = build_splits(cfg, raw)
    del raw
    for name, frame in splits.items():
        save_df(frame, stage_dir / f"{name}.parquet")
    save_to_joblib(preprocessor, stage_dir / "preprocessor.joblib")
    save_to_json(
        build_meta(
            {name: frame["label"] for name, frame in splits.items()},
            num_cols=list(cfg.data.num_cols),
            cat_cols=list(cfg.data.cat_cols),
            label_mapping=label_mapping,
            raw_classes=raw_classes,
            n_raw_rows=n_raw_rows,
            n_raw_columns=n_raw_columns,
        ),
        stage_dir / "meta.json",
    )
    flush_timing(stage_dir / "timing.json")
    write_record(stage_dir, config=config, inputs={})


if __name__ == "__main__":
    main()
