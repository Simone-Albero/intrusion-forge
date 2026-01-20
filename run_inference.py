import json
from typing import Optional
from pathlib import Path
import logging
import sys

import numpy as np
import torch
import torch.nn as nn
from ignite.handlers.tensorboard_logger import TensorboardLogger
from sklearn.metrics.pairwise import paired_distances, pairwise_distances
from sklearn.metrics import confusion_matrix

from src.common.config import load_config
from src.common.logging import setup_logger

from src.data.io import load_data_splits
from src.data.preprocessing import (
    subsample_df,
    random_oversample_df,
    random_undersample_df,
    query_filter,
)

from src.torch.module.checkpoint import load_latest_checkpoint
from src.torch.builders import create_model

from src.ml.projection import tsne_projection, create_subsample_mask

from src.plot.array import vectors_plot
from src.plot.array import confusion_matrix_to_plot

setup_logger()
logger = logging.getLogger(__name__)

# Constants
EPSILON = 1e-8


def prepare_data(cfg, filter_query: Optional[str] = None):
    """Prepare data for PyTorch."""
    logger.info("Preparing data for PyTorch...")

    num_cols = list(cfg.data.num_cols)
    cat_cols = list(cfg.data.cat_cols)
    label_col = "multi_" + cfg.data.label_col

    base_path = Path(cfg.path.processed_data)
    train_df, _, test_df = load_data_splits(
        base_path, cfg.data.file_name, cfg.data.extension
    )

    train_df = query_filter(train_df, filter_query)
    test_df = query_filter(test_df, filter_query)

    x_num_train = torch.tensor(train_df[num_cols].to_numpy(dtype=np.float32))
    x_cat_train = torch.tensor(train_df[cat_cols].to_numpy(dtype=np.int64))
    y_train = torch.tensor(train_df[label_col].to_numpy(dtype=np.int64))

    x_num_test = torch.tensor(test_df[num_cols].to_numpy(dtype=np.float32))
    x_cat_test = torch.tensor(test_df[cat_cols].to_numpy(dtype=np.int64))
    y_test = torch.tensor(test_df[label_col].to_numpy(dtype=np.int64))

    logger.info("Data preparation for PyTorch completed.")
    return (x_num_train, x_cat_train, y_train), (x_num_test, x_cat_test, y_test)


def forward(
    model: nn.Module, x_numerical: torch.Tensor, x_categorical: torch.Tensor
) -> torch.Tensor:
    """Perform a forward pass through the model."""
    model.eval()
    with torch.no_grad():
        output = model(x_numerical, x_categorical)
    return output


def failure_analysis_for_class(
    X, y_true, y_pred, target_class, metric="euclidean", max_samples=1000
):
    """
    Perform failure analysis for a specific class by sampling distances
    between true positives and false negatives:
     - fn - fn: small distances may indicate clustered failures
     - fn - tp: high distances may indicate hard-to-classify points, far away from correct ones
    """
    logger.info(f"Computing failure analysis for class {target_class}...")
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask_class = y_true == target_class
    idx_class = np.where(mask_class)[0]

    idx_tp = np.where(mask_class & (y_pred == target_class))[0]
    idx_fn = np.where(mask_class & (y_pred != target_class))[0]

    # Subsample if needed
    n_tp = min(len(idx_tp), max_samples)
    n_fn = min(len(idx_fn), max_samples)

    if n_tp < len(idx_tp):
        idx_tp = np.random.choice(idx_tp, size=n_tp, replace=False)
    if n_fn < len(idx_fn):
        idx_fn = np.random.choice(idx_fn, size=n_fn, replace=False)

    def compute_paired_distances(idx_pool, n_pairs, metric):
        """Helper to compute distances between random pairs from the same pool."""
        if len(idx_pool) < 2 or n_pairs < 1:
            return np.array([])

        # Sample pairs ensuring idx_a != idx_b
        n_pairs = min(n_pairs, len(idx_pool) * (len(idx_pool) - 1) // 2)
        pairs = set()
        while len(pairs) < n_pairs:
            a, b = np.random.choice(idx_pool, size=2, replace=False)
            pairs.add((min(a, b), max(a, b)))

        pairs = list(pairs)
        idx_a = np.array([p[0] for p in pairs])
        idx_b = np.array([p[1] for p in pairs])

        return paired_distances(X[idx_a], X[idx_b], metric=metric)

    def compute_cross_distances(idx_a_pool, idx_b_pool, n_pairs, metric):
        """Helper to compute distances between pairs from different pools."""
        if len(idx_a_pool) < 1 or len(idx_b_pool) < 1 or n_pairs < 1:
            return np.array([])

        n_pairs = min(n_pairs, len(idx_a_pool), len(idx_b_pool))
        idx_a = np.random.choice(idx_a_pool, size=n_pairs, replace=False)
        idx_b = np.random.choice(idx_b_pool, size=n_pairs, replace=False)

        return paired_distances(X[idx_a], X[idx_b], metric=metric)

    # Compute distances
    d_fn_fn = compute_paired_distances(idx_fn, n_fn, metric)
    d_tp_tp = compute_paired_distances(idx_tp, n_tp, metric)
    d_fn_tp = compute_cross_distances(idx_fn, idx_tp, min(n_fn, n_tp), metric)

    def report(d):
        """Generate statistics report for distance array."""
        if d.size == 0:
            return {"mean": None, "median": None, "p90": None, "n": 0}
        return {
            "mean": float(np.mean(d)),
            "median": float(np.median(d)),
            "p90": float(np.quantile(d, 0.9)),
            "n": int(d.size),
        }

    stats = {
        "target_class": int(target_class),
        "class_samples": int(len(idx_class)),
        "n_tp": int(len(idx_tp)),
        "n_fn": int(len(idx_fn)),
        "tp_tp": report(d_tp_tp),
        "fn_fn": report(d_fn_fn),
        "fn_tp": report(d_fn_tp),
    }

    return stats


def perform_inference(
    cfg,
    x_numerical,
    x_categorical,
    y,
    device,
    tb_log_dir: Path,
    json_log_dir: Path,
    suffix: str = "",
) -> None:
    logger.info("Running inference...")
    VISUALIZATION_SAMPLES = 3000
    EVALUATION_SAMPLES = 10_000

    tb_logger = TensorboardLogger(log_dir=tb_log_dir)

    model = create_model(cfg.model.name, cfg.model.params, device)
    checkpoint_dir = Path(cfg.path.models)
    load_latest_checkpoint(checkpoint_dir, model, device)

    output = forward(
        model,
        x_numerical.to(device),
        x_categorical.to(device),
    )

    logits = output["logits"].cpu().numpy()
    z = output["z"].cpu().numpy() if "z" in output else None
    y_pred = logits.argmax(axis=1)
    y_true = y.numpy()
    unique_classes = np.unique(y_true)

    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    cm_figure = confusion_matrix_to_plot(
        cm=cm,
        title="Confusion Matrix",
        normalize="true",
    )
    tb_logger.writer.add_figure(
        f"{suffix}/confusion_matrix", cm_figure, cfg.get("run_id", 0)
    )

    all_stats = []
    for label in unique_classes:
        stats = failure_analysis_for_class(
            X=z,
            y_true=y_true,
            y_pred=y_pred,
            target_class=label,
            metric="euclidean",
            max_samples=EVALUATION_SAMPLES,
        )
        logger.info(f"Failure analysis for class {label}:\n{stats}")
        tb_logger.writer.add_text(
            f"failure_analysis_{suffix}/class_{label}", json.dumps(stats, indent=2)
        )
        all_stats.append(stats)

    json_path = (
        json_log_dir
        / f"failure_analysis{'_run_' + str(cfg.get('run_id')) if cfg.get('run_id') is not None else ''}.json"
    )
    with open(json_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    visualizzation_mask = create_subsample_mask(
        labels=y_true,
        n_samples=VISUALIZATION_SAMPLES,
        stratify=True,
    )
    reduced_z = tsne_projection(z[visualizzation_mask])
    latent_figure = vectors_plot(reduced_z, y_true[visualizzation_mask])
    tb_logger.writer.add_figure(f"{suffix}/latent_space", latent_figure)
    tb_logger.close()
    logger.info("Inference completed.")


def main():
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )

    df_meta = json.load(open(Path(cfg.path.json_logs) / "df_metadata.json", "r"))
    # cfg.data.num_cols = df_meta["numerical_columns"]
    # cfg.data.cat_cols = df_meta["categorical_columns"]
    # cfg.model.params.num_numerical_features = len(df_meta["numerical_columns"])
    # cfg.model.params.num_categorical_features = len(df_meta["categorical_columns"])
    cfg.model.params.num_classes = df_meta["num_classes"]
    cfg.loss.params.class_weight = df_meta["class_weights"]

    device = torch.device(cfg.device)
    logger.info(f"Using device: {device}")

    tb_log_dir = Path(cfg.path.tb_logs)
    tb_log_dir.mkdir(parents=True, exist_ok=True)

    json_log_dir = Path(cfg.path.json_logs)
    json_log_dir.mkdir(parents=True, exist_ok=True)

    filter_query = cfg.get("filter_query", None)
    (x_num_train, x_cat_train, y_train), (x_num_test, x_cat_test, y_test) = (
        prepare_data(cfg, filter_query=filter_query)
    )

    perform_inference(
        cfg,
        x_num_train,
        x_cat_train,
        y_train,
        device,
        tb_log_dir,
        json_log_dir,
        suffix="train",
    )

    perform_inference(
        cfg,
        x_num_test,
        x_cat_test,
        y_test,
        device,
        tb_log_dir,
        json_log_dir,
        suffix="test",
    )


if __name__ == "__main__":
    main()
