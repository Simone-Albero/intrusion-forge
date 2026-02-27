import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ignite.handlers.tensorboard_logger import TensorboardLogger
from sklearn.metrics import confusion_matrix

from src.common.config import load_config
from src.common.logging import setup_logger
from src.common.utils import load_from_json, save_to_json, save_to_pickle

from src.data.analyze import sample_distances
from src.data.io import load_listed_dfs

from src.ml.projection import create_subsample_mask, tsne_projection

from src.plot.array import confusion_matrix_to_plot, samples_plot

from src.torch.builders import create_model
from src.torch.infer import df_to_tensors, get_predictions
from src.torch.module.checkpoint import load_latest_checkpoint

setup_logger()
logger = logging.getLogger(__name__)


def compute_distance_stats(distances: np.ndarray) -> dict:
    if distances.size == 0:
        return {"mean": None, "median": None, "p90": None, "n": 0}
    return {
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "p90": float(np.quantile(distances, 0.9)),
        "n": int(distances.size),
    }


def analyze_class_failures(X, y_true, y_pred, target_class, max_samples=1000) -> dict:
    mask = y_true == target_class
    idx_tp = np.where(mask & (y_pred == target_class))[0]
    idx_fn = np.where(mask & (y_pred != target_class))[0]
    if len(idx_tp) > max_samples:
        idx_tp = np.random.choice(idx_tp, size=max_samples, replace=False)
    if len(idx_fn) > max_samples:
        idx_fn = np.random.choice(idx_fn, size=max_samples, replace=False)
    n_pairs = min(len(idx_tp), len(idx_fn))
    return {
        "target_class": int(target_class),
        "n_tp": len(idx_tp),
        "n_fn": len(idx_fn),
        "tp_tp": compute_distance_stats(sample_distances(X, idx_tp, max_pairs=n_pairs)),
        "fn_fn": compute_distance_stats(sample_distances(X, idx_fn, max_pairs=n_pairs)),
        "fn_tp": compute_distance_stats(
            sample_distances(X, idx_fn, idx_tp, max_pairs=n_pairs)
        ),
    }


def analyze_classes_failures(X, y_true, y_pred, max_samples=1000) -> list[dict]:
    return [
        analyze_class_failures(X, y_true, y_pred, cls, max_samples)
        for cls in np.unique(y_true)
    ]


def count_cluster_failures(
    df, y_true, y_pred, confidences, cluster_col: str = "cluster"
) -> list[dict]:
    rows = []
    for c_label in df[cluster_col].unique():
        mask = (df[cluster_col] == c_label).to_numpy()
        total = int(mask.sum())
        failures = int((y_true[mask] != y_pred[mask]).sum())
        rows.append(
            {
                "cluster": int(c_label),
                "failures": failures,
                "total": total,
                "failure_rate": failures / total if total > 0 else None,
                "mean_confidence": (
                    float(confidences[mask].mean()) if total > 0 else None
                ),
            }
        )
    return sorted(rows, key=lambda r: r["failure_rate"] or 0.0)


def visualize_samples(X, y_1, y_2=None, exclude_classes=None, n_samples=3000):
    mask = ~np.isin(y_1, exclude_classes or [])
    vis_mask = create_subsample_mask(y_1[mask], n_samples=n_samples, stratify=False)
    return samples_plot(
        tsne_projection(X[mask][vis_mask]),
        y_1[mask][vis_mask],
        y_2[mask][vis_mask] if y_2 is not None else None,
    )


def visualize_cm(y_true, y_pred, normalize=None):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    return confusion_matrix_to_plot(cm, normalize=normalize), cm


def infer():
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )
    json_logs_path = Path(cfg.path.json_logs)
    df_meta = load_from_json(json_logs_path / "data/metadata.json")
    cfg.model.params.num_classes = df_meta["num_classes"]
    cfg.loss.params.class_weight = df_meta["class_weights"]
    device = torch.device(cfg.device)
    num_cols = list(cfg.data.num_cols) if cfg.data.num_cols else []
    cat_cols = list(cfg.data.cat_cols) if cfg.data.cat_cols else []
    label_col = "encoded_" + cfg.data.label_col
    feat_cols = num_cols + cat_cols

    splits = load_listed_dfs(
        Path(cfg.path.processed_data),
        [f"{s}.{cfg.data.extension}" for s in ("train", "val", "test")],
    )
    model = create_model(cfg.model.name, cfg.model.params, device)
    load_latest_checkpoint(Path(cfg.path.models), model, device)

    stats = {"class_confidence": {}, "class_failures": {}, "cluster_failures": {}}
    for suffix, df in zip(("train", "val", "test"), splits):
        logger.info(f"Running inference on {suffix} set ...")
        log_dir = Path(cfg.path.tb_logs) / "inference" / suffix
        log_dir.mkdir(parents=True, exist_ok=True)
        tb_logger = TensorboardLogger(log_dir=log_dir)
        step = cfg.run_id or 0
        X = df[feat_cols].to_numpy()

        *inputs, y = df_to_tensors(
            df,
            [num_cols, cat_cols, [label_col]],
            [torch.float32, torch.long, torch.long],
        )
        y_true, y_pred, z, confidences = get_predictions(model, inputs, y, device)

        y_true, y_pred, z, confidences = (
            y_true.cpu().numpy(),
            y_pred.cpu().numpy(),
            z.cpu().numpy() if z is not None else None,
            confidences.cpu().numpy(),
        )

        class_confidence = {
            int(cls): float(confidences[y_true == cls].mean())
            for cls in np.unique(y_true)
        }

        for cls, conf in class_confidence.items():
            logger.info(f"Class {cls}: Mean Confidence {conf:.4f}")
        save_to_json(
            class_confidence, json_logs_path / f"inference/confidence/{suffix}.json"
        )
        stats["class_confidence"][suffix] = class_confidence

        cm_fig, cm = visualize_cm(y_true, y_pred, normalize=None)
        tb_logger.writer.add_figure("confusion_matrix", cm_fig, step)
        plt.close(cm_fig)

        save_to_pickle(
            cm, json_logs_path / f"inference/confusion_matrices/{suffix}.pkl"
        )

        for tag, data in (("raw/classes", X), ("latent/classes", z)):
            if data is None:
                continue
            fig = visualize_samples(data, y_pred, y_true)
            tb_logger.writer.add_figure(tag, fig, step)
            plt.close(fig)

        if "cluster" in df.columns:
            clusters = df["cluster"].to_numpy()
            for tag, data in (("raw/clusters", X), ("latent/clusters", z)):
                if data is None:
                    continue
                fig = visualize_samples(data, clusters, y_true)
                tb_logger.writer.add_figure(tag, fig, step)
                plt.close(fig)

        tb_logger.close()

        logger.info("Analyzing class failures ...")
        class_failures = analyze_classes_failures(X, y_true, y_pred)
        logger.info(f"\n{pd.DataFrame(class_failures)}")
        save_to_json(
            class_failures, json_logs_path / f"inference/class_failures/{suffix}.json"
        )
        stats["class_failures"][suffix] = class_failures

        if "cluster" in df.columns:
            logger.info("Counting failures per cluster ...")
            cluster_failures = count_cluster_failures(df, y_true, y_pred, confidences)
            logger.info(f"\n{pd.DataFrame(cluster_failures)}")
            save_to_json(
                cluster_failures,
                json_logs_path / f"inference/cluster_failures/{suffix}.json",
            )
            stats["cluster_failures"][suffix] = cluster_failures

    return stats


if __name__ == "__main__":
    infer()
