import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


def _cluster_error_rates(clusters: np.ndarray, error_mask: np.ndarray) -> dict:
    """Return {cluster_id: {n_error, n_total, error_rate}} sorted by error_rate desc."""
    failed = clusters[error_mask]
    stats = {}
    for c in np.unique(clusters):
        n_total = int((clusters == c).sum())
        n_error = int((failed == c).sum())
        stats[str(c)] = {
            "n_error": n_error,
            "n_total": n_total,
            "error_rate": (n_error / n_total) if n_total > 0 else None,
        }
    return dict(
        sorted(stats.items(), key=lambda x: x[1]["error_rate"] or 0.0, reverse=True)
    )


def evaluate_predictions(
    df, y_true, y_pred, confidences, cluster_col: str = "cluster"
) -> dict:
    """
    Evaluate per-class prediction quality and cluster-level error rates.

    Returns:
        {
            "classes": {
                "<label>": {
                    "tot_failures", "tot_samples", "failure_rate",
                    "mean_confidence", "cluster_in_fn", "cluster_in_tp"
                }, ...
            },
            "clusters": {
                "global":   {<cluster_id>: {n_error, n_total, error_rate}, ...} | None,
                "by_class": {<label>: {<cluster_id>: {...}}, ...}              | None,
            },
        }
    """
    has_cluster = cluster_col in df.columns
    all_clusters = df[cluster_col].to_numpy() if has_cluster else None
    global_error_mask = y_true != y_pred

    cluster_errors_total = (
        _cluster_error_rates(all_clusters, global_error_mask) if has_cluster else None
    )
    cluster_errors_by_class = {} if has_cluster else None

    classes = {}
    for label in np.unique(y_true):
        mask = y_true == label
        n_total = int(mask.sum())
        n_errors = int((y_true[mask] != y_pred[mask]).sum())
        error_mask = mask & global_error_mask

        if has_cluster:
            wrong_preds = y_pred[error_mask]
            wrong_clusters = df[cluster_col].loc[df.index[error_mask]].to_numpy()
            cluster_in_fn = {
                str(cls): np.unique(wrong_clusters[wrong_preds == cls]).tolist()
                for cls in np.unique(wrong_preds)
            }
            tp_clusters = (
                df[cluster_col].loc[df.index[mask & ~global_error_mask]].to_numpy()
            )
            cluster_in_tp = np.unique(tp_clusters).tolist()

            class_clusters = df[cluster_col].loc[df.index[mask]].to_numpy()
            cluster_errors_by_class[str(label)] = _cluster_error_rates(
                class_clusters, error_mask[mask]
            )
        else:
            cluster_in_fn = cluster_in_tp = None

        classes[str(label)] = {
            "tot_failures": n_errors,
            "tot_samples": n_total,
            "failure_rate": n_errors / n_total if n_total > 0 else None,
            "mean_confidence": float(confidences[mask].mean()) if n_total > 0 else None,
            "cluster_in_fn": cluster_in_fn,
            "cluster_in_tp": cluster_in_tp,
        }

    classes = dict(
        sorted(classes.items(), key=lambda x: x[1]["failure_rate"] or 0.0, reverse=True)
    )

    return {
        "classes": classes,
        "clusters": {
            "global": cluster_errors_total,
            "by_class": cluster_errors_by_class,
        },
    }


def visualize_samples(
    X, y_1, y_2=None, exclude_classes=None, n_samples=2000, n_components=2
):
    mask = ~np.isin(y_1, exclude_classes or [])
    vis_mask = create_subsample_mask(y_1[mask], n_samples=n_samples, stratify=False)
    return samples_plot(
        tsne_projection(X[mask][vis_mask], n_components=n_components),
        y_1[mask][vis_mask],
        y_2[mask][vis_mask] if y_2 is not None else None,
    )


def visualize_cm(y_true, y_pred, normalize=None):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true), normalize=normalize)
    return confusion_matrix_to_plot(cm), cm


def infer():
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )
    json_logs_path = Path(cfg.path.json_logs)
    df_meta = load_from_json(json_logs_path / "data/df_meta.json")
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

    stats = {"class_confidence": {}, "pred_infos": {}, "cluster_failures": {}}
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

        cm_fig, cm = visualize_cm(y_true, y_pred, normalize=None)
        tb_logger.writer.add_figure("confusion_matrices/original", cm_fig, step)
        plt.close(cm_fig)

        save_to_pickle(
            cm, Path(cfg.path.pickle) / f"inference/confusion_matrices/{suffix}.pkl"
        )

        cm_fig, cm = visualize_cm(y_true, y_pred, normalize="true")
        tb_logger.writer.add_figure(
            "confusion_matrices/normalized_by_row", cm_fig, step
        )
        plt.close(cm_fig)

        for tag, data in (("raw/classes", X), ("latent/classes", z)):
            if data is None:
                continue
            correct = (y_pred == y_true).astype(int)

            fig = visualize_samples(data, y_true, correct, n_components=2)
            tb_logger.writer.add_figure(tag + "_2D", fig, step)

            fig = visualize_samples(data, y_true, correct, n_components=3)
            tb_logger.writer.add_figure(tag + "_3D", fig, step)

            plt.close(fig)

        logger.info("Counting failures per class ...")
        pred_infos = evaluate_predictions(df, y_true, y_pred, confidences)
        save_to_json(
            pred_infos,
            json_logs_path / f"inference/pred_infos/{suffix}.json",
        )
        stats["pred_infos"][suffix] = pred_infos

    return stats


if __name__ == "__main__":
    infer()
