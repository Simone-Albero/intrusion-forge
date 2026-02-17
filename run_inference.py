import shutil
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.special import softmax
from sklearn.discriminant_analysis import unique_labels
import torch
from ignite.handlers.tensorboard_logger import TensorboardLogger
from sklearn.metrics.pairwise import paired_distances
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from src.common.config import load_config
from src.common.logging import setup_logger
from src.common.utils import save_to_json, save_to_pickle, load_from_json
from src.data.io import load_data_splits
from src.torch.module.checkpoint import load_latest_checkpoint
from src.torch.builders import create_model
from src.ml.projection import tsne_projection, create_subsample_mask
from src.plot.array import samples_plot, confusion_matrix_to_plot

setup_logger()
logger = logging.getLogger(__name__)


def to_tensors(df, num_cols, cat_cols, label_col):
    return (
        torch.tensor(df[num_cols].to_numpy(dtype=np.float32)),
        torch.tensor(df[cat_cols].to_numpy(dtype=np.int64)),
        torch.tensor(df[label_col].to_numpy(dtype=np.int64)),
    )


def load_data(
    processed_data_path,
    file_name,
    extension,
):
    """Load and convert data to PyTorch tensors."""
    train_df, val_df, test_df = load_data_splits(
        processed_data_path, file_name, extension
    )

    return train_df, val_df, test_df


def run_inference(model, x_numerical, x_categorical, device):
    """Run model inference."""
    model.eval()
    with torch.no_grad():
        if x_categorical.size(1) == 0:
            return model(x_numerical.to(device))
        elif x_numerical.size(1) == 0:
            return model(x_categorical.to(device))
        return model(x_numerical.to(device), x_categorical.to(device))


def compute_distance_stats(distances):
    """Compute statistics for distance array."""
    if distances.size == 0:
        return {"mean": None, "median": None, "p90": None, "n": 0}
    return {
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "p90": float(np.quantile(distances, 0.9)),
        "n": int(distances.size),
    }


def sample_paired_distances(X, idx_pool, n_pairs):
    """Sample distances between pairs from the same pool."""
    if len(idx_pool) < 2:
        return np.array([])

    n_pairs = min(n_pairs, len(idx_pool))
    i = np.random.choice(idx_pool, size=n_pairs, replace=False)
    j = np.random.choice(idx_pool, size=n_pairs, replace=False)
    mask = i != j

    if mask.sum() == 0:
        return np.array([])

    return paired_distances(X[i[mask]], X[j[mask]])


def sample_cross_distances(X, idx_a, idx_b, n_pairs):
    """Sample distances between pairs from different pools."""
    if len(idx_a) == 0 or len(idx_b) == 0:
        return np.array([])

    n_pairs = min(n_pairs, len(idx_a), len(idx_b))
    i = np.random.choice(idx_a, size=n_pairs, replace=False)
    j = np.random.choice(idx_b, size=n_pairs, replace=False)
    return paired_distances(X[i], X[j])


def get_predictions(model, x_numerical, x_categorical, y, device):
    """Run inference and return predictions and embeddings."""
    output = run_inference(model, x_numerical, x_categorical, device)

    logits = output["logits"].cpu().numpy()
    z = output.get("z")
    if z is not None:
        z = z.cpu().numpy()

    probs = softmax(logits, axis=1)
    y_pred = probs.argmax(axis=1)
    y_true = y.numpy()
    confidences = np.max(probs, axis=1)

    return y_true, y_pred, z, confidences


def fix_predictions(y_pred, mapping):
    """Map predictions back to original class labels."""
    return np.array([mapping[pred] if pred in mapping else pred for pred in y_pred])


def cm_indices(y_true, y_pred):
    """Get indices for all combinations of true and predicted classes (confusion matrix cells)."""
    unique_classes = unique_labels(y_true, y_pred)
    class_indices = {}

    for true_cls in unique_classes:
        class_indices[true_cls] = {}
        mask_true_cls = y_true == true_cls

        for pred_cls in unique_classes:
            idx = np.where(mask_true_cls & (y_pred == pred_cls))[0]
            class_indices[true_cls][pred_cls] = idx

    return class_indices


def analyze_class_failures(X, y_true, y_pred, target_class, max_samples=1000):
    """Analyze failure patterns for a specific class."""
    mask_class = y_true == target_class
    idx_tp = np.where(mask_class & (y_pred == target_class))[0]
    idx_fn = np.where(mask_class & (y_pred != target_class))[0]

    # Subsample if needed
    if len(idx_tp) > max_samples:
        idx_tp = np.random.choice(idx_tp, size=max_samples, replace=False)
    if len(idx_fn) > max_samples:
        idx_fn = np.random.choice(idx_fn, size=max_samples, replace=False)

    # Compute distances
    n_pairs = min(len(idx_tp), len(idx_fn))
    d_tp_tp = sample_paired_distances(X, idx_tp, n_pairs)
    d_fn_fn = sample_paired_distances(X, idx_fn, n_pairs)
    d_fn_tp = sample_cross_distances(X, idx_fn, idx_tp, n_pairs)

    return {
        "target_class": int(target_class),
        "n_tp": len(idx_tp),
        "n_fn": len(idx_fn),
        "tp_tp": compute_distance_stats(d_tp_tp),
        "fn_fn": compute_distance_stats(d_fn_fn),
        "fn_tp": compute_distance_stats(d_fn_tp),
    }


def analyze_classes_failures(X, y_true, y_pred, max_samples=1000):
    """Analyze failure patterns for all classes."""
    unique_classes = np.unique(y_true)
    all_stats = []
    for target_class in unique_classes:
        logger.info(f"Analyzing class {target_class} ...")
        stats = analyze_class_failures(
            X, y_true, y_pred, target_class, max_samples=max_samples
        )
        all_stats.append(stats)
    return all_stats


def visualize_samples(X, z, y_1, y_2, exclude_classes=[], n_samples=3000):
    """Create overall visualizations excluding specific class."""
    if z is None:
        return

    mask = ~np.isin(y_1, exclude_classes)
    vis_mask = create_subsample_mask(y_1[mask], n_samples=n_samples, stratify=False)

    reduced_x = tsne_projection(X[mask][vis_mask])
    reduced_z = tsne_projection(z[mask][vis_mask])
    y_2 = y_2[mask][vis_mask] if y_2 is not None else None

    return samples_plot(reduced_x, y_1[mask][vis_mask], y_2), samples_plot(
        reduced_z, y_1[mask][vis_mask], y_2
    )


def visualize_cm(y_true, y_pred, normalize=None):
    """Visualize confusion matrix."""
    unique_classes = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    cm_figure = confusion_matrix_to_plot(
        cm, title="Confusion Matrix", normalize=normalize
    )
    return cm_figure


def main():
    """Main entry point for inference."""
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )

    df_meta = load_from_json(Path(cfg.path.json_logs) / "metadata" / f"df.json")
    cfg.model.params.num_classes = df_meta["num_classes"]
    cfg.loss.params.class_weight = df_meta["class_weights"]
    device = torch.device(cfg.device)

    num_cols = list(cfg.data.num_cols) if cfg.data.num_cols else []
    cat_cols = list(cfg.data.cat_cols) if cfg.data.cat_cols else []
    label_col = "multi_" + cfg.data.label_col

    train_df, val_df, test_df = load_data(
        Path(cfg.path.processed_data),
        cfg.data.file_name,
        cfg.data.extension,
    )

    pickles_path = Path(cfg.path.pickles)
    json_logs_path = Path(cfg.path.json_logs)

    model = create_model(cfg.model.name, cfg.model.params, device)
    load_latest_checkpoint(Path(cfg.path.models), model, device)

    # Run inference
    for suffix, df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        logger.info(f"Running inference on {suffix} set ...")
        log_dir = Path(cfg.path.tb_logs) / "inference" / suffix
        log_dir.mkdir(parents=True, exist_ok=True)
        tb_logger = TensorboardLogger(log_dir=log_dir)

        x_num, x_cat, y = to_tensors(df, num_cols, cat_cols, label_col)
        y_true, y_pred, z, confidences = get_predictions(model, x_num, x_cat, y, device)

        for cls in np.unique(y_true):
            cls_mask = y_true == cls
            cls_mean_confidence = confidences[cls_mask].mean()
            logger.info(f"Class {cls}: Mean Confidence {cls_mean_confidence:.4f}")

        logger.info(f"Computing confusion matrix.")
        cm_fig = visualize_cm(y_true, y_pred, normalize=None)
        tb_logger.writer.add_figure(
            "confusion_matrix",
            cm_fig,
            global_step=cfg.run_id or 0,
        )
        plt.close(cm_fig)

        logger.info("Computing visualizations ...")
        visual = visualize_samples(
            df[num_cols + cat_cols].to_numpy(), z, y_pred, y_true
        )
        tb_logger.writer.add_figure(
            "raw/classes", visual[0], global_step=cfg.run_id or 0
        )
        plt.close(visual[0])
        tb_logger.writer.add_figure(
            "latent/classes", visual[1], global_step=cfg.run_id or 0
        )
        plt.close(visual[1])

        visual = visualize_samples(
            df[num_cols + cat_cols].to_numpy(),
            z,
            df["cluster"].to_numpy(),
            y_true,
        )
        tb_logger.writer.add_figure(
            "raw/clusters", visual[0], global_step=cfg.run_id or 0
        )
        plt.close(visual[0])
        tb_logger.writer.add_figure(
            "latent/clusters", visual[1], global_step=cfg.run_id or 0
        )
        plt.close(visual[1])
        tb_logger.close()

        logger.info("Analyzing class failures ...")
        classes_failures = analyze_classes_failures(
            df[num_cols + cat_cols].to_numpy(), y_true, y_pred
        )
        classes_failures_df = pd.DataFrame(classes_failures)
        logger.info(f"\n{classes_failures_df}")
        save_to_json(
            classes_failures,
            json_logs_path / f"class_failures/{suffix}.json",
        )

        logger.info("Counting failures per cluster ...")
        cluster_failures = []
        for c_label in df["cluster"].unique():
            cluster_mask = df["cluster"] == c_label
            failures = (y_true[cluster_mask] != y_pred[cluster_mask]).sum()
            mean_confidence = confidences[cluster_mask].mean()
            total_in_cluster = cluster_mask.sum()
            cluster_failures.append(
                {
                    "cluster": int(c_label),
                    "failures": int(failures),
                    "total": int(total_in_cluster),
                    "failure_rate": (
                        float(failures / total_in_cluster)
                        if total_in_cluster > 0
                        else None
                    ),
                    "mean_confidence": (
                        float(mean_confidence) if total_in_cluster > 0 else None
                    ),
                }
            )
        cluster_failures.sort(key=lambda x: x["failure_rate"])

        cluster_failures_df = pd.DataFrame(cluster_failures)
        logger.info(f"\n{cluster_failures_df}")
        save_to_json(
            cluster_failures,
            json_logs_path / f"cluster_failures/{suffix}.json",
        )

        logger.info("Saving per-class predictions ...")
        classes_indices = cm_indices(y_true, y_pred)
        save_to_pickle(
            classes_indices,
            pickles_path / f"class_indices/{suffix}.pkl",
        )


if __name__ == "__main__":
    main()
