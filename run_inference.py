import shutil
import logging
import sys
from pathlib import Path

import numpy as np
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
from src.plot.array import vectors_plot, confusion_matrix_to_plot

setup_logger()
logger = logging.getLogger(__name__)


def load_data(
    processed_data_path,
    file_name,
    extension,
    num_cols,
    cat_cols,
    label_col,
):
    """Load and convert data to PyTorch tensors."""
    train_df, val_df, test_df = load_data_splits(
        processed_data_path, file_name, extension
    )

    def to_tensors(df):
        return (
            torch.tensor(df[num_cols].to_numpy(dtype=np.float32)),
            torch.tensor(df[cat_cols].to_numpy(dtype=np.int64)),
            torch.tensor(df[label_col].to_numpy(dtype=np.int64)),
        )

    return to_tensors(train_df), to_tensors(val_df), to_tensors(test_df)


def run_inference(model, x_numerical, x_categorical, device):
    """Run model inference."""
    model.eval()
    with torch.no_grad():
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

    y_pred = logits.argmax(axis=1)
    y_true = y.numpy()

    return y_true, y_pred, z, x_numerical.cpu().numpy()


def fix_predictions(y_pred, mapping):
    """Map predictions back to original class labels."""
    return np.array([mapping[pred] if pred in mapping else pred for pred in y_pred])


def per_classes_predictions(y_true, y_pred):
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


def visualize_class(X, z, y_true, y_pred, label, max_samples=3000):
    """Create visualizations for a specific class."""
    if z is None:
        return

    class_mask = y_true == label
    if class_mask.sum() > max_samples:
        class_mask_indices = np.where(class_mask)[0]
        selected_indices = np.random.choice(
            class_mask_indices, size=max_samples, replace=False
        )
        new_mask = np.zeros_like(class_mask, dtype=bool)
        new_mask[selected_indices] = True
        class_mask = new_mask

    is_correct = (y_true[class_mask] == y_pred[class_mask]).astype(int)

    reduced_x = tsne_projection(X[class_mask])
    reduced_z = tsne_projection(z[class_mask])

    return vectors_plot(reduced_x, is_correct), vectors_plot(reduced_z, is_correct)


def visualize_classes(X, z, y_true, y_pred, labels):
    """Generate visualizations for all classes one at a time."""
    if z is None:
        return

    for label in labels:
        logger.info(f"Visualizing class {label} ...")
        yield label, visualize_class(X, z, y_true, y_pred, label)


def visualize_overall(X, z, y_true, exclude_classes=[], n_samples=3000):
    """Create overall visualizations excluding specific class."""
    if z is None:
        return

    mask = ~np.isin(y_true, exclude_classes)
    vis_mask = create_subsample_mask(y_true[mask], n_samples=n_samples, stratify=True)

    reduced_x = tsne_projection(X[mask][vis_mask])
    reduced_z = tsne_projection(z[mask][vis_mask])

    return vectors_plot(reduced_x, y_true[mask][vis_mask]), vectors_plot(
        reduced_z, y_true[mask][vis_mask]
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

    df_meta = load_from_json(Path(cfg.path.json_logs) / "df_metadata.json")
    cfg.model.params.num_classes = df_meta["num_classes"]
    cfg.loss.params.class_weight = df_meta["class_weights"]
    device = torch.device(cfg.device)

    num_cols = list(cfg.data.num_cols)
    cat_cols = list(cfg.data.cat_cols)
    label_col = "multi_" + cfg.data.label_col

    (
        (x_num_train, x_cat_train, y_train),
        (x_num_val, x_cat_val, y_val),
        (x_num_test, x_cat_test, y_test),
    ) = load_data(
        Path(cfg.path.processed_data),
        cfg.data.file_name,
        cfg.data.extension,
        num_cols,
        cat_cols,
        label_col,
    )

    pickles_path = Path(cfg.path.pickles)
    json_logs_path = Path(cfg.path.json_logs)

    model = create_model(cfg.model.name, cfg.model.params, device)
    load_latest_checkpoint(Path(cfg.path.models), model, device)

    # Run inference
    for suffix, (x_num, x_cat, y) in [
        ("train", (x_num_train, x_cat_train, y_train)),
        ("val", (x_num_val, x_cat_val, y_val)),
        ("test", (x_num_test, x_cat_test, y_test)),
    ]:
        log_dir = Path(cfg.path.tb_logs) / "inference" / suffix
        # if log_dir.exists():
        #     shutil.rmtree(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        tb_logger = TensorboardLogger(log_dir=log_dir)

        y_true, y_pred, z, X = get_predictions(model, x_num, x_cat, y, device)
        # y_pred = fix_predictions(y_pred, {3: 3, 4: 3, 5: 3})
        unique_classes = np.unique(y_true)

        cm_fig = visualize_cm(y_true, y_pred, normalize="true")
        tb_logger.writer.add_figure(
            "confusion_matrix",
            cm_fig,
            global_step=cfg.run_id or 0,
        )
        plt.close(cm_fig)

        logger.info("Computing class visualizations ...")
        for label, (fig_x, fig_z) in visualize_classes(
            X, z, y_true, y_pred, unique_classes
        ):
            tb_logger.writer.add_figure(
                f"raw/{label}", fig_x, global_step=cfg.run_id or 0
            )
            plt.close(fig_x)
            tb_logger.writer.add_figure(
                f"latent/{label}", fig_z, global_step=cfg.run_id or 0
            )
            plt.close(fig_z)

        logger.info("Computing overall visualizations ...")
        overall_visual = visualize_overall(X, z, y_true)
        tb_logger.writer.add_figure(
            "raw/overall", overall_visual[0], global_step=cfg.run_id or 0
        )
        plt.close(overall_visual[0])
        tb_logger.writer.add_figure(
            "latent/overall", overall_visual[1], global_step=cfg.run_id or 0
        )
        plt.close(overall_visual[1])
        tb_logger.close()

        logger.info("Analyzing class failures ...")
        classes_failures = analyze_classes_failures(X, y_true, y_pred)
        save_to_json(
            classes_failures,
            json_logs_path
            / f"class_failures/{suffix}{'_' + str(cfg.run_id) if cfg.run_id else ''}.json",
        )

        logger.info("Saving per-class predictions ...")
        classes_indices = per_classes_predictions(y_true, y_pred)
        save_to_pickle(
            classes_indices,
            pickles_path
            / f"class_indices/{suffix}{'_' + str(cfg.run_id) if cfg.run_id else ''}.pkl",
        )


if __name__ == "__main__":
    main()
