from pathlib import Path
import numpy as np
import sys
import matplotlib.pyplot as plt 
from src.data.io import load_listed_dfs
from src.common.config import load_config
from src.common.utils import load_from_joblib, save_to_pickle, save_to_json
from src.data.analyze import evaluate_predictions
from sklearn.metrics import confusion_matrix
from ignite.handlers.tensorboard_logger import TensorboardLogger
from src.plot.array import confusion_matrix_to_plot, samples_plot
from src.ml.projection import create_subsample_mask, tsne_projection


def visualize_cm(y_true, y_pred, normalize=None):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true), normalize=normalize)
    return confusion_matrix_to_plot(cm), cm

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


def infer(cfg):
    """Run inference on all splits and collect prediction diagnostics."""

    json_logs_path = Path(cfg.path.json_logs)
    num_cols = list(cfg.data.num_cols) if cfg.data.num_cols else []
    cat_cols = list(cfg.data.cat_cols) if cfg.data.cat_cols else []
    label_col = "encoded_" + cfg.data.label_col
    feat_cols = num_cols + cat_cols
    splits = load_listed_dfs(
        Path(cfg.path.processed_data),
        [f"{s}.{cfg.data.extension}" for s in ("train", "val", "test")],
    )

    model = load_from_joblib(Path(cfg.path.joblib)/f"{cfg.model.name}.joblib")
    stats = {"class_confidence": {}, "pred_infos": {}, "cluster_failures": {}}
    for suffix, df in zip(("train", "val", "test"), splits):
    #    logger.info("Running inference on %s set ...", suffix)
        log_dir = Path(cfg.path.tb_logs) / "analysis" / suffix
        log_dir.mkdir(parents=True, exist_ok=True)
        tb_logger = TensorboardLogger(log_dir=log_dir)
        step = cfg.run_id or 0

        y_true = df[label_col]
        X = df[feat_cols]
        y_pred = model.predict(X)
        confidences = model.predict_proba(X)

        cm_fig, cm = visualize_cm(y_true, y_pred, normalize=None)
        tb_logger.writer.add_figure("confusion_matrices/original", cm_fig, step)
        plt.close(cm_fig)

        save_to_pickle(
            cm, Path(cfg.path.pickle) / f"analysis/confusion_matrices/{suffix}.pkl"
        )

        cm_fig, cm = visualize_cm(y_true, y_pred, normalize="true")
        tb_logger.writer.add_figure(
            "confusion_matrices/normalized_by_row", cm_fig, step
        )
        plt.close(cm_fig)

        for tag, data in (("raw/classes", X),):
            if data is None:
                continue
            correct = (y_pred == y_true).astype(int)

            fig = visualize_samples(data, y_true, correct, n_components=2)
            tb_logger.writer.add_figure(tag + "_2D", fig, step)
            plt.close(fig)

            fig = visualize_samples(data, y_true, correct, n_components=3)
            tb_logger.writer.add_figure(tag + "_3D", fig, step)
            plt.close(fig)

        pred_infos = evaluate_predictions(df, y_true, y_pred, confidences)
        save_to_json(
            pred_infos,
            json_logs_path / f"analysis/predictions/{suffix}.json",
        )
        stats["pred_infos"][suffix] = pred_infos
    
    return stats

def main():
    """Main entry point for inference."""
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="ml_config",
        overrides=sys.argv[1:],
    )
    infer(cfg)


if __name__ == "__main__":
    main()