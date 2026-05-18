import sys
from pathlib import Path
import numpy as np

from src.common.config import load_config
from src.common.logging import setup_logger
from src.common.utils import save_to_json, save_to_joblib, load_from_joblib, load_from_json
from src.data.io import load_listed_dfs
from src.data.preprocessing import subsample_df
from src.ml.classification import train_sklearn_classifier, evaluate_sklearn_classifier
from ignite.handlers.tensorboard_logger import TensorboardLogger
from src.plot.array import confusion_matrix_to_plot
from src.plot.dict import dict_to_bar_plot

SCALAR_METRICS = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
]

PER_CLASS_METRICS = ["precision_per_class", "recall_per_class", "f1_per_class"]


def load_data(base_path, extension, label_col, n_samples, random_state):
    """Load train/val/test splits and optionally subsample the training set."""
    train_df, val_df, test_df = load_listed_dfs(
        Path(base_path),
        [f"train.{extension}", f"val.{extension}", f"test.{extension}"],
    )
    if n_samples is not None:
        train_df = subsample_df(train_df, n_samples, random_state, label_col)
    return train_df, val_df, test_df

def train(processed_data_path, 
          extension,  
          label_col,
          num_cols, 
          n_samples, 
          random_state,
          classifier_name,
          classifier_params,
          json_logs_path,
          joblib_path):
    """Train the supervised classifier."""
    train_df, val_df, _ = load_data(base_path=processed_data_path, 
                                          extension=extension, 
                                          label_col=label_col, 
                                          n_samples=n_samples,
                                          random_state=random_state)

    model = train_sklearn_classifier(X=train_df[num_cols], 
                                     y=train_df[label_col], 
                                     classifier_name=classifier_name, 
                                     classifier_params=classifier_params)
    save_to_joblib(obj=model, path=joblib_path/f"{classifier_name}.joblib")

    train_metrics = evaluate_sklearn_classifier(model, train_df[num_cols], train_df[label_col])
    val_metrics = evaluate_sklearn_classifier(model, val_df[num_cols], val_df[label_col])

    save_to_json(data={k:v.tolist() if isinstance(v, np.ndarray) else v for k,v in train_metrics.items()},
                  file_path=json_logs_path/f"{classifier_name}_training_metrics.json")
    save_to_json(data={k:v.tolist() if isinstance(v, np.ndarray) else v for k,v in val_metrics.items()},
                  file_path=json_logs_path/f"{classifier_name}_validation_metrics.json")

def test(processed_data_path,
         extension,
         num_cols,
         label_col,
         n_samples,
         random_state,
         classifier_name,
         json_logs_path,
         tb_logs_path,
         joblib_path,
         run_id):
    """Test the trained classifier."""

    _, _, test_df = load_data(base_path=processed_data_path, 
                                extension=extension, 
                                label_col=label_col, 
                                n_samples=n_samples,
                                random_state=random_state)
    
    model = load_from_joblib(path=joblib_path/f"{classifier_name}.joblib")
    test_metrics = evaluate_sklearn_classifier(model, test_df[num_cols], test_df[label_col])

    save_to_json(data={k:v.tolist() if isinstance(v, np.ndarray) else v for k,v in test_metrics.items()},
                  file_path=json_logs_path/f"{classifier_name}_test_metrics.json")

     # --- TensorBoard ---

    log_dir = tb_logs_path / "testing"
    log_dir.mkdir(parents=True, exist_ok=True)
    tb_logger = TensorboardLogger(log_dir=log_dir)
       
    writer = tb_logger.writer
    for name in SCALAR_METRICS:
        if name in test_metrics:
            writer.add_scalar(f"test/{name}", test_metrics[name], run_id)

    if "confusion_matrix" in test_metrics:
        writer.add_figure(
            "test/confusion_matrix",
            confusion_matrix_to_plot(test_metrics["confusion_matrix"]),
            run_id,
        )
    if "f1_per_class" in test_metrics:
        f1_dict = {
            f"class_{i}": float(v)
            for i, v in enumerate(test_metrics["f1_per_class"])
        }
        writer.add_figure("test/f1_per_class", dict_to_bar_plot(f1_dict), run_id)


def ml_classify(cfg):
    """Run supervised classification pipeline (train and/or test)."""
    json_logs_path = Path(cfg.path.json_logs)
    joblib_path = Path(cfg.path.joblib)
    num_cols = list(cfg.data.num_cols) if cfg.data.num_cols else []
    label_col = "encoded_" + cfg.data.label_col

    processed_data_path = Path(cfg.path.processed_data)
    tb_logs_path = Path(cfg.path.tb_logs)

    common = dict(
        processed_data_path=processed_data_path,
        extension=cfg.data.extension,
        num_cols=num_cols,
        label_col=label_col,
        n_samples=cfg.n_samples,
        random_state=cfg.seed,
        classifier_name=cfg.model.name,
        json_logs_path=json_logs_path,
        joblib_path=joblib_path
    )

    stage = cfg.get("stage", "all")
    model = None

    if stage in ("all", "training"):
        model = train(**common,
                      classifier_params=cfg.model.params,
                )
    
    if stage in ("all", "testing"):
        model = test(**common,
                     tb_logs_path=tb_logs_path,
                     run_id=cfg.run_id
                )
    if stage not in ("all", "training", "testing"):
        #logger.error("Unknown stage: %r. Valid: 'all', 'training', 'testing'.", stage)
        sys.exit(1)

    #logger.info("All stages completed.")
    return model

def main():
    """Main entry point for supervised classification."""
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="ml_config",
        overrides=sys.argv[1:],
    )
    ml_classify(cfg)

if __name__ == "__main__":
    main()
