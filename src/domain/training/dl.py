import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.engine import Events
from ignite.metrics import Average
from torch.utils.data import DataLoader

from src.domain.training.base import ComponentSpec
from src.engine.dl.builders import (
    create_dataloader,
    create_dataset,
    create_loss,
    create_optimizer,
    create_scheduler,
)
from src.engine.dl.engine import eval_step, train_step
from src.engine.dl.ignite_builder import EngineBuilder
from src.engine.dl.infer import df_to_tensors, run_model
from src.engine.dl.model import DLClassifierFactory
from src.engine.dl.model.checkpoint import load_best_checkpoint

logger = logging.getLogger(__name__)


def _create_model(name: str, params: dict, device: torch.device) -> nn.Module:
    """Instantiate a registered DL classifier on `device`."""
    return DLClassifierFactory.create(name, params).to(device)


def _build_train_engine(model, loss_fn, optimizer, scheduler, device, max_grad_norm):
    """Engine that trains for one epoch and collects per-step loss into history."""
    builder = (
        EngineBuilder(train_step)
        .with_state(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            max_grad_norm=max_grad_norm,
        )
        .with_metric("loss", Average(output_transform=lambda x: x["loss"]))
        .with_history(output_transform=lambda x: {"loss": x["loss"]})
    )
    return builder.build(), builder.history


def _build_validation_engine(
    model, loss_fn, device, trainer, patience, min_delta, models_path
):
    """Validator engine with early stopping + best-loss checkpointing."""
    return (
        EngineBuilder(eval_step)
        .with_state(model=model, loss_fn=loss_fn, device=device)
        .with_metric("loss", Average(output_transform=lambda x: x["loss"]))
        .with_early_stopping(
            trainer=trainer, metric="loss", patience=patience, min_delta=min_delta
        )
        .with_checkpointing(
            trainer=trainer,
            checkpoint_dir=models_path,
            objects_to_save={"model": model},
            metric="loss",
        )
        .build()
    )


@dataclass
class DLTrainer:
    """Fits PyTorch classifiers through Ignite engines over a tabular dataset."""

    device: torch.device
    num_cols: list[str]
    cat_cols: list[str]
    label_col: str
    class_weights: list[float] | None
    loss: ComponentSpec
    optimizer: ComponentSpec
    scheduler: ComponentSpec
    epochs: int
    max_grad_norm: float
    patience: int
    min_delta: float
    train_loader_params: dict
    val_loader_params: dict

    def features(self, df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
        """The whole frame: the dataset selects its own feature columns."""
        return df

    def prepare(
        self, df: pd.DataFrame, feat_cols: list[str], label_col: str
    ) -> tuple[pd.DataFrame, None]:
        """The whole frame: the dataset selects its own feature and label columns."""
        return df, None

    def _loader(self, df: pd.DataFrame, params: dict) -> DataLoader:
        """Wrap a split in a DataLoader over the tabular dataset."""
        return create_dataloader(
            create_dataset(
                df, self.num_cols, self.cat_cols, label_col=[self.label_col]
            ),
            params,
        )

    def fit(
        self,
        name: str,
        params: dict,
        X: pd.DataFrame,
        y: object = None,
        *,
        X_val: pd.DataFrame,
        save_dir: Path,
    ) -> tuple[nn.Module, dict]:
        """Train with early stopping and return the best checkpoint plus its loss history."""
        models_path = Path(save_dir)

        loss_params = dict(self.loss.params)
        loss_params.setdefault("class_weight", self.class_weights)

        model = _create_model(name, params, self.device)
        loss_fn = create_loss(self.loss.name, loss_params, self.device)

        train_loader = self._loader(X, self.train_loader_params)
        val_loader = self._loader(X_val, self.val_loader_params)

        optimizer = create_optimizer(
            self.optimizer.name, self.optimizer.params, model, loss_fn=loss_fn
        )
        scheduler = create_scheduler(
            self.scheduler.name, self.scheduler.params, optimizer, train_loader
        )

        trainer, history = _build_train_engine(
            model, loss_fn, optimizer, scheduler, self.device, self.max_grad_norm
        )
        validator = _build_validation_engine(
            model,
            loss_fn,
            self.device,
            trainer,
            self.patience,
            self.min_delta,
            models_path,
        )

        @trainer.on(Events.EPOCH_COMPLETED)
        def _run_validation(engine) -> None:
            logger.info(
                "Epoch [%d] Train Loss: %.6f",
                engine.state.epoch,
                engine.state.metrics["loss"],
            )
            validator.run(val_loader)
            logger.info(
                "Epoch [%d] Val Loss: %.6f",
                engine.state.epoch,
                validator.state.metrics["loss"],
            )

        trainer.run(train_loader, max_epochs=self.epochs)

        load_best_checkpoint(models_path, model, self.device)
        logger.info("Best checkpoint reloaded after training.")

        return model, {"history": history}

    def grid_search(
        self,
        name: str,
        params: dict,
        grid: dict,
        X: pd.DataFrame,
        y: object = None,
        *,
        scoring: str = "f1_macro",
        cv: int = 5,
        max_samples: int | None = None,
        random_state: int = 42,
    ) -> tuple[nn.Module, dict]:
        """Not available: grid search is implemented for ML classifiers only."""
        raise NotImplementedError(
            "Grid search is implemented for ML classifiers only; "
            f"{name!r} is a DL classifier."
        )

    def predict(
        self, model: nn.Module, X: pd.DataFrame, *, return_embedding: bool = False
    ) -> tuple:
        """Predict a DataFrame → (y_pred, y_proba), plus the latent embedding on request."""
        inputs = df_to_tensors(
            X,
            [self.num_cols, self.cat_cols],
            dtypes=[torch.float32, torch.long],
        )
        output = run_model(model, inputs, self.device)
        probs = F.softmax(output["logits"].cpu(), dim=1)
        y_pred = probs.argmax(dim=1).numpy()
        y_proba = probs.numpy()
        if return_embedding:
            z = output["z"].cpu().numpy() if "z" in output else None
            return y_pred, y_proba, z
        return y_pred, y_proba

    def save(
        self,
        model: nn.Module,
        path: Path,
        *,
        name: str = "",
        params: dict | None = None,
    ) -> None:
        """Save the state dict and its metadata to `path / model.pt`."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": model.state_dict(), "name": name, "params": params or {}},
            path / "model.pt",
        )

    def load(self, path: Path) -> nn.Module:
        """Load the model from `path / model.pt` onto this trainer's device."""
        ckpt = torch.load(
            Path(path) / "model.pt", map_location="cpu", weights_only=True
        )
        model = _create_model(ckpt["name"], ckpt["params"], self.device)
        model.load_state_dict(ckpt["state_dict"])
        return model

    def has_model(self, path: Path) -> bool:
        """True when `path` holds a saved state dict."""
        return (Path(path) / "model.pt").exists()
