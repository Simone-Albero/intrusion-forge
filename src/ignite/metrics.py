from typing import Callable

import torch
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce

_EPS = 1e-10
_VALID_AVERAGES = ("micro", "macro", "weighted", None)


class _ClassificationMetric(Metric):
    """Shared base for per-class classification metrics (Precision, Recall, F1)."""

    def __init__(
        self,
        num_classes: int,
        output_transform: Callable = lambda x: x,
        pred_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        average: str | None = "macro",
        device: str | torch.device = torch.device("cpu"),
    ):
        if average not in _VALID_AVERAGES:
            raise ValueError(
                f"average must be one of {_VALID_AVERAGES}, got {average!r}."
            )
        self._num_classes = num_classes
        self._pred_transform = pred_transform
        self._average = average
        self._tp = self._fp = self._fn = self._support = None
        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self._tp = self._fp = self._fn = self._support = None
        super().reset()

    def _to_preds(self, y_pred: torch.Tensor) -> torch.Tensor:
        if self._pred_transform is not None:
            return self._pred_transform(y_pred)
        if y_pred.dim() > 1 and y_pred.size(1) > 1:
            return torch.argmax(y_pred, dim=1)
        return (y_pred > 0.5).long().squeeze()

    def _init_buffers(self) -> None:
        zeros = lambda: torch.zeros(
            self._num_classes, dtype=torch.long, device=self._device
        )
        self._tp = self._fp = self._fn = self._support = None  # reset first
        self._tp, self._fp, self._fn, self._support = zeros(), zeros(), zeros(), zeros()

    @reinit__is_reduced
    def update(self, output: tuple) -> None:
        y_pred, y_true = output[0].detach(), output[1].detach()
        if y_pred.numel() == 0 or y_true.numel() == 0:
            return

        y_pred = self._to_preds(y_pred)
        y_true = y_true.long().squeeze()

        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}."
            )

        if self._tp is None:
            self._init_buffers()

        for c in range(self._num_classes):
            pred_c, true_c = y_pred == c, y_true == c
            self._tp[c] += (pred_c & true_c).sum()
            self._fp[c] += (pred_c & ~true_c).sum()
            self._fn[c] += (~pred_c & true_c).sum()
            self._support[c] += true_c.sum()

    def _aggregate(self, per_class: torch.Tensor) -> torch.Tensor | float:
        if self._average == "macro":
            valid = self._support > 0
            return per_class[valid].mean().item() if valid.any() else 0.0
        if self._average == "weighted":
            weights = self._support.float() / (self._support.sum().float() + _EPS)
            return (per_class * weights).sum().item()
        return per_class  # None → per-class tensor

    def _check_initialized(self, name: str) -> None:
        if self._tp is None:
            raise RuntimeError(f"{name} must have at least one update before compute.")


class Precision(_ClassificationMetric):
    """Precision for binary or multiclass classification.

    average: 'micro' | 'macro' | 'weighted' | None
    """

    @sync_all_reduce("_tp", "_fp", "_support")
    def compute(self) -> torch.Tensor | float:
        self._check_initialized("Precision")
        tp, fp = self._tp.float(), self._fp.float()
        if self._average == "micro":
            return (tp.sum() / (tp.sum() + fp.sum() + _EPS)).item()
        return self._aggregate(tp / (tp + fp + _EPS))


class Recall(_ClassificationMetric):
    """Recall for binary or multiclass classification.

    average: 'micro' | 'macro' | 'weighted' | None
    """

    @sync_all_reduce("_tp", "_fn", "_support")
    def compute(self) -> torch.Tensor | float:
        self._check_initialized("Recall")
        tp, fn = self._tp.float(), self._fn.float()
        if self._average == "micro":
            return (tp.sum() / (tp.sum() + fn.sum() + _EPS)).item()
        return self._aggregate(tp / (tp + fn + _EPS))


class F1(_ClassificationMetric):
    """F1 score for binary or multiclass classification.

    average: 'micro' | 'macro' | 'weighted' | None
    """

    @sync_all_reduce("_tp", "_fp", "_fn", "_support")
    def compute(self) -> torch.Tensor | float:
        self._check_initialized("F1")
        tp, fp, fn = self._tp.float(), self._fp.float(), self._fn.float()
        if self._average == "micro":
            tp_s, fp_s, fn_s = tp.sum(), fp.sum(), fn.sum()
            p = tp_s / (tp_s + fp_s + _EPS)
            r = tp_s / (tp_s + fn_s + _EPS)
            return (2 * p * r / (p + r + _EPS)).item()
        p = tp / (tp + fp + _EPS)
        r = tp / (tp + fn + _EPS)
        return self._aggregate(2 * p * r / (p + r + _EPS))
