from typing import Callable, Optional, Union

import torch
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce


class Precision(Metric):
    """
    Calculates Precision for binary or multiclass classification.

    Args:
        output_transform: A callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. Default is lambda x: x.
        pred_transform: A callable that converts raw predictions to class predictions.
            Default converts logits to classes (argmax for multi-class, threshold at 0.5 for binary).
            Custom function should take a tensor and return a tensor of predicted class indices.
        average: The averaging method. Options: 'micro', 'macro', 'weighted', or None.
            - 'micro': Calculate metrics globally by counting total TP, FP.
            - 'macro': Calculate metrics for each label and find unweighted mean.
            - 'weighted': Calculate metrics for each label and find average weighted by support.
            - None: Return precision for each class.
        device: Specifies which device updates are accumulated on. Default is cpu.
    """

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        pred_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        average: Optional[str] = "macro",
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        self._pred_transform = pred_transform
        self._average = average
        if average not in ["micro", "macro", "weighted", None]:
            raise ValueError(
                f"Average must be 'micro', 'macro', 'weighted', or None, got {average}"
            )

        self._true_positives = None
        self._false_positives = None
        self._support = None
        self._num_classes = None

        super(Precision, self).__init__(
            output_transform=output_transform, device=device
        )

    @reinit__is_reduced
    def reset(self) -> None:
        self._true_positives = None
        self._false_positives = None
        self._support = None
        self._num_classes = None
        super(Precision, self).reset()

    @reinit__is_reduced
    def update(self, output: tuple) -> None:
        y_pred, y_true = output[0].detach(), output[1].detach()

        # Apply custom prediction transform or use default
        if self._pred_transform is not None:
            y_pred = self._pred_transform(y_pred)
        else:
            # Default: Handle predictions (convert logits to class predictions)
            if y_pred.dim() > 1 and y_pred.size(1) > 1:
                # Multi-class: get argmax
                y_pred = torch.argmax(y_pred, dim=1)
            else:
                # Binary: threshold at 0.5
                y_pred = (y_pred > 0.5).long().squeeze()

        y_true = y_true.long().squeeze()

        # Initialize on first update
        if self._num_classes is None:
            self._num_classes = max(y_pred.max().item(), y_true.max().item()) + 1
            self._true_positives = torch.zeros(
                self._num_classes, dtype=torch.long, device=self._device
            )
            self._false_positives = torch.zeros(
                self._num_classes, dtype=torch.long, device=self._device
            )
            self._support = torch.zeros(
                self._num_classes, dtype=torch.long, device=self._device
            )

        # Update confusion matrix components for each class
        for c in range(self._num_classes):
            pred_c = y_pred == c
            true_c = y_true == c

            self._true_positives[c] += (pred_c & true_c).sum()
            self._false_positives[c] += (pred_c & ~true_c).sum()
            self._support[c] += true_c.sum()

    @sync_all_reduce("_true_positives", "_false_positives", "_support")
    def compute(self) -> Union[torch.Tensor, float]:
        if self._true_positives is None:
            raise RuntimeError(
                "PrecisionScore must have at least one update before compute."
            )

        # Calculate precision for each class
        tp = self._true_positives.float()
        fp = self._false_positives.float()

        # Avoid division by zero
        precision = tp / (tp + fp + 1e-15)

        if self._average == "micro":
            # Micro-average: aggregate TP, FP globally
            total_tp = tp.sum()
            total_fp = fp.sum()
            return (total_tp / (total_tp + total_fp + 1e-15)).item()
        elif self._average == "macro":
            # Macro-average: unweighted mean
            return precision.mean().item()
        elif self._average == "weighted":
            # Weighted average by support
            weights = self._support.float() / (self._support.sum().float() + 1e-15)
            return (precision * weights).sum().item()
        else:
            # Return per-class precision
            return precision


class Recall(Metric):
    """
    Calculates Recall for binary or multiclass classification.

    Args:
        output_transform: A callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. Default is lambda x: x.
        pred_transform: A callable that converts raw predictions to class predictions.
            Default converts logits to classes (argmax for multi-class, threshold at 0.5 for binary).
            Custom function should take a tensor and return a tensor of predicted class indices.
        average: The averaging method. Options: 'micro', 'macro', 'weighted', or None.
            - 'micro': Calculate metrics globally by counting total TP, FN.
            - 'macro': Calculate metrics for each label and find unweighted mean.
            - 'weighted': Calculate metrics for each label and find average weighted by support.
            - None: Return recall for each class.
        device: Specifies which device updates are accumulated on. Default is cpu.
    """

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        pred_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        average: Optional[str] = "macro",
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        self._pred_transform = pred_transform
        self._average = average
        if average not in ["micro", "macro", "weighted", None]:
            raise ValueError(
                f"Average must be 'micro', 'macro', 'weighted', or None, got {average}"
            )

        self._true_positives = None
        self._false_negatives = None
        self._support = None
        self._num_classes = None

        super(Recall, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self._true_positives = None
        self._false_negatives = None
        self._support = None
        self._num_classes = None
        super(Recall, self).reset()

    @reinit__is_reduced
    def update(self, output: tuple) -> None:
        y_pred, y_true = output[0].detach(), output[1].detach()

        # Apply custom prediction transform or use default
        if self._pred_transform is not None:
            y_pred = self._pred_transform(y_pred)
        else:
            # Default: Handle predictions (convert logits to class predictions)
            if y_pred.dim() > 1 and y_pred.size(1) > 1:
                # Multi-class: get argmax
                y_pred = torch.argmax(y_pred, dim=1)
            else:
                # Binary: threshold at 0.5
                y_pred = (y_pred > 0.5).long().squeeze()

        y_true = y_true.long().squeeze()

        # Initialize on first update
        if self._num_classes is None:
            self._num_classes = max(y_pred.max().item(), y_true.max().item()) + 1
            self._true_positives = torch.zeros(
                self._num_classes, dtype=torch.long, device=self._device
            )
            self._false_negatives = torch.zeros(
                self._num_classes, dtype=torch.long, device=self._device
            )
            self._support = torch.zeros(
                self._num_classes, dtype=torch.long, device=self._device
            )

        # Update confusion matrix components for each class
        for c in range(self._num_classes):
            pred_c = y_pred == c
            true_c = y_true == c

            self._true_positives[c] += (pred_c & true_c).sum()
            self._false_negatives[c] += (~pred_c & true_c).sum()
            self._support[c] += true_c.sum()

    @sync_all_reduce("_true_positives", "_false_negatives", "_support")
    def compute(self) -> Union[torch.Tensor, float]:
        if self._true_positives is None:
            raise RuntimeError(
                "RecallScore must have at least one update before compute."
            )

        # Calculate recall for each class
        tp = self._true_positives.float()
        fn = self._false_negatives.float()

        # Avoid division by zero
        recall = tp / (tp + fn + 1e-15)

        if self._average == "micro":
            # Micro-average: aggregate TP, FN globally
            total_tp = tp.sum()
            total_fn = fn.sum()
            return (total_tp / (total_tp + total_fn + 1e-15)).item()
        elif self._average == "macro":
            # Macro-average: unweighted mean
            return recall.mean().item()
        elif self._average == "weighted":
            # Weighted average by support
            weights = self._support.float() / (self._support.sum().float() + 1e-15)
            return (recall * weights).sum().item()
        else:
            # Return per-class recall
            return recall


class F1(Metric):
    """
    Calculates F1 Score for binary or multiclass classification.

    Args:
        output_transform: A callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. Default is lambda x: x.
        pred_transform: A callable that converts raw predictions to class predictions.
            Default converts logits to classes (argmax for multi-class, threshold at 0.5 for binary).
            Custom function should take a tensor and return a tensor of predicted class indices.
        average: The averaging method. Options: 'micro', 'macro', 'weighted', or None.
            - 'micro': Calculate metrics globally by counting total TP, FP, FN.
            - 'macro': Calculate metrics for each label and find unweighted mean.
            - 'weighted': Calculate metrics for each label and find average weighted by support.
            - None: Return F1 score for each class.
        device: Specifies which device updates are accumulated on. Default is cpu.
    """

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        pred_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        average: Optional[str] = "macro",
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        self._pred_transform = pred_transform
        self._average = average
        if average not in ["micro", "macro", "weighted", None]:
            raise ValueError(
                f"Average must be 'micro', 'macro', 'weighted', or None, got {average}"
            )

        self._true_positives = None
        self._false_positives = None
        self._false_negatives = None
        self._support = None
        self._num_classes = None

        super(F1, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self._true_positives = None
        self._false_positives = None
        self._false_negatives = None
        self._support = None
        self._num_classes = None
        super(F1, self).reset()

    @reinit__is_reduced
    def update(self, output: tuple) -> None:
        y_pred, y_true = output[0].detach(), output[1].detach()

        # Apply custom prediction transform or use default
        if self._pred_transform is not None:
            y_pred = self._pred_transform(y_pred)
        else:
            # Default: Handle predictions (convert logits to class predictions)
            if y_pred.dim() > 1 and y_pred.size(1) > 1:
                # Multi-class: get argmax
                y_pred = torch.argmax(y_pred, dim=1)
            else:
                # Binary: threshold at 0.5
                y_pred = (y_pred > 0.5).long().squeeze()

        y_true = y_true.long().squeeze()

        # Initialize on first update
        if self._num_classes is None:
            self._num_classes = max(y_pred.max().item(), y_true.max().item()) + 1
            self._true_positives = torch.zeros(
                self._num_classes, dtype=torch.long, device=self._device
            )
            self._false_positives = torch.zeros(
                self._num_classes, dtype=torch.long, device=self._device
            )
            self._false_negatives = torch.zeros(
                self._num_classes, dtype=torch.long, device=self._device
            )
            self._support = torch.zeros(
                self._num_classes, dtype=torch.long, device=self._device
            )

        # Update confusion matrix components for each class
        for c in range(self._num_classes):
            pred_c = y_pred == c
            true_c = y_true == c

            self._true_positives[c] += (pred_c & true_c).sum()
            self._false_positives[c] += (pred_c & ~true_c).sum()
            self._false_negatives[c] += (~pred_c & true_c).sum()
            self._support[c] += true_c.sum()

    @sync_all_reduce(
        "_true_positives", "_false_positives", "_false_negatives", "_support"
    )
    def compute(self) -> Union[torch.Tensor, float]:
        if self._true_positives is None:
            raise RuntimeError("F1Score must have at least one update before compute.")

        # Calculate precision and recall for each class
        tp = self._true_positives.float()
        fp = self._false_positives.float()
        fn = self._false_negatives.float()

        # Avoid division by zero
        precision = tp / (tp + fp + 1e-15)
        recall = tp / (tp + fn + 1e-15)

        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall + 1e-15)

        if self._average == "micro":
            # Micro-average: aggregate TP, FP, FN globally
            total_tp = tp.sum()
            total_fp = fp.sum()
            total_fn = fn.sum()
            micro_precision = total_tp / (total_tp + total_fp + 1e-15)
            micro_recall = total_tp / (total_tp + total_fn + 1e-15)
            return (
                2
                * micro_precision
                * micro_recall
                / (micro_precision + micro_recall + 1e-15)
            ).item()
        elif self._average == "macro":
            # Macro-average: unweighted mean
            return f1.mean().item()
        elif self._average == "weighted":
            # Weighted average by support
            weights = self._support.float() / (self._support.sum().float() + 1e-15)
            return (f1 * weights).sum().item()
        else:
            # Return per-class F1 scores
            return f1
