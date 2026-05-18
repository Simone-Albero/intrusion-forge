import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from .projection import stratified_subsample, tsne_projection
from ..plot.base import Plot
from ..plot.charts import bar_plot, scatter_plot
from ..plot.metrics import confusion_matrix_plot
from ..plot.style import extended_palette


def build_test_figures(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_mapping: dict,
    z: np.ndarray | None = None,
    n_samples: int = 2000,
) -> dict[str, Plot]:
    """Confusion matrix + per-class F1 bar + t-SNE scatter on raw (and latent).

    Returns a dict pre-keyed under "figure/testing/..." so it can be splattered
    into a LogBundle. `z` (latent embedding) is optional — only DL pipelines
    provide it.
    """
    figures: dict[str, Plot] = {}

    classes = np.unique(y_true)
    class_names = [label_mapping.get(str(int(c)), str(c)) for c in classes]
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")
    figures["figure/testing/confusion_matrix"] = confusion_matrix_plot(
        cm, class_names=class_names, normalize=None
    )

    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    f1_dict = {
        label_mapping.get(str(int(c)), str(c)): float(v)
        for c, v in zip(classes, f1_per_class)
    }
    figures["figure/testing/f1_per_class"] = bar_plot(
        list(f1_dict.keys()),
        list(f1_dict.values()),
        orientation="v",
        color=extended_palette(len(f1_dict)),
        sort=None,
        ylim=(0, 1),
    )

    names = {int(c): label_mapping.get(str(int(c)), str(c)) for c in classes}
    correct = y_pred == y_true
    vis_idx = stratified_subsample(y_true, n_samples=n_samples, stratify=False)
    for tag, data in (("raw/classes", X), ("latent/classes", z)):
        if data is None:
            continue
        figures[f"figure/testing/{tag}"] = scatter_plot(
            tsne_projection(data[vis_idx], n_components=2),
            y_true[vis_idx],
            highlight_mask=~correct[vis_idx],
            names=names,
            marker_size=12.0,
        )

    return figures
