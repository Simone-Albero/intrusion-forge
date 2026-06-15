import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.core.config import load_config, save_config
from src.core.log import (
    FilesystemFigureSubscriber,
    JSONSubscriber,
    LogBundle,
    LogDispatcher,
    setup_logger,
)
from pipelines.common import paths_from_cfg
from src.core.utils import flush_timing, load_from_json, load_from_pickle, timed
from src.domain.plot.charts import (
    bar_plot,
    beeswarm_plot,
    numeric_scatter_plot,
    strip_count_panel_plot,
)
from src.domain.plot.base import Plot, set_figure_format
from src.domain.plot.style import PALETTE, apply_plot_style

setup_logger(log_file="resources/logs.txt")
apply_plot_style()
logger = logging.getLogger(__name__)


def _plot_failure_strips(
    summary_df: pd.DataFrame,
    oof_predicted_rate: dict[str, float] | None = None,
) -> dict[str, Plot]:
    """Strip plot of failure rate per class, dots colored by RF predicted rate.

    When `oof_predicted_rate` is provided each dot's color encodes the RF
    prediction — dots whose color mismatches their x-axis position are
    prediction errors at a glance.
    """
    class_order = (
        summary_df.groupby("class_name")["failure_rate"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    classes = summary_df["class_name"].values
    failure_rate = summary_df["failure_rate"].values
    counts_by_class = summary_df.groupby("class_name").size().to_dict()

    if oof_predicted_rate is not None:
        cids = [str(cid) for cid in summary_df.index]
        fill_vals = np.array(
            [oof_predicted_rate.get(cid, np.nan) for cid in cids], dtype=float
        )
        fill_cmap: str | None = "viridis"
        fill_cmap_label = "RF predicted rate"
        fill_categorical_colors: tuple[str, ...] = ()
    else:
        fill_vals = np.zeros(len(classes), dtype=float)
        fill_cmap = None
        fill_cmap_label = ""
        fill_categorical_colors = (PALETTE[0],)

    return {
        "summary/failure_rate_strip_box": strip_count_panel_plot(
            categories=classes,
            values=failure_rate,
            category_order=class_order,
            counts_by_class=counts_by_class,
            fill_values=fill_vals,
            fill_categorical_colors=fill_categorical_colors,
            fill_cmap=fill_cmap,
            fill_cmap_label=fill_cmap_label,
            x_label="Failure rate",
        ),
    }


def _plot_feature_vs_failure(
    summary_df: pd.DataFrame, features: list[str]
) -> dict[str, Plot]:
    """Per-feature scatter of complexity feature vs failure rate.

    A linear trend line (dashed) shows the direction of the relationship; the
    cloud spread around it communicates that each feature alone is insufficient.
    Spearman ρ quantifies the monotonic association without assuming linearity.
    """
    rate = summary_df["failure_rate"].to_numpy(dtype=float)
    out: dict[str, Plot] = {}
    for feature in features:
        x = summary_df[feature].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(rate)
        rho = float(spearmanr(x[finite], rate[finite]).statistic) if int(finite.sum()) >= 3 else float("nan")
        out[f"summary/global/{feature}"] = numeric_scatter_plot(
            x,
            rate,
            color_values=rate,
            colorbar_label="failure rate",
            x_label=feature,
            y_label="failure rate",
            trend_line=True,
            annotations={"Spearman ρ": rho},
        )
    return out


def _plot_rf_evaluation(
    summary_df: pd.DataFrame, classifier_results: dict
) -> dict[str, Plot]:
    """Predicted-vs-observed scatter and importance bar for the RF regressor.

    Each dot is a cluster. X = observed failure rate, Y = RF OOF prediction.
    Dots colored by observed rate (viridis): warm = high rate, cool = low rate.
    Mismatches between dot color and y-axis position expose systematic bias —
    e.g. yellow dots (high rate) below the diagonal = RF underestimates hard
    clusters. The dashed y=x reference line is perfect prediction.
    """
    predicted = classifier_results["oof_predicted_rate"]
    cids = [c for c in predicted if c in summary_df.index]
    y_pred = np.array([predicted[c] for c in cids], dtype=float)
    y_true = summary_df.loc[cids, "failure_rate"].to_numpy(dtype=float)
    importances = classifier_results["feature_importances"]

    return {
        "summary/correlation/pred_vs_actual": numeric_scatter_plot(
            y_true,
            y_pred,
            color_values=y_true,
            colorbar_label="Observed failure rate",
            reference_line=True,
            annotations={
                "Spearman": classifier_results["spearman"],
                "R²": classifier_results["r2"],
                "MAE": classifier_results["mae"],
            },
            x_label="Observed failure rate",
            y_label="Predicted failure rate (OOF)",
            title="Failure-rate regression (OOF)",
        ),
        "summary/correlation/feature_importances": bar_plot(
            labels=list(importances.keys()),
            values=list(importances.values()),
            orientation="h",
            sort="asc",
            top_k=20,
            annotate_values=False,
            color_gradient=True,
            x_label="Importance",
        ),
    }


@timed
def assemble_analysis_figures(
    cluster_summary: dict,
    df_meta: dict,
    classifier_results: dict,
    *,
    analysis_bus: LogDispatcher | None = None,
) -> dict[str, Plot]:
    """Build all analysis figures and publish to log bus."""
    logger.info("Building summary visualizations ...")
    summary_df = pd.DataFrame.from_dict(cluster_summary, orient="index")
    label_mapping = {str(k): v for k, v in df_meta["label_mapping"].items()}
    summary_df["class_name"] = (
        summary_df["cluster_class"].astype(str).map(label_mapping)
    )

    if classifier_results.get("skipped"):
        logger.warning(
            "[STAGE-SKIP] Skipping failure-classifier plots: %s",
            classifier_results.get("message", classifier_results.get("reason")),
        )
        figures: dict[str, Plot] = {}
        if analysis_bus is not None:
            analysis_bus.publish(LogBundle(figures=figures))
        return figures

    sorted_by_importance = sorted(
        classifier_results["feature_importances"].items(),
        key=lambda kv: kv[1],
        reverse=True,
    )
    top10 = [name for name, _ in sorted_by_importance[:10]]
    scatter_features = [f for f in top10 if f in summary_df.columns]

    figures: dict[str, Plot] = {}
    figures.update(_plot_failure_strips(summary_df, classifier_results.get("oof_predicted_rate")))
    figures.update(_plot_feature_vs_failure(summary_df, scatter_features))
    figures.update(_plot_rf_evaluation(summary_df, classifier_results))
    if analysis_bus is not None:
        analysis_bus.publish(LogBundle(figures=figures))
    return figures


def assemble_explain_figures(
    shap_payload: dict,
    explain_meta: dict,
    *,
    max_display: int = 20,
    explain_bus: LogDispatcher | None = None,
) -> dict[str, Plot]:
    """Build one SHAP beeswarm per class from persisted SHAP values."""
    values = np.asarray(shap_payload["values"])  # (n, f, c)
    data = np.asarray(shap_payload["data"])  # (n, f)
    feature_names = explain_meta["feature_names"]
    class_names = explain_meta["class_names"]

    figures: dict[str, Plot] = {}
    for k, name in enumerate(class_names):
        slug = re.sub(r"[^0-9A-Za-z._-]+", "_", str(name)).strip("_") or str(k)
        figures[f"figure/explain/beeswarm_{slug}"] = beeswarm_plot(
            values[:, :, k],
            data,
            feature_names,
            max_display=max_display,
            title=f"SHAP — class '{name}'",
        )
    if explain_bus is not None:
        explain_bus.publish(LogBundle.from_dict(figures))
    return figures


def main():
    """Main entry point for plot rendering."""
    cfg = load_config(
        config_path=Path(__file__).parent.parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )
    set_figure_format(cfg.plots.format)
    paths = paths_from_cfg(cfg)
    save_config(cfg, paths.configs / "config_composed_render.json")

    analysis_bus = LogDispatcher()
    analysis_bus.subscribe(JSONSubscriber(paths.outputs))
    analysis_bus.subscribe(FilesystemFigureSubscriber(paths.figures))

    summary_path = paths.outputs / "analysis/cluster_summary.json"
    results_path = paths.outputs / "analysis/classifier_results.json"
    if summary_path.exists() and results_path.exists():
        assemble_analysis_figures(
            cluster_summary=load_from_json(summary_path),
            df_meta=load_from_json(paths.shared / "metadata/df_meta.json"),
            classifier_results=load_from_json(results_path),
            analysis_bus=analysis_bus,
        )
    else:
        logger.warning(
            "[STAGE-SKIP] Missing failure-analysis artifacts in %s; run `make failure-classify` first. Skipping summary figures.",
            paths.outputs / "analysis",
        )

    shap_path = paths.pickle / "explain/shap_values.pkl"
    meta_path = paths.outputs / "explain/meta.json"
    if shap_path.exists() and meta_path.exists():
        explain_bus = LogDispatcher()
        explain_bus.subscribe(FilesystemFigureSubscriber(paths.figures))
        assemble_explain_figures(
            load_from_pickle(shap_path),
            load_from_json(meta_path),
            max_display=cfg.extend.max_display,
            explain_bus=explain_bus,
        )
    else:
        logger.warning(
            "[STAGE-SKIP] Missing SHAP artifacts under %s; run `make classify-extended` first. Skipping beeswarm figures.",
            paths.pickle / "explain",
        )

    flush_timing(paths.outputs / "timing.json")


if __name__ == "__main__":
    main()
