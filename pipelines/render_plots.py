import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pipelines import paths_from_cfg
from src.core.config import load_config, save_config
from src.core.log import (
    FilesystemFigureSubscriber,
    JSONSubscriber,
    LogBundle,
    LogDispatcher,
    setup_logger,
)
from src.core.utils import flush_timing, load_from_json, timed
from src.domain.plot.analysis_charts import dual_scatter_plot, strip_count_panel_plot
from src.domain.plot.base import Plot, set_figure_format
from src.domain.plot.primitives import bar_plot, numeric_scatter_plot, violin_plot
from src.domain.plot.style import PALETTE, apply_plot_style

setup_logger(log_file="resources/logs.txt")
apply_plot_style()
logger = logging.getLogger(__name__)


def _plot_failure_strips(
    summary_df: pd.DataFrame,
    oof_predicted_rate: dict[str, float] | None = None,
) -> dict[str, Plot]:
    """Strip plot of failure rate per class, dots coloured by RF predicted rate."""
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
    """Per-feature scatter of complexity vs failure rate, with trend line and Spearman ρ."""
    rate = summary_df["failure_rate"].to_numpy(dtype=float)
    out: dict[str, Plot] = {}
    for feature in features:
        x = summary_df[feature].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(rate)
        rho = (
            float(spearmanr(x[finite], rate[finite]).statistic)
            if int(finite.sum()) >= 3
            else float("nan")
        )
        out[f"summary/global/{feature}"] = numeric_scatter_plot(
            x,
            rate,
            color_values=rate,
            colorbar_label="failure rate",
            x_label=_feature_label(feature),
            y_label="failure rate",
            trend_line=True,
            annotations={"Spearman ρ": rho},
        )
    return out


_MEASURE_LABEL = {
    "f1": "F1",
    "f2": "F2",
    "f3": "F3",
    "f4": "F4",
    "n1": "N1",
    "n2": "N2",
    "n3": "N3",
    "n4": "N4",
    "network_density": "Density",
    "cls_coef": "ClsCoef",
    "hub": "Hubs",
    "t2": "T2",
    "t3": "T3",
    "t4": "T4",
    "max_dispersion": r"$\delta_{\max}$",
    "p95_dispersion": r"$\delta_{95}$",
    "dist_to_nearest_centroid": r"$\delta_{\mathrm{near}}$",
    "p5_silhouette": r"$s_{5}$",
    "frac_at_risk": r"$s^{-}$",
}


def _feature_label(feature: str) -> str:
    """Map a raw `{cluster|class}_{measure}[_agg]` feature name to its display notation."""
    if feature == "cluster_class":
        return "class label"
    level, _, measure = feature.partition("_")
    agg = None
    for suffix in ("_mean", "_max", "_min"):
        if measure.endswith(suffix):
            measure, agg = measure[: -len(suffix)], suffix[1:]
            break
    label = _MEASURE_LABEL.get(measure, measure)
    qualifiers = [q for q in ("class" if level == "class" else None, agg) if q]
    return f"{label} ({', '.join(qualifiers)})" if qualifiers else label


def _plot_feature_violin_by_rate_bin(
    summary_df: pd.DataFrame, features: list[str], n_bins: int = 4
) -> dict[str, Plot]:
    """Violin distribution of each complexity feature split by failure-rate quartile bins."""
    rate = summary_df["failure_rate"]
    try:
        bins = pd.qcut(rate, q=n_bins, duplicates="drop")
    except Exception:
        return {}
    if bins.nunique() < 2:
        return {}

    bin_means = rate.groupby(bins, observed=True).mean()
    categories = list(bin_means.index)
    bin_labels = [
        f"G{i + 1}\n({bin_means[cat]:.2f})" for i, cat in enumerate(categories)
    ]
    label_map = {cat: lab for cat, lab in zip(categories, bin_labels)}
    bin_str = bins.map(label_map)
    ordered = bin_labels

    out: dict[str, Plot] = {}
    for feature in features:
        x = summary_df[feature]
        valid = x.notna() & rate.notna()
        if valid.sum() < 4:
            continue
        p = violin_plot(
            categories=bin_str[valid].to_numpy(),
            values=x[valid].to_numpy(dtype=float),
            category_order=ordered,
            x_label="Failure rate bin",
            y_label=_feature_label(feature),
            show_legend=False,
            inner="box",
        )
        if p is not None:
            out[f"summary/global/{feature}_violin"] = p
    return out


def _plot_rf_evaluation(
    summary_df: pd.DataFrame, regressor_results: dict
) -> dict[str, Plot]:
    """Predicted-vs-observed scatter (regressor + MCP, shared colorbar) and feature-importance bar."""
    predicted = regressor_results["oof_predicted_rate"]
    cids = [c for c in predicted if c in summary_df.index]
    y_true = summary_df.loc[cids, "failure_rate"].to_numpy(dtype=float)
    y_pred_reg = np.array([predicted[c] for c in cids], dtype=float)
    y_pred_mcp = summary_df.loc[cids, "mcp_risk"].to_numpy(dtype=float)
    importances = regressor_results["feature_importances"]

    se_reg = (y_pred_reg - y_true) ** 2
    se_mcp = (y_pred_mcp - y_true) ** 2
    mse_reg = float(np.mean(se_reg)) if se_reg.size else float("nan")
    mse_mcp = float(np.mean(se_mcp)) if se_mcp.size else float("nan")
    finite_mcp = np.isfinite(y_pred_mcp) & np.isfinite(y_true)
    spearman_mcp = (
        float(spearmanr(y_pred_mcp[finite_mcp], y_true[finite_mcp]).statistic)
        if finite_mcp.sum() >= 2
        and np.std(y_pred_mcp[finite_mcp]) > 0
        and np.std(y_true[finite_mcp]) > 0
        else float("nan")
    )

    return {
        "summary/correlation/pred_vs_actual": dual_scatter_plot(
            y_true,
            [
                (
                    "Regressor",
                    y_pred_reg,
                    se_reg,
                    {"Spearman": regressor_results["spearman"], "MSE": mse_reg},
                ),
                (
                    "MCP",
                    y_pred_mcp,
                    se_mcp,
                    {"Spearman": spearman_mcp, "MSE": mse_mcp},
                ),
            ],
            cmap="RdYlGn_r",
            colorbar_label="Squared error (pred − obs)²",
            reference_line=True,
            x_label="Observed failure rate",
            y_label="Predicted failure rate (OOF)",
        ),
        "summary/correlation/feature_importances": bar_plot(
            labels=[_feature_label(name) for name in importances],
            values=list(importances.values()),
            orientation="v",
            sort="desc",
            top_k=10,
            annotate_values=False,
            color_gradient=True,
            y_label="Importance",
            figsize=(5.6, 3.9),
        ),
    }


@timed
def assemble_analysis_figures(
    cluster_summary: dict,
    df_meta: dict,
    regressor_results: dict,
    *,
    analysis_bus: LogDispatcher | None = None,
) -> dict[str, Plot]:
    """Build every analysis figure and publish it on the log bus."""
    logger.info("Building summary visualizations ...")
    summary_df = pd.DataFrame.from_dict(cluster_summary, orient="index")
    label_mapping = {str(k): v for k, v in df_meta["label_mapping"].items()}
    summary_df["class_name"] = (
        summary_df["cluster_class"].astype(str).map(label_mapping)
    )

    if regressor_results.get("skipped"):
        logger.warning(
            "[STAGE-SKIP] Skipping failure-classifier plots: %s",
            regressor_results.get("message", regressor_results.get("reason")),
        )
        figures: dict[str, Plot] = {}
        if analysis_bus is not None:
            analysis_bus.publish(LogBundle(figures=figures))
        return figures

    sorted_by_importance = sorted(
        regressor_results["feature_importances"].items(),
        key=lambda kv: kv[1],
        reverse=True,
    )
    top10 = [name for name, _ in sorted_by_importance[:10]]
    scatter_features = [f for f in top10 if f in summary_df.columns]

    figures: dict[str, Plot] = {}
    figures.update(
        _plot_failure_strips(summary_df, regressor_results.get("oof_predicted_rate"))
    )
    figures.update(_plot_feature_vs_failure(summary_df, scatter_features))
    figures.update(_plot_feature_violin_by_rate_bin(summary_df, scatter_features))
    figures.update(_plot_rf_evaluation(summary_df, regressor_results))
    if analysis_bus is not None:
        analysis_bus.publish(LogBundle(figures=figures))
    return figures


def main() -> None:
    """Entry point for the plot rendering stage."""
    cfg = load_config(
        config_path=Path(__file__).parent.parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )
    set_figure_format(cfg.figure_format)
    paths = paths_from_cfg(cfg)
    save_config(cfg, paths.configs / "config_composed_render.json")

    analysis_bus = LogDispatcher()
    analysis_bus.subscribe(JSONSubscriber(paths.outputs))
    analysis_bus.subscribe(FilesystemFigureSubscriber(paths.figures))

    summary_path = paths.outputs / "analysis/cluster_summary.json"
    results_path = paths.outputs / "analysis/failure_regressor_results.json"
    if summary_path.exists() and results_path.exists():
        assemble_analysis_figures(
            cluster_summary=load_from_json(summary_path),
            df_meta=load_from_json(paths.shared / "metadata/df_meta.json"),
            regressor_results=load_from_json(results_path),
            analysis_bus=analysis_bus,
        )
    else:
        logger.warning(
            "[STAGE-SKIP] Missing failure-analysis artifacts in %s; "
            "run `make failure-regress` first. Skipping summary figures.",
            paths.outputs / "analysis",
        )

    flush_timing(paths.outputs / "timing.json")


if __name__ == "__main__":
    main()
