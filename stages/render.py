import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.core.io import load_df, save_figures
from src.core.log import setup_logger
from src.core.record import clear_dir, write_record
from src.core.utils import flush_timing, load_from_json, timed
from src.domain.analysis.failure_regressor import (
    SIZE_ERROR_VARIANTS,
    SIZE_VARIANTS,
    join_region_summary,
)
from src.domain.plot.analysis_charts import dual_scatter_plot, strip_count_panel_plot
from src.domain.plot.base import Plot, set_figure_format
from src.domain.plot.comparison_charts import dual_axis_bar_plot, grouped_bar_plot
from src.domain.plot.primitives import bar_plot, numeric_scatter_plot, violin_plot
from src.domain.plot.style import (
    BASELINE_COLOR,
    BASELINE_LABEL,
    PALETTE,
    apply_plot_style,
)
from stages import load_cli_config, paths_from_cfg, stage_config, upstream_ids

setup_logger()
apply_plot_style()
logger = logging.getLogger(__name__)


def _plot_failure_strips(
    summary_df: pd.DataFrame, predicted_rate: pd.Series
) -> dict[str, Plot]:
    """Strip plot of failure rate per class, dots coloured by the predicted rate."""
    class_order = (
        summary_df.groupby("class_name")["failure_rate"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    classes = summary_df["class_name"].values
    failure_rate = summary_df["failure_rate"].values
    counts_by_class = summary_df.groupby("class_name").size().to_dict()

    fill_vals = predicted_rate.reindex(summary_df.index).to_numpy(dtype=float)

    return {
        "failure_rate_strip_box": strip_count_panel_plot(
            categories=classes,
            values=failure_rate,
            category_order=class_order,
            counts_by_class=counts_by_class,
            fill_values=fill_vals,
            fill_cmap="viridis",
            fill_cmap_label="RF predicted rate",
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
        out[f"global/{feature}"] = numeric_scatter_plot(
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
    "dist_to_nearest_rival": r"$\delta_{\mathrm{near}}$",
    "p5_silhouette": r"$s_{5}$",
    "frac_at_risk": r"$s^{-}$",
}


def _feature_label(feature: str) -> str:
    """Map a raw `{cluster|class}_{measure}[_agg]` feature name to its display notation."""
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
    bins = pd.qcut(rate, q=n_bins, duplicates="drop")
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
        out[f"global/{feature}_violin"] = violin_plot(
            categories=bin_str[valid].to_numpy(),
            values=x[valid].to_numpy(dtype=float),
            category_order=ordered,
            x_label="Failure rate bin",
            y_label=_feature_label(feature),
        )
    return out


def _plot_rf_evaluation(
    summary_df: pd.DataFrame, regressor_results: dict, predicted_rate: pd.Series
) -> dict[str, Plot]:
    """Predicted-vs-observed scatter (regressor + MCP, shared colorbar) and feature-importance bar."""
    cids = [c for c in predicted_rate.index if c in summary_df.index]
    y_true = summary_df.loc[cids, "failure_rate"].to_numpy(dtype=float)
    y_pred_reg = predicted_rate.loc[cids].to_numpy(dtype=float)
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
        "correlation/pred_vs_actual": dual_scatter_plot(
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
        "correlation/feature_importances": bar_plot(
            labels=[_feature_label(r["feature"]) for r in importances],
            values=[r["importance"] for r in importances],
            orientation="v",
            sort="desc",
            top_k=10,
            annotate_values=False,
            color_gradient=True,
            y_label="Importance",
            figsize=(5.6, 3.9),
        ),
    }


def _plot_regressor_comparison(models: list[dict]) -> dict[str, Plot]:
    """Spearman rho and MSE of every failure regressor, on the same outer folds."""
    return {
        "correlation/regressor_comparison": dual_axis_bar_plot(
            [m["model"] for m in models],
            ("Spearman ρ", [m["spearman"] for m in models], PALETTE[0]),
            ("MSE", [m["mse"] for m in models], PALETTE[1]),
            x_label="Failure regressor",
        )
    }


def _plot_error_by_size(error_by_size: list[dict]) -> dict[str, Plot]:
    """Squared error, signed error and rho per bin of region size, one bar per prediction."""
    if not error_by_size:
        return {}
    table = pd.DataFrame(error_by_size)
    labels = [
        f"{row.size_min}–{row.size_max}"
        for row in table[table["variant"] == "region"]
        .sort_values("size_bin")
        .itertuples()
    ]

    def series(field: str, variants: tuple[str, ...], *, with_se: bool) -> list[tuple]:
        out = []
        for variant in variants:
            rows = table[table["variant"] == variant].sort_values("size_bin")
            err = rows[f"{field}_se"].tolist() if with_se else None
            out.append(
                (
                    BASELINE_LABEL[variant],
                    rows[field].tolist(),
                    err,
                    err,
                    BASELINE_COLOR[variant],
                )
            )
        return out

    return {
        "baselines/mse_by_region_size": grouped_bar_plot(
            labels,
            series("mse", SIZE_ERROR_VARIANTS, with_se=True),
            x_label="Region size (train rows)",
            y_label="MSE",
            log_y=True,
        ),
        "baselines/bias_by_region_size": grouped_bar_plot(
            labels,
            series("bias", SIZE_ERROR_VARIANTS, with_se=True),
            x_label="Region size (train rows)",
            y_label="Predicted − observed rate",
            hline=0.0,
        ),
        "baselines/spearman_by_region_size": grouped_bar_plot(
            labels,
            series("spearman", SIZE_VARIANTS, with_se=False),
            x_label="Region size (train rows)",
            y_label="Spearman ρ",
            hline=0.0,
        ),
    }


@timed
def build_analysis_figures(
    summary_df: pd.DataFrame,
    meta: dict,
    regressor_results: dict,
    predicted_rate: pd.Series,
    error_by_size: list[dict],
) -> dict[str, Plot]:
    """Every analysis figure, keyed by its path under the stage's figures folder."""
    logger.info("Building summary visualizations ...")
    class_names = {c["class_id"]: c["class_name"] for c in meta["classes"]}
    summary_df = summary_df.assign(class_name=summary_df["class_id"].map(class_names))

    if regressor_results.get("skipped"):
        logger.warning(
            "[STAGE-SKIP] Skipping failure-regressor plots: %s",
            regressor_results["message"],
        )
        return {}

    ranked = sorted(
        regressor_results["feature_importances"],
        key=lambda r: r["importance"],
        reverse=True,
    )
    top10 = [r["feature"] for r in ranked[:10]]
    scatter_features = [f for f in top10 if f in summary_df.columns]

    figures: dict[str, Plot] = {}
    figures.update(_plot_failure_strips(summary_df, predicted_rate))
    figures.update(_plot_feature_vs_failure(summary_df, scatter_features))
    figures.update(_plot_feature_violin_by_rate_bin(summary_df, scatter_features))
    figures.update(_plot_rf_evaluation(summary_df, regressor_results, predicted_rate))
    figures.update(_plot_regressor_comparison(regressor_results["models"]))
    figures.update(_plot_error_by_size(error_by_size))
    return figures


def main() -> None:
    """Entry point for the render stage."""
    cfg = load_cli_config()
    set_figure_format(cfg.figure_format)
    paths = paths_from_cfg(cfg)
    stage_dir = paths.of("render")
    config = stage_config(cfg, "render")
    inputs = upstream_ids(cfg, paths, "render")

    clear_dir(stage_dir)
    failures = load_df(paths.of("regress") / "regions.parquet")
    summary_df = join_region_summary(
        load_df(paths.of("complexity") / "regions.parquet"),
        load_df(paths.of("complexity") / "classes.parquet"),
        failures,
    )
    predicted_rate = failures.set_index("region")["predicted_rate"].dropna()
    baselines_path = paths.of("regress") / "baselines.json"
    figures = build_analysis_figures(
        summary_df,
        load_from_json(paths.of("split") / "meta.json"),
        load_from_json(paths.of("regress") / "results.json"),
        predicted_rate,
        # Absent when the regressor skipped, which returns before this is read.
        (
            load_from_json(baselines_path)["error_by_size"]
            if baselines_path.exists()
            else []
        ),
    )
    save_figures(figures, stage_dir / "figures")
    flush_timing(stage_dir / "timing.json")
    write_record(stage_dir, config=config, inputs=inputs)


if __name__ == "__main__":
    main()
