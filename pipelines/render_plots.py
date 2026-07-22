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
from pipelines import paths_from_cfg
from src.core.utils import flush_timing, load_from_json, load_from_pickle, timed
from src.domain.plot.charts import (
    bar_plot,
    beeswarm_plot,
    cost_impact_plot,
    cost_quality_plot,
    cost_scaling_plot,
    numeric_scatter_plot,
    selective_accuracy_plot,
    strip_count_panel_plot,
    violin_plot,
)
from src.domain.analysis.selective_prediction import (
    macro_recall_curve,
    risk_coverage_curve,
    selective_prediction_metrics,
    selective_recall_metrics,
)
from src.domain.plot.base import Plot, set_figure_format
from src.domain.plot.style import CORRECT_COLOR, FAILED_COLOR, MUTED_COLOR, PALETTE, apply_plot_style

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
    """Per-feature scatter of complexity feature vs failure rate, with trend line and Spearman ρ."""
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
            x_label=_feature_label(feature),
            y_label="failure rate",
            trend_line=True,
            annotations={"Spearman ρ": rho},
        )
    return out


# Display names for the geometric measures, matching the paper's measure table
# (Table "measures": feature overlap F1–F4, neighbourhood N1–N4, network,
# dimensionality T2–T4, cluster geometry).
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
    """Map a raw `{cluster|class}_{measure}[_agg]` feature name to the paper's measure notation."""
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

    # Iterate over observed bins only: qcut(duplicates="drop") on few distinct
    # rates (small datasets) can leave empty interval categories that the
    # observed=True groupby drops — indexing them below would KeyError.
    bin_means = rate.groupby(bins, observed=True).mean()
    categories = list(bin_means.index)
    bin_labels = [
        f"Q{i + 1}\n({bin_means[cat]:.2f})" for i, cat in enumerate(categories)
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
    summary_df: pd.DataFrame, classifier_results: dict
) -> dict[str, Plot]:
    """Predicted-vs-observed scatter and feature-importance bar for the RF regressor."""
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
            },
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


_BASELINE_LABEL = {"mcp_risk": "MCP", "margin_risk": "Margin", "entropy_risk": "Entropy"}


def _baseline_curves(
    summary_df: pd.DataFrame, cids: list, rejection_curve
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Native-classifier-confidence baseline curves (MCP/margin/entropy), where present."""
    return {
        label: rejection_curve(summary_df.loc[cids, col].to_numpy(dtype=float))
        for col, label in _BASELINE_LABEL.items()
        if col in summary_df.columns
    }


def _plot_selective_accuracy(
    summary_df: pd.DataFrame, classifier_results: dict
) -> dict[str, Plot]:
    """Selective-accuracy curves: accuracy on the retained set as high predicted-risk clusters are rejected first."""
    predicted = classifier_results["oof_predicted_rate"]
    cids = [c for c in predicted if c in summary_df.index]
    if not cids:
        return {}
    y_pred = np.array([predicted[c] for c in cids], dtype=float)
    y_true = summary_df.loc[cids, "failure_rate"].to_numpy(dtype=float)
    support = summary_df.loc[cids, "n_test"].to_numpy(dtype=float)

    metrics = selective_prediction_metrics(y_pred, y_true, support)
    if not metrics:
        return {}

    def _rejection_curve(score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coverage, accuracy = risk_coverage_curve(score, y_true, support)
        return 1.0 - coverage, accuracy  # x = fraction rejected

    curves = {
        "Predictor": _rejection_curve(y_pred),
        "Oracle": _rejection_curve(y_true),
        **_baseline_curves(summary_df, cids, _rejection_curve),
    }
    keep = metrics["coverage_target"]
    return {
        "summary/selectivity/selective_accuracy": selective_accuracy_plot(
            curves,
            baseline=metrics["global_accuracy"],
            title="",
            annotations={
                f"Retained accuracy ({keep:.0%})": metrics["acc_at_target_predictor"],
                "Random baseline": metrics["global_accuracy"],
                "Oracle benefit recovered": metrics["oracle_benefit_recovered"],
            },
        )
    }


def _plot_selective_macro_recall(
    summary_df: pd.DataFrame, classifier_results: dict
) -> dict[str, Plot]:
    """Class-balanced counterpart to `_plot_selective_accuracy`: macro-recall on the retained set.

    The Oracle ranks by true error, so it need not dominate here — a dip below
    Random means error-greedy rejection costs class balance.
    """
    predicted = classifier_results["oof_predicted_rate"]
    cids = [c for c in predicted if c in summary_df.index]
    if not cids:
        return {}
    y_pred = np.array([predicted[c] for c in cids], dtype=float)
    y_true = summary_df.loc[cids, "failure_rate"].to_numpy(dtype=float)
    support = summary_df.loc[cids, "n_test"].to_numpy(dtype=float)
    cluster_class = summary_df.loc[cids, "cluster_class"].to_numpy()

    metrics = selective_recall_metrics(y_pred, y_true, support, cluster_class)
    if not metrics:
        return {}

    def _rejection_curve(score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coverage, recall = macro_recall_curve(score, y_true, support, cluster_class)
        return 1.0 - coverage, recall  # x = fraction rejected

    curves = {
        "Predictor": _rejection_curve(y_pred),
        "Oracle": _rejection_curve(y_true),
        **_baseline_curves(summary_df, cids, _rejection_curve),
    }
    keep = metrics["coverage_target"]
    return {
        "summary/selectivity/selective_macro_recall": selective_accuracy_plot(
            curves,
            baseline=metrics["global_macro_recall"],
            y_label="Macro-recall on retained set",
            title="",
            annotations={
                f"Retained macro-recall ({keep:.0%})": metrics["recall_at_target_predictor"],
                "Random baseline": metrics["global_macro_recall"],
                "Oracle benefit recovered": metrics["oracle_benefit_recovered"],
            },
        )
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
    figures.update(_plot_feature_violin_by_rate_bin(summary_df, scatter_features))
    figures.update(_plot_rf_evaluation(summary_df, classifier_results))
    figures.update(_plot_selective_accuracy(summary_df, classifier_results))
    figures.update(_plot_selective_macro_recall(summary_df, classifier_results))
    if analysis_bus is not None:
        analysis_bus.publish(LogBundle(figures=figures))
    return figures


def _plot_base_vs_extended_f1(
    base_summary: dict,
    ext_summary: dict,
    df_meta: dict,
) -> dict[str, Plot]:
    """Delta bar chart: per-class F1 improvement of extended over base classifier."""
    label_mapping = {str(k): v for k, v in df_meta["label_mapping"].items()}
    n = len(base_summary["f1_per_class"])
    names = [label_mapping.get(str(i), str(i)) for i in range(n)]
    delta = np.array(ext_summary["f1_per_class"]) - np.array(base_summary["f1_per_class"])

    # Pre-sort here so the colour list stays aligned with the bars (sort=None below).
    order = np.argsort(delta)
    names_sorted = [names[i] for i in order]
    delta_sorted = delta[order]
    colors = [CORRECT_COLOR if d >= 0 else FAILED_COLOR for d in delta_sorted]

    return {
        "figure/explain/f1_delta": bar_plot(
            names_sorted,
            delta_sorted.tolist(),
            orientation="h",
            color=colors,
            sort=None,
            annotate_values=True,
            value_format="{:+.3f}",
            axvline=0.0,
            axvline_color=MUTED_COLOR,
            x_label="F1 improvement (extended − base)",
        )
    }


def assemble_explain_figures(
    shap_payload: dict,
    explain_meta: dict,
    *,
    max_display: int = 20,
    explain_bus: LogDispatcher | None = None,
) -> dict[str, Plot]:
    """Build one SHAP beeswarm per class plus a global importance bar from persisted SHAP values."""
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

    mean_abs = np.abs(values).mean(axis=(0, 2))  # (n_features,)
    top_k = min(max_display, len(mean_abs))
    idx = np.argsort(mean_abs)[-top_k:]  # ascending: least→most important (top of barh)
    figures["figure/explain/global_importance"] = bar_plot(
        [feature_names[i] for i in idx],
        [float(mean_abs[i]) for i in idx],
        orientation="h",
        sort=None,
        color_gradient=True,
        x_label="Mean |SHAP| (all classes)",
        title="Global SHAP feature importance (extended classifier)",
    )

    if explain_bus is not None:
        explain_bus.publish(LogBundle.from_dict(figures))
    return figures


def _aggregate_cost_models(root: Path) -> tuple[dict, dict, dict, float | None]:
    """Scan `<root>/**/shared/cost_model.json`, group by distance, and aggregate per-seed fits (points, fits, share, m_prod)."""
    groups: dict[str, list[dict]] = {}
    for cm_path in sorted(root.glob("**/shared/cost_model.json")):
        cm = load_from_json(cm_path)
        if cm.get("cost_model", {}).get("alpha") is None:
            logger.warning("skip %s (degenerate fit, alpha=None)", cm_path)
            continue
        groups.setdefault(cm.get("distance"), []).append(cm)

    points, fits, share = {}, {}, {}
    for dist, cms in groups.items():
        per_m: dict[float, list[float]] = {}
        for cm in cms:
            for g in cm["m_grid"]:
                per_m.setdefault(g["m"], []).append(g["build_s"])
        ms = sorted(per_m)
        points[dist] = {
            "m": np.array(ms, dtype=float),
            "build_mean": np.array([float(np.mean(per_m[m])) for m in ms]),
            "build_min": np.array([float(np.min(per_m[m])) for m in ms]),
            "build_max": np.array([float(np.max(per_m[m])) for m in ms]),
        }
        alphas = np.array([cm["cost_model"]["alpha"] for cm in cms], dtype=float)
        cs = np.array([cm["cost_model"]["c"] for cm in cms], dtype=float)
        r2s = [cm["cost_model"]["r2"] for cm in cms if cm["cost_model"].get("r2") is not None]
        fits[dist] = {
            "alpha_mean": float(alphas.mean()),
            "alpha_std": float(alphas.std(ddof=1)) if len(cms) > 1 else 0.0,
            "c_mean": float(cs.mean()),
            "r2_min": min(r2s) if r2s else None,
        }
        comp = np.array([cm["pipeline_cost"]["complexity_build_s_pred"] for cm in cms], dtype=float)
        rest = np.array([cm["pipeline_cost"]["non_complexity_s"] for cm in cms], dtype=float)
        prep = np.array([cm["pipeline_cost"]["prep_clustering_s"] for cm in cms], dtype=float)
        clf = np.array([cm["pipeline_cost"]["classify_s"] for cm in cms], dtype=float)
        shr = np.array([cm["pipeline_cost"]["complexity_share_pred"] for cm in cms], dtype=float)
        share[dist] = {
            "complexity_s": float(comp.mean()),
            "prep_clustering_s": float(prep.mean()),
            "classify_s": float(clf.mean()),
            "non_complexity_s": float(rest.mean()),
            "share": float(shr.mean()),
        }

    all_mprod = [
        cm["pipeline_cost"]["m_prod"]
        for cms in groups.values()
        for cm in cms
        if cm.get("pipeline_cost", {}).get("m_prod") is not None
    ]
    m_prod = float(np.mean(all_mprod)) if all_mprod else None
    return points, fits, share, m_prod


def _render_cost_model(root: Path, fmt: str = "pdf", out: Path | None = None) -> None:
    """Aggregate the cost-model JSONs under `root` and write the two cost-model paper figures to `out`."""
    set_figure_format(fmt)
    points, fits, share, m_prod = _aggregate_cost_models(root)
    if not points:
        logger.warning("No usable cost_model.json found under %s; nothing to render.", root)
        return
    figures = {
        "figure/cost_model_scaling": cost_scaling_plot(points, fits, m_prod=m_prod),
        "figure/cost_model_impact": cost_impact_plot(share),
    }
    base = out or root
    bus = LogDispatcher()
    bus.subscribe(FilesystemFigureSubscriber(base))
    bus.publish(LogBundle.from_dict(figures))
    logger.info(
        "Cost-model figures (%s) -> %s/{cost_model_scaling,cost_model_impact}.%s",
        ", ".join(sorted(points)), base, fmt,
    )


_COS, _EUC = PALETTE[0], PALETTE[1]

_DATASET_LABEL = {
    "bank_marketing": "Bank",
    "bot_iot_v2": "Bot-IoT",
    "cic_ids2018_v2": "CIC",
    "covertype": "Covertype",
    "letter_recognition": "Letter",
    "statlog_landsat_satellite": "Statlog",
    "thyroid_disease": "Thyroid",
    "ton_iot_v2": "ToN-IoT",
    "unsw_nb15_v2": "UNSW",
}


def _render_cost_quality(table: Path, fmt: str = "pdf", out: Path | None = None) -> None:
    """Render the cost-vs-quality figure from a cost-quality table (file or containing directory)."""
    set_figure_format(fmt)
    table_path = table / "cost_quality_table.json" if table.is_dir() else table
    rows = load_from_json(table_path)["rows"]

    def _median(dataset: str, distance: str, key: str) -> float | None:
        vals = [r[key] for r in rows
                if r["dataset"] == dataset and r["distance"] == distance
                and r.get(key) is not None]
        return float(np.median(vals)) if vals else None

    datasets = sorted(
        {r["dataset"] for r in rows},
        key=lambda d: _median(d, "cosine", "complexity_build_s") or float("inf"),
    )
    labels, cosine, euclidean = [], [], []
    for d in datasets:
        labels.append(_DATASET_LABEL.get(d, d))
        cos_c = _median(d, "cosine", "complexity_build_s")
        cos_r = _median(d, "cosine", "rho")
        euc_c = _median(d, "euclidean", "complexity_build_s")
        euc_r = _median(d, "euclidean", "rho")
        cosine.append((cos_c, cos_r) if cos_c is not None and cos_r is not None else None)
        euclidean.append((euc_c, euc_r) if euc_c is not None and euc_r is not None else None)

    plot = cost_quality_plot(labels, cosine, euclidean, cos_color=_COS, euc_color=_EUC)
    base = out or table_path.parent
    bus = LogDispatcher()
    bus.subscribe(FilesystemFigureSubscriber(base))
    bus.publish(LogBundle.from_dict({"figure/cost_quality": plot}))
    logger.info("Cost-quality figure (%d datasets) -> %s/cost_quality.%s", len(datasets), base, fmt)


def _parse_render_args(argv: list[str], key: str) -> tuple[Path, str, Path | None] | None:
    """Parse `key=<path> [format=..] [out=..]` from argv; None if `key` is absent."""
    if not any(a.startswith(f"{key}=") for a in argv):
        return None
    kv = dict(
        a.split("=", 1) for a in argv
        if "=" in a and a.split("=", 1)[0] in (key, "format", "out")
    )
    return (
        Path(kv[key]),
        kv.get("format", "pdf"),
        Path(kv["out"]) if kv.get("out") else None,
    )


def main():
    """Main entry point for plot rendering."""
    argv = sys.argv[1:]
    special = (
        ("cost_quality", _render_cost_quality),
        ("cost_model", _render_cost_model),
    )
    for key, render_fn in special:
        parsed = _parse_render_args(argv, key)
        if parsed is not None:
            path, fmt, out = parsed
            render_fn(path, fmt=fmt, out=out)
            return

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

    summary_base_path = paths.outputs / "testing" / "summary.json"
    summary_ext_path = paths.outputs / "testing" / "summary_extended.json"
    if summary_base_path.exists() and summary_ext_path.exists():
        delta_figures = _plot_base_vs_extended_f1(
            load_from_json(summary_base_path),
            load_from_json(summary_ext_path),
            load_from_json(paths.shared / "metadata/df_meta.json"),
        )
        analysis_bus.publish(LogBundle.from_dict(delta_figures))
    else:
        logger.warning(
            "[STAGE-SKIP] Missing base or extended summary in %s; run both `make classify` and `make classify-extended` first. Skipping F1 delta figure.",
            paths.outputs / "testing",
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
