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
    band_curve_plot,
    bar_plot,
    beeswarm_plot,
    box_strip_plot,
    cost_model_plot,
    cost_quality_plot,
    line_whisker_plot,
    numeric_scatter_plot,
    selective_accuracy_plot,
    stacked_bar_plot,
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


def _plot_feature_violin_by_rate_bin(
    summary_df: pd.DataFrame, features: list[str], n_bins: int = 4
) -> dict[str, Plot]:
    """Violin distribution of each complexity feature split by failure-rate quartile bins.

    Each violin shows how the feature distributes within clusters that share a
    similar observed failure rate. A clear shift from Q1 to Q4 confirms the
    feature discriminates easy vs hard regions.
    """
    rate = summary_df["failure_rate"]
    try:
        bins = pd.qcut(rate, q=n_bins, duplicates="drop")
    except Exception:
        return {}
    if bins.nunique() < 2:
        return {}

    midpoints = [interval.mid for interval in bins.cat.categories]
    bin_labels = [f"Q{i + 1}\n({m:.2f})" for i, m in enumerate(midpoints)]
    label_map = {cat: lab for cat, lab in zip(bins.cat.categories, bin_labels)}
    bin_str = bins.map(label_map)
    ordered = [label_map[c] for c in bins.cat.categories]

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
            y_label=feature,
            show_legend=False,
            inner="box",
        )
        if p is not None:
            out[f"summary/global/{feature}_violin"] = p
    return out


def _plot_rf_evaluation(
    summary_df: pd.DataFrame, classifier_results: dict
) -> dict[str, Plot]:
    """Predicted-vs-observed scatter and importance bar for the RF regressor.

    Each dot is a cluster (X = observed rate, Y = OOF prediction), coloured by
    observed rate; the dashed y=x line is perfect prediction.
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
        ),
        "summary/correlation/feature_importances": bar_plot(
            labels=list(importances.keys()),
            values=list(importances.values()),
            orientation="v",
            sort="desc",
            top_k=10,
            annotate_values=False,
            color_gradient=True,
            y_label="Importance",
        ),
    }


def _plot_selective_accuracy(
    summary_df: pd.DataFrame, classifier_results: dict
) -> dict[str, Plot]:
    """Selective-accuracy curves: reject high predicted-risk clusters first.

    X = fraction of test traffic rejected (riskiest first), Y = accuracy on the
    remainder. Predictor rejects by predicted rate, Oracle by the true rate
    (ceiling), Random is the flat global-accuracy baseline.
    """
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
    """Class-balanced selective curve: macro-recall on the retained set as the
    riskiest clusters are rejected.

    Counterpart to `_plot_selective_accuracy` for imbalanced settings. The Oracle
    ranks by true error, so here it need not dominate — a dip below Random means
    error-greedy rejection costs class balance.
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

    # Pre-sort to avoid the known bar_plot bug where color list is not reindexed with sort
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

    # Global importance: mean |SHAP| aggregated over samples and classes
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
    """Scan `<root>/**/shared/cost_model.json`, group by distance, and aggregate
    the per-seed fits into the dicts `cost_model_plot` expects (points, fits,
    share) plus the production operating point `m_prod`. Degenerate fits
    (alpha=None) are skipped."""
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
    """Aggregate the cost-model JSONs under `root` and write the paper figure
    (`cost_model.<fmt>`) to `out` (defaults to `root`) via the figure subscriber."""
    set_figure_format(fmt)
    points, fits, share, m_prod = _aggregate_cost_models(root)
    if not points:
        logger.warning("No usable cost_model.json found under %s; nothing to render.", root)
        return
    plot = cost_model_plot(points, fits, share, m_prod=m_prod)
    base = out or root
    bus = LogDispatcher()
    bus.subscribe(FilesystemFigureSubscriber(base))
    bus.publish(LogBundle.from_dict({"figure/cost_model": plot}))
    logger.info("Cost-model figure (%s) -> %s/cost_model.%s", ", ".join(sorted(points)), base, fmt)


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
    """Render the cost-vs-quality figure (Deliverable B) from a cost-quality table.

    `table` is either the `cost_quality_table.json` file or a directory containing
    it. Rows are grouped per (dataset, distance) into median build time and median
    Spearman rho, drawn as one cosine/euclidean point pair per dataset.
    """
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


# ---- sweep-level paper figures -------------------------------------------------

_ALGO_ORDER = ["kmeans", "spectral", "birch", "hdbscan"]
_ALGO_LABEL = {"kmeans": "$k$-means", "spectral": "spectral", "birch": "BIRCH", "hdbscan": "HDBSCAN"}
_FEATURE_FAMILIES = [
    ("feature-overlap", ("f1", "f2", "f3", "f4")),
    ("neighbourhood", ("n1", "n2", "n3", "n4")),
    ("network", ("network_density", "cls_coef", "hub")),
    ("cluster-geometry",
     ("max_dispersion", "p95_dispersion", "dist_to_nearest_centroid", "p5_silhouette", "frac_at_risk")),
    ("dimensionality", ("t2", "t3", "t4")),
]
_COS, _EUC, _ORACLE = PALETTE[0], PALETTE[1], PALETTE[2]


def _load_sweep_runs(root: Path) -> list[dict]:
    """Collect one record per `<config>/<dataset>/<classifier>` run under `root`.

    `distance` and `algorithm` are read from each run's composed config (the
    verifiable source), not parsed from directory names. Runs without a results
    file are skipped.
    """
    runs: list[dict] = []
    for cfg_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for ds_dir in sorted(p for p in cfg_dir.iterdir() if p.is_dir()):
            cfg_path = ds_dir / "shared/config_composed.json"
            if not cfg_path.exists():
                continue
            composed = load_from_json(cfg_path)
            algorithm = next(iter(composed["clustering"]["algorithms"]), None)
            for clf_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir() and p.name != "shared"):
                base = clf_dir / "outputs/analysis"
                results_path = base / "classifier_results.json"
                if not results_path.exists():
                    continue
                runs.append({
                    "config": cfg_dir.name,
                    "dataset": ds_dir.name,
                    "clf": clf_dir.name,
                    "distance": composed.get("distance"),
                    "algorithm": algorithm,
                    "base": base,
                    "results": load_from_json(results_path),
                })
    return runs


def _dist_color(distance: str) -> str:
    return _COS if distance == "cosine" else _EUC


def _fig_rho_by_config(runs: list[dict]) -> Plot | None:
    """Figure A: Spearman rho distribution per clustering configuration."""
    def sort_key(cfg: str) -> tuple[int, int]:
        meta = next(r for r in runs if r["config"] == cfg)
        algo = meta["algorithm"]
        return (0 if meta["distance"] == "cosine" else 1,
                _ALGO_ORDER.index(algo) if algo in _ALGO_ORDER else 99)

    labels, values, colors, faded = [], [], [], []
    for cfg in sorted({r["config"] for r in runs}, key=sort_key):
        crows = [r for r in runs if r["config"] == cfg]
        vals = [r["results"]["spearman"] for r in crows if r["results"].get("spearman") is not None]
        if not vals:
            continue
        meta = crows[0]
        is_hdb = meta["algorithm"] == "hdbscan"
        label = f"{meta['distance']} {_ALGO_LABEL.get(meta['algorithm'], meta['algorithm'])}"
        labels.append(label + (r"$^\dagger$" if is_hdb else ""))
        values.append(np.asarray(vals, dtype=float))
        colors.append(_dist_color(meta["distance"]))
        faded.append(False)
    return box_strip_plot(
        labels, values, colors=colors, faded=faded, show_points=False,
        x_label=r"Spearman $\rho$", x_lim=(-1.05, 1.05), axvline=0.0,
        legend={"cosine": _COS, "euclidean": _EUC},
    )


def _fig_rho_vs_clusters(runs: list[dict]) -> Plot | None:
    """Figure C: Spearman rho against the number of clusters per run."""
    series: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    for distance in ("cosine", "euclidean"):
        xs, ys = [], []
        for r in runs:
            if r["distance"] != distance:
                continue
            n = r["results"].get("n_clusters_used")
            rho = r["results"].get("spearman")
            if n and rho is not None:
                xs.append(n)
                ys.append(rho)
        series[distance] = (np.asarray(xs, float), np.asarray(ys, float), _dist_color(distance))
    return line_whisker_plot(
        series, x_label="number of clusters per run (log scale)",
        y_label=r"Spearman $\rho$", log_x=True, y_lim=(-1.05, 1.05),
        vline=10, vline_label="unstable regime", hline=0.0,
    )


def _fig_family_importance(runs: list[dict]) -> Plot | None:
    """Figure E: feature-family importance, cluster- vs class-level (full sweep)."""
    acc: dict[str, list[float]] = {}
    for r in runs:
        for k, v in r["results"].get("feature_importances", {}).items():
            acc.setdefault(k, []).append(v)
    if not acc:
        return None
    mean_imp = {k: float(np.mean(v)) for k, v in acc.items()}

    def part(prefix: str, members: tuple[str, ...]) -> float:
        return 100.0 * sum(
            v for k, v in mean_imp.items()
            if k.startswith(prefix) and any(k[len(prefix):].startswith(m) for m in members)
        )

    names = [name for name, _ in _FEATURE_FAMILIES]
    cluster = [part("cluster_", members) for _, members in _FEATURE_FAMILIES]
    klass = [part("class_", members) for _, members in _FEATURE_FAMILIES]
    return stacked_bar_plot(
        names,
        [("cluster-level", cluster, _COS), ("class-level", klass, _EUC)],
        x_label="mean importance (% of total)", total_format="{:.1f}%",
    )


def _fig_selective_sweep(runs: list[dict]) -> Plot | None:
    """Figure B: accuracy-coverage curves aggregated over the full sweep.

    Each run's curve is reconstructed from its OOF predictions and cluster
    summary via `risk_coverage_curve`, interpolated onto a shared grid, then
    aggregated (median, with an IQR band for the predictor).
    """
    grid = np.linspace(0.0, 1.0, 101)
    predictor, oracle, randoms = [], [], []
    for r in runs:
        oof = r["results"].get("oof_predicted_rate", {})
        summary_path = r["base"] / "cluster_summary.json"
        if not oof or not summary_path.exists():
            continue
        summary = load_from_json(summary_path)
        ids = [c for c in oof if c in summary]
        if len(ids) < 2:
            continue
        pred = np.array([oof[c] for c in ids], dtype=float)
        fail = np.array([summary[c]["failure_rate"] for c in ids], dtype=float)
        support = np.array([summary[c]["n_test"] for c in ids], dtype=float)
        if support.sum() <= 0:
            continue
        cov_p, acc_p = risk_coverage_curve(pred, fail, support)
        cov_o, acc_o = risk_coverage_curve(fail, fail, support)
        predictor.append(np.interp(grid, cov_p, acc_p))
        oracle.append(np.interp(grid, cov_o, acc_o))
        randoms.append(1.0 - (fail * support).sum() / support.sum())
    if not predictor:
        return None

    # Mirror coverage -> fraction rejected (riskiest first), so the curves rise
    # left-to-right and match the single-run selective figure (Fig.~\ref{fig:cic_selective}).
    pred_curves = np.vstack(predictor)[:, ::-1]
    oracle_curves = np.vstack(oracle)[:, ::-1]
    return band_curve_plot(
        grid,
        [("Oracle", np.median(oracle_curves, axis=0), _ORACLE),
         ("Predictor", np.median(pred_curves, axis=0), _COS)],
        band=(np.percentile(pred_curves, 25, axis=0),
              np.percentile(pred_curves, 75, axis=0), _COS, "Predictor (IQR across runs)"),
        baseline=float(np.median(randoms)),
        x_label="fraction rejected (riskiest first)",
        y_label="accuracy on retained points",
    )


def _render_sweep(root: Path, fmt: str = "pdf", out: Path | None = None) -> None:
    """Aggregate the sweep under `root` and write the four sweep-level paper
    figures to `out` (defaults to `root`).

    All clustering configurations, including HDBSCAN, contribute to every
    figure; HDBSCAN's higher variability is reported rather than hidden.
    """
    set_figure_format(fmt)
    runs = _load_sweep_runs(root)
    if not runs:
        logger.warning("No sweep runs found under %s; nothing to render.", root)
        return
    n_hdb = sum(1 for r in runs if r["algorithm"] == "hdbscan")
    logger.info(
        "Sweep figures from %d runs (%d main, %d hdbscan) under %s",
        len(runs), len(runs) - n_hdb, n_hdb, root,
    )

    figures = {
        "figure/rho_by_config": _fig_rho_by_config(runs),
        "figure/rho_vs_clusters": _fig_rho_vs_clusters(runs),
        "figure/family_importance": _fig_family_importance(runs),
        "figure/selective_sweep": _fig_selective_sweep(runs),
    }
    figures = {k: v for k, v in figures.items() if v is not None}

    base = out or root
    bus = LogDispatcher()
    bus.subscribe(FilesystemFigureSubscriber(base))
    bus.publish(LogBundle.from_dict(figures))
    logger.info(
        "Sweep figures (%s) -> %s",
        ", ".join(sorted(k.split("/")[-1] for k in figures)), base,
    )


def main():
    """Main entry point for plot rendering."""
    argv = sys.argv[1:]
    if any(a.startswith("sweep=") for a in argv):
        kv = dict(
            a.split("=", 1) for a in argv
            if "=" in a and a.split("=", 1)[0] in ("sweep", "format", "out")
        )
        _render_sweep(
            Path(kv["sweep"]),
            fmt=kv.get("format", "pdf"),
            out=Path(kv["out"]) if kv.get("out") else None,
        )
        return
    if any(a.startswith("cost_quality=") for a in argv):
        kv = dict(
            a.split("=", 1) for a in argv
            if "=" in a and a.split("=", 1)[0] in ("cost_quality", "format", "out")
        )
        _render_cost_quality(
            Path(kv["cost_quality"]),
            fmt=kv.get("format", "pdf"),
            out=Path(kv["out"]) if kv.get("out") else None,
        )
        return
    if any(a.startswith("cost_model=") for a in argv):
        kv = dict(
            a.split("=", 1) for a in argv
            if "=" in a and a.split("=", 1)[0] in ("cost_model", "format", "out")
        )
        _render_cost_model(
            Path(kv["cost_model"]),
            fmt=kv.get("format", "pdf"),
            out=Path(kv["out"]) if kv.get("out") else None,
        )
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
