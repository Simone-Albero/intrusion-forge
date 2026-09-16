import logging
import sys
from pathlib import Path

import numpy as np

from src.core.log import (
    FilesystemFigureSubscriber,
    JSONSubscriber,
    LogBundle,
    LogDispatcher,
    setup_logger,
)
from src.core.utils import load_from_json
from src.domain.plot.base import Plot, set_figure_format
from src.domain.plot.charts import (
    box_strip_facets,
    box_strip_plot,
    line_whisker_plot,
    stacked_bar_plot,
)
from src.domain.plot.style import PALETTE, apply_plot_style

setup_logger(log_file="resources/logs.txt")
apply_plot_style()
logger = logging.getLogger(__name__)

_ALGO_ORDER = ["kmeans", "spectral", "birch", "hdbscan"]
_CLF_MERGE = {"numerical": "mlp", "tabular": "mlp"}
_ALGO_LABEL = {
    "kmeans": "$k$-means",
    "spectral": "spectral",
    "birch": "BIRCH",
    "hdbscan": "HDBSCAN",
}
_FEATURE_FAMILIES = [
    ("feature-overlap", ("f1", "f2", "f3", "f4")),
    ("neighbourhood", ("n1", "n2", "n3", "n4")),
    ("network", ("network_density", "cls_coef", "hub")),
    (
        "cluster-geometry",
        (
            "max_dispersion",
            "p95_dispersion",
            "dist_to_nearest_centroid",
            "p5_silhouette",
            "frac_at_risk",
        ),
    ),
    ("dimensionality", ("t2", "t3", "t4")),
]
_COS, _EUC = PALETTE[0], PALETTE[1]
_BASELINE_COLOR = {
    "mcp_risk": PALETTE[3],
    "margin_risk": PALETTE[4],
    "entropy_risk": PALETTE[5],
}
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
_CLF_LABEL = {
    "decision_tree": "Decision Tree",
    "naive_bayes": "Naive Bayes",
    "lda": "LDA",
    "linear_svc": "Linear SVC",
    "logistic_regression": "Logistic Reg.",
    "mlp": "MLP",
    "knn": "$k$-NN",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGB",
    "xgboost": "XGBoost",
}


def _load_sweep_runs(root: Path) -> list[dict]:
    """Collect one record per `<config>/<dataset>/<classifier>` run under `root`."""
    runs: list[dict] = []
    for cfg_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for ds_dir in sorted(p for p in cfg_dir.iterdir() if p.is_dir()):
            cfg_path = ds_dir / "shared/config_composed.json"
            if not cfg_path.exists():
                continue
            composed = load_from_json(cfg_path)
            algorithm = next(iter(composed["clustering"]["algorithms"]), None)
            dataset_size = _dataset_size(ds_dir / "shared/metadata/df_info.json")
            for clf_dir in sorted(
                p for p in ds_dir.iterdir() if p.is_dir() and p.name != "shared"
            ):
                base = clf_dir / "outputs/analysis"
                results_path = base / "classifier_results.json"
                if not results_path.exists():
                    continue
                runs.append(
                    {
                        "config": cfg_dir.name,
                        "dataset": ds_dir.name,
                        "clf": clf_dir.name,
                        "distance": composed.get("distance"),
                        "algorithm": algorithm,
                        "dataset_size": dataset_size,
                        "base": base,
                        "results": load_from_json(results_path),
                    }
                )
    return runs


def _dataset_size(info_path: Path) -> int | None:
    """Full training-set row count for x-axis ordering, from the shared df_info.json."""
    if not info_path.exists():
        return None
    return int(load_from_json(info_path)["shape"][0])


def _dataset_base(name: str) -> str:
    """Strip a trailing `_<seed>` suffix from a run's dataset directory name."""
    head, _, tail = name.rpartition("_")
    return head if tail.isdigit() else name


def _dist_color(distance: str) -> str:
    """Palette colour of a distance metric."""
    return _COS if distance == "cosine" else _EUC


def _fig_rho_by_config(runs: list[dict]) -> Plot | None:
    """Figure A: Spearman rho distribution per clustering configuration."""

    def sort_key(cfg: str) -> tuple[int, int]:
        meta = next(r for r in runs if r["config"] == cfg)
        algo = meta["algorithm"]
        return (
            0 if meta["distance"] == "cosine" else 1,
            _ALGO_ORDER.index(algo) if algo in _ALGO_ORDER else 99,
        )

    labels, values, colors, faded = [], [], [], []
    for cfg in sorted({r["config"] for r in runs}, key=sort_key):
        crows = [r for r in runs if r["config"] == cfg]
        vals = [
            r["results"]["spearman"]
            for r in crows
            if r["results"].get("spearman") is not None
        ]
        if not vals:
            continue
        meta = crows[0]
        label = f"{meta['distance']} {_ALGO_LABEL.get(meta['algorithm'], meta['algorithm'])}"
        labels.append(label)
        values.append(np.asarray(vals, dtype=float))
        colors.append(_dist_color(meta["distance"]))
        faded.append(False)
    return box_strip_plot(
        labels,
        values,
        colors=colors,
        faded=faded,
        show_points=False,
        x_label=r"Spearman $\rho$",
        x_lim=(-1.05, 1.05),
        axvline=0.0,
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
        series[distance] = (
            np.asarray(xs, float),
            np.asarray(ys, float),
            _dist_color(distance),
        )
    return line_whisker_plot(
        series,
        x_label="number of clusters per run",
        y_label=r"Spearman $\rho$",
        log_x=True,
        y_lim=(-1.05, 1.05),
        vline=10,
        vline_label="unstable regime",
        hline=0.0,
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
            v
            for k, v in mean_imp.items()
            if k.startswith(prefix)
            and any(k[len(prefix) :].startswith(m) for m in members)
        )

    names = [name for name, _ in _FEATURE_FAMILIES]
    cluster = [part("cluster_", members) for _, members in _FEATURE_FAMILIES]
    klass = [part("class_", members) for _, members in _FEATURE_FAMILIES]
    return stacked_bar_plot(
        names,
        [("cluster-level", cluster, _COS), ("class-level", klass, _EUC)],
        x_label="mean importance (% of total)",
        total_format="{:.1f}%",
    )


def _fig_selective_stability(runs: list[dict]) -> Plot | None:
    """Figure: oracle benefit per classifier, predictor vs MCP, one panel per algorithm."""
    algos = ["kmeans", "spectral", "birch", "hdbscan"]
    cell: dict[tuple[str, str], dict[str, list[float]]] = {}
    net: dict[str, list[float]] = {}
    for r in runs:
        res = r["results"]
        if res.get("spearman") is None or not res.get("risk_coverage"):
            continue
        mcp = res.get("confidence_baselines", {}).get("mcp_risk")
        sig = res.get("significance", {}).get("vs_reference", {}).get("mcp_risk")
        if not mcp or not sig or r["algorithm"] not in algos:
            continue
        clf = _CLF_MERGE.get(r["clf"], r["clf"])
        c = cell.setdefault((clf, r["algorithm"]), {"pred": [], "mcp": []})
        c["pred"].append(100.0 * res["risk_coverage"]["oracle_benefit_recovered"])
        c["mcp"].append(100.0 * mcp["risk_coverage"]["oracle_benefit_recovered"])
        net.setdefault(clf, []).append(
            float(sig["lift_significant"] and sig["lift_diff"]["mean"] < 0)
            - float(sig["lift_significant"] and sig["lift_diff"]["mean"] > 0)
        )
    if not cell:
        return None

    order = sorted(net, key=lambda c: np.mean(net[c]), reverse=True)
    pred_c, mcp_c = _COS, _EUC
    panels = []
    for i, algo in enumerate(algos):
        labels, values, colors = [], [], []
        for clf in order:
            c = cell.get((clf, algo), {"pred": [], "mcp": []})
            labels.append(_CLF_LABEL.get(clf, clf) if i == 0 else "")
            values.append(np.asarray(c["pred"], dtype=float))
            colors.append(pred_c)
            labels.append("")
            values.append(np.asarray(c["mcp"], dtype=float))
            colors.append(mcp_c)
        panels.append((_ALGO_LABEL[algo], labels, values, colors))
    return box_strip_facets(
        panels,
        x_label="oracle benefit recovered (%)",
        x_lim=(-25.0, 105.0),
        axvline=0.0,
        legend={"predictor": pred_c, "MCP": mcp_c},
        figsize=(10.0, 5.5),
    )


def _fig_gain_by_algo(runs: list[dict]) -> Plot | None:
    """Figure: oracle benefit recovered by clustering algorithm, predictor vs MCP."""
    cell: dict[str, dict[str, list[float]]] = {}
    for r in runs:
        res = r["results"]
        mcp = res.get("confidence_baselines", {}).get("mcp_risk")
        if not res.get("risk_coverage") or not mcp or r["algorithm"] not in _ALGO_ORDER:
            continue
        c = cell.setdefault(r["algorithm"], {"pred": [], "mcp": []})
        c["pred"].append(100.0 * res["risk_coverage"]["oracle_benefit_recovered"])
        c["mcp"].append(100.0 * mcp["risk_coverage"]["oracle_benefit_recovered"])
    if not cell:
        return None
    labels, values, colors = [], [], []
    for algo in _ALGO_ORDER:
        c = cell.get(algo)
        if not c:
            continue
        labels.append(_ALGO_LABEL[algo])
        values.append(np.asarray(c["pred"], dtype=float))
        colors.append(_COS)
        labels.append("")
        values.append(np.asarray(c["mcp"], dtype=float))
        colors.append(_EUC)
    return box_strip_plot(
        labels,
        values,
        colors=colors,
        show_points=False,
        x_label="oracle benefit recovered (%)",
        x_lim=(-25.0, 105.0),
        axvline=0.0,
        legend={"predictor": _COS, "MCP": _EUC},
    )


_INSTANCE_METHODS = [
    ("mcp_sample", "MCP (per sample)"),
    ("mcp_cluster", "MCP (per cluster)"),
    ("atc_cluster", "ATC (per cluster)"),
    ("region", "Predictor (region)"),
    ("combo_rankavg", "Region+MCP (rank-avg)"),
    ("combo_within", "Region+MCP (within)"),
    ("combo_atc_rankavg", "Region+ATC (rank-avg)"),
]


def _load_instance(base: Path) -> dict | None:
    """Per-run instance-level baselines JSON, if the run produced one."""
    path = base / "instance_baselines.json"
    return load_from_json(path) if path.exists() else None


def _fig_instance_gain(runs: list[dict]) -> Plot | None:
    """Figure: instance-level oracle benefit recovered per score, over runs with a dump."""
    acc: dict[str, list[float]] = {k: [] for k, _ in _INSTANCE_METHODS}
    for r in runs:
        inst = _load_instance(r["base"])
        if not inst:
            continue
        for key, _ in _INSTANCE_METHODS:
            v = inst["scores"].get(key, {}).get("oracle_benefit_recovered")
            if v is not None and not np.isnan(v):
                acc[key].append(100.0 * v)
    if not any(acc.values()):
        return None
    palette = [PALETTE[i % len(PALETTE)] for i in (3, 4, 5, 0, 1, 2, 6)]
    labels, values, colors = [], [], []
    for (key, label), color in zip(_INSTANCE_METHODS, palette):
        labels.append(label)
        values.append(np.asarray(acc[key], dtype=float))
        colors.append(color)
    return box_strip_plot(
        labels,
        values,
        colors=colors,
        show_points=False,
        x_label="oracle benefit recovered (%)",
        x_lim=(-25.0, 105.0),
        axvline=0.0,
    )


def _std(values: np.ndarray) -> float:
    """Sample standard deviation, 0 for fewer than two values."""
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _table_perconfig(runs: list[dict]) -> dict:
    """Table: Spearman rho by clustering configuration (tab:perconfig)."""
    groups: dict[tuple[str, str], list[float]] = {}
    for r in runs:
        rho = r["results"].get("spearman")
        if rho is None:
            continue
        groups.setdefault((r["distance"], r["algorithm"]), []).append(rho)

    rows = []
    for (distance, algorithm), vals in sorted(groups.items()):
        arr = np.array(vals, dtype=float)
        rows.append(
            {
                "distance": distance,
                "algorithm": algorithm,
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": _std(arr),
                "pct_gt_0_7": 100.0 * float((arr > 0.7).mean()),
                "pct_lt_0": 100.0 * float((arr < 0.0).mean()),
                "n_runs": len(arr),
            }
        )
    return {"rows": rows}


def _table_nclusters(runs: list[dict]) -> dict:
    """Table: number of clusters per configuration (tab:nclusters)."""
    groups: dict[tuple[str, str], list[int]] = {}
    for r in runs:
        if r["results"].get("spearman") is None:
            continue
        n = r["results"].get("n_clusters_used")
        if n is None:
            continue
        groups.setdefault((r["distance"], r["algorithm"]), []).append(n)

    rows = []
    for (distance, algorithm), vals in sorted(groups.items()):
        arr = np.array(vals, dtype=float)
        rows.append(
            {
                "distance": distance,
                "algorithm": algorithm,
                "median": float(np.median(arr)),
                "min": int(arr.min()),
                "max": int(arr.max()),
                "n_runs": len(arr),
            }
        )
    return {"rows": rows}


def _table_perclf_perdataset(runs: list[dict]) -> dict:
    """Table: Spearman rho by classifier and by dataset (tab:perclf_perdataset)."""

    def _group(key: str) -> list[dict]:
        groups: dict[str, list[float]] = {}
        for r in runs:
            rho = r["results"].get("spearman")
            if rho is None:
                continue
            name = _CLF_MERGE.get(r[key], r[key]) if key == "clf" else r[key]
            groups.setdefault(name, []).append(rho)
        rows = []
        for name, vals in sorted(groups.items()):
            arr = np.array(vals, dtype=float)
            rows.append(
                {
                    key: name,
                    "mean": float(arr.mean()),
                    "median": float(np.median(arr)),
                    "std": _std(arr),
                    "min": float(arr.min()),
                    "n_runs": len(arr),
                }
            )
        return rows

    return {"by_classifier": _group("clf"), "by_dataset": _group("dataset")}


def _lift_stats(values: list[float]) -> dict:
    """Median lift (pp) and % of runs with positive lift, over a coverage-target value list."""
    arr = np.array(values, dtype=float)
    return {
        "lift_pp": 100.0 * float(np.median(arr)),
        "pos_pct": 100.0 * float((arr > 0).mean()),
    }


def _table_selective(runs: list[dict]) -> dict:
    """Table: selective-prediction lift by configuration, predictor vs confidence baselines."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in runs:
        if r["results"].get("spearman") is None:
            continue
        if not r["results"].get("risk_coverage"):
            continue
        groups.setdefault((r["distance"], r["algorithm"]), []).append(r["results"])

    rows = []
    for (distance, algorithm), results_list in sorted(groups.items()):
        lift = [res["risk_coverage"]["lift_over_random"] for res in results_list]
        oracle = np.array(
            [res["risk_coverage"]["oracle_benefit_recovered"] for res in results_list],
            dtype=float,
        )
        row = {
            "distance": distance,
            "algorithm": algorithm,
            **_lift_stats(lift),
            "oracle_pct": 100.0 * float(np.nanmedian(oracle)),
            "n_runs": len(results_list),
        }
        for name in ("mcp_risk", "margin_risk", "entropy_risk"):
            baselines = [
                res["confidence_baselines"][name]["risk_coverage"]
                for res in results_list
                if res.get("confidence_baselines", {}).get(name)
            ]
            if not baselines:
                continue
            stats = _lift_stats([b["lift_over_random"] for b in baselines])
            row[f"{name}_lift_pp"] = stats["lift_pp"]
            row[f"{name}_pos_pct"] = stats["pos_pct"]
            row[f"{name}_oracle_pct"] = 100.0 * float(
                np.nanmedian([b["oracle_benefit_recovered"] for b in baselines])
            )
            sig_vs = [
                res["significance"]["vs_reference"][name]
                for res in results_list
                if res.get("significance", {}).get("vs_reference", {}).get(name)
            ]
            if sig_vs:
                row[f"{name}_sig_better_pct"] = 100.0 * float(
                    np.mean(
                        [
                            s["lift_significant"] and s["lift_diff"]["mean"] < 0
                            for s in sig_vs
                        ]
                    )
                )
                row[f"{name}_sig_worse_pct"] = 100.0 * float(
                    np.mean(
                        [
                            s["lift_significant"] and s["lift_diff"]["mean"] > 0
                            for s in sig_vs
                        ]
                    )
                )
        rows.append(row)
    return {"rows": rows}


def _table_selective_by_clf(runs: list[dict]) -> dict:
    """Table: predictor vs MCP per classifier, ranked by net significant wins."""
    groups: dict[str, list[dict]] = {}
    for r in runs:
        res = r["results"]
        sig = res.get("significance", {}).get("vs_reference", {}).get("mcp_risk")
        mcp = res.get("confidence_baselines", {}).get("mcp_risk")
        if not sig or not mcp or not res.get("risk_coverage"):
            continue
        clf = _CLF_MERGE.get(r["clf"], r["clf"])
        groups.setdefault(clf, []).append(
            {
                "sig": sig,
                "pred_orc": res["risk_coverage"]["oracle_benefit_recovered"],
                "mcp_orc": mcp["risk_coverage"]["oracle_benefit_recovered"],
            }
        )

    rows = []
    for clf, items in groups.items():
        sigs = [it["sig"] for it in items]
        better = 100.0 * float(
            np.mean(
                [s["lift_significant"] and s["lift_diff"]["mean"] < 0 for s in sigs]
            )
        )
        worse = 100.0 * float(
            np.mean(
                [s["lift_significant"] and s["lift_diff"]["mean"] > 0 for s in sigs]
            )
        )
        rows.append(
            {
                "clf": clf,
                "pred_oracle_pct": 100.0
                * float(np.nanmedian([it["pred_orc"] for it in items])),
                "mcp_oracle_pct": 100.0
                * float(np.nanmedian([it["mcp_orc"] for it in items])),
                "sig_better_pct": better,
                "sig_worse_pct": worse,
                "net": better - worse,
                "n_runs": len(items),
            }
        )
    rows.sort(key=lambda row: row["net"], reverse=True)
    return {"rows": rows}


def _table_instance(runs: list[dict]) -> dict:
    """Table: instance-level oracle benefit per score, with a dataset-level bootstrap test."""
    rows = []
    for r in runs:
        inst = _load_instance(r["base"])
        if not inst:
            continue
        vals = {
            key: inst["scores"].get(key, {}).get("oracle_benefit_recovered")
            for key, _ in _INSTANCE_METHODS
        }
        core = ("mcp_sample", "region", "combo_rankavg")
        if any(vals[k] is None or np.isnan(vals[k]) for k in core):
            continue
        spear = {
            key: inst["scores"].get(key, {}).get("spearman")
            for key, _ in _INSTANCE_METHODS
        }
        sig = (
            inst.get("significance", {})
            .get("vs_reference", {})
            .get("combo_rankavg", {})
        )
        cal = inst.get("calibration", {})
        rows.append(
            {
                "dataset": _dataset_base(r["dataset"]),
                "sig_combo": bool(sig.get("significant"))
                and sig.get("oracle_diff", {}).get("mean", 0.0) > 0,
                "cluster_rate_mse": cal.get("cluster_rate_mse", {}),
                "atc_accuracy_abs_error": cal.get("atc_accuracy_abs_error"),
                "spearman": spear,
                **vals,
            }
        )
    if not rows:
        return {"n_runs": 0}

    median = {
        key: float(np.median([row[key] for row in rows if row.get(key) is not None]))
        for key, _ in _INSTANCE_METHODS
        if any(row.get(key) is not None for row in rows)
    }
    median_spearman = {
        key: float(
            np.nanmedian(
                [
                    row["spearman"][key]
                    for row in rows
                    if row["spearman"].get(key) is not None
                ]
            )
        )
        for key, _ in _INSTANCE_METHODS
        if any(row["spearman"].get(key) is not None for row in rows)
    }
    mse_keys = ("region", "mcp_cluster", "atc_cluster")
    median_rate_mse = {
        k: float(
            np.median(
                [
                    row["cluster_rate_mse"][k]
                    for row in rows
                    if k in row["cluster_rate_mse"]
                ]
            )
        )
        for k in mse_keys
        if any(k in row["cluster_rate_mse"] for row in rows)
    }
    atc_errs = [
        row["atc_accuracy_abs_error"]
        for row in rows
        if row["atc_accuracy_abs_error"] is not None
    ]
    per_ds = {}
    for d in sorted({row["dataset"] for row in rows}):
        deltas = [
            row["combo_rankavg"] - row["mcp_sample"]
            for row in rows
            if row["dataset"] == d
        ]
        per_ds[d] = float(np.mean(deltas))
    ds_vals = np.array(list(per_ds.values()), dtype=float)
    rng = np.random.default_rng(0)
    boot = np.array(
        [
            float(np.mean(rng.choice(ds_vals, size=ds_vals.size, replace=True)))
            for _ in range(2000)
        ]
    )
    ci_low, ci_high = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
    return {
        "n_runs": len(rows),
        "median_oracle_benefit_recovered": median,
        "median_spearman": median_spearman,
        "median_cluster_rate_mse": median_rate_mse,
        "median_atc_accuracy_abs_error": (
            float(np.median(atc_errs)) if atc_errs else None
        ),
        "combo_rankavg_vs_mcp_sample": {
            "per_run_sig_better_pct": 100.0
            * float(np.mean([row["sig_combo"] for row in rows])),
            "mean_delta_over_datasets": float(np.mean(ds_vals)),
            "dataset_bootstrap_ci": [ci_low, ci_high],
            "significant": bool(not (ci_low <= 0 <= ci_high)),
        },
    }


def _render_sweep_results(
    root: Path, fmt: str = "pdf", out: Path | None = None
) -> None:
    """Aggregate the sweep under `root` into the cross-run figures and result tables."""
    set_figure_format(fmt)
    runs = _load_sweep_runs(root)
    if not runs:
        logger.warning("No sweep runs found under %s; nothing to render.", root)
        return
    n_hdb = sum(1 for r in runs if r["algorithm"] == "hdbscan")
    logger.info(
        "Sweep results from %d runs (%d main, %d hdbscan) under %s",
        len(runs),
        len(runs) - n_hdb,
        n_hdb,
        root,
    )

    figures = {
        "figure/rho_by_config": _fig_rho_by_config(runs),
        "figure/rho_vs_clusters": _fig_rho_vs_clusters(runs),
        "figure/family_importance": _fig_family_importance(runs),
        "figure/gain_by_algo": _fig_gain_by_algo(runs),
        "figure/selective_stability": _fig_selective_stability(runs),
        "figure/instance_gain": _fig_instance_gain(runs),
    }
    figures = {k: v for k, v in figures.items() if v is not None}

    tables = {
        "json/perconfig_table": _table_perconfig(runs),
        "json/nclusters_table": _table_nclusters(runs),
        "json/perclf_perdataset_table": _table_perclf_perdataset(runs),
        "json/selective_table": _table_selective(runs),
        "json/selective_by_clf_table": _table_selective_by_clf(runs),
        "json/instance_table": _table_instance(runs),
    }

    figures_base = out or root
    bus = LogDispatcher()
    bus.subscribe(FilesystemFigureSubscriber(figures_base))
    bus.subscribe(JSONSubscriber(root))
    bus.publish(LogBundle.from_dict({**figures, **tables}))
    logger.info(
        "Sweep figures (%s) -> %s",
        ", ".join(sorted(k.split("/")[-1] for k in figures)),
        figures_base,
    )
    logger.info(
        "Sweep tables (%s) -> %s",
        ", ".join(sorted(k.split("/")[-1] for k in tables)),
        root,
    )


def _parse_args(argv: list[str]) -> tuple[Path, str, Path | None]:
    """Parse `sweep=<path> [format=..] [out=..]` from argv."""
    kv = dict(a.split("=", 1) for a in argv if "=" in a)
    if "sweep" not in kv:
        raise ValueError("sweep_results requires sweep=<path>.")
    return (
        Path(kv["sweep"]),
        kv.get("format", "pdf"),
        Path(kv["out"]) if kv.get("out") else None,
    )


def main() -> None:
    """Entry point for the sweep aggregation stage."""
    root, fmt, out = _parse_args(sys.argv[1:])
    _render_sweep_results(root, fmt=fmt, out=out)


if __name__ == "__main__":
    main()
