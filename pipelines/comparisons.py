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
from src.domain.plot.comparison_charts import (
    box_strip_plot,
    grouped_bar_plot,
    line_whisker_plot,
    stacked_bar_plot,
)
from src.domain.plot.style import PALETTE, apply_plot_style

setup_logger()
apply_plot_style()
logger = logging.getLogger(__name__)

_ALGO_ORDER = ["kmeans", "spectral", "birch", "hdbscan"]
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
# The 5 unified baseline variants from `instance_baselines` (see failure_regressor.py),
# in the order the paper reports them.
_VARIANT_ORDER = [
    "mcp_cluster",
    "atc_cluster",
    "region",
    "combo_rankavg",
    "combo_atc_rankavg",
]
_VARIANT_LABEL = {
    "mcp_cluster": "MCP",
    "atc_cluster": "ATC",
    "region": "regressor",
    "combo_rankavg": "regressor + MCP",
    "combo_atc_rankavg": "regressor + ATC",
}
_VARIANT_COLOR = {
    "mcp_cluster": PALETTE[0],
    "atc_cluster": PALETTE[1],
    "region": PALETTE[2],
    "combo_rankavg": PALETTE[3],
    "combo_atc_rankavg": PALETTE[4],
}
# The 3 variants on the failure-rate scale (the only ones with a cluster_rate_mse).
_RATE_VARIANT_ORDER = ["mcp_cluster", "atc_cluster", "region"]
# Fig 10's top-to-bottom row order.
_ORACLE_BENEFIT_ORDER = [
    "combo_atc_rankavg",
    "combo_rankavg",
    "region",
    "atc_cluster",
    "mcp_cluster",
]


def _load_instance(base: Path) -> dict | None:
    """Per-run instance-level baselines JSON, if the run produced one."""
    path = base / "instance_baselines.json"
    return load_from_json(path) if path.exists() else None


def _load_sweep_runs(root: Path) -> list[dict]:
    """Collect one record per `<config>/<dataset>/<classifier>` run under `root`.

    `clf` is the run directory name, taken verbatim: runs saved under a name a classifier
    no longer uses are reported as a classifier of their own. Re-run them before comparing.
    """
    runs: list[dict] = []
    for cfg_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for ds_dir in sorted(p for p in cfg_dir.iterdir() if p.is_dir()):
            cfg_path = ds_dir / "shared/config_composed.json"
            if not cfg_path.exists():
                continue
            composed = load_from_json(cfg_path)
            algorithm = next(iter(composed["clustering"]["algorithms"]), None)
            dataset_size = _dataset_size(ds_dir / "shared/metadata/df_info.json")
            data_cfg = composed.get("data", {}) or {}
            n_features = len(data_cfg.get("num_cols") or []) + len(
                data_cfg.get("cat_cols") or []
            )
            for clf_dir in sorted(
                p for p in ds_dir.iterdir() if p.is_dir() and p.name != "shared"
            ):
                base = clf_dir / "outputs/analysis"
                results_path = base / "failure_regressor_results.json"
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
                        "n_features": n_features,
                        "ds_dir": ds_dir,
                        "base": base,
                        "results": load_from_json(results_path),
                        "instance": _load_instance(base),
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


def _is_kmeans_euclidean(run: dict) -> bool:
    """True for the k-means + Euclidean configuration the paper restricts RQ2-RQ5 to."""
    return run["algorithm"] == "kmeans" and run["distance"] == "euclidean"


def _median_iqr(values: list[float]) -> dict:
    """Median and interquartile range (p25, p75) of a value list; None fields if empty."""
    arr = np.array([v for v in values if v is not None], dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"median": None, "p25": None, "p75": None, "n": 0}
    return {
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "n": int(arr.size),
    }


def _variant_value(run: dict, variant: str, field: str) -> float | None:
    """One scalar field of one baseline variant from a run's instance-baselines table, if valid."""
    inst = run.get("instance")
    if not inst:
        return None
    entry = next(
        (r for r in inst.get("baselines", []) if r["variant"] == variant), None
    )
    if not entry:
        return None
    v = entry.get(field)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return float(v)


def _variant_series(
    runs: list[dict],
    groups: list[str],
    group_key,
    field: str,
    variants: list[str] = _VARIANT_ORDER,
) -> list[tuple[str, list[float], list[float], list[float], str]]:
    """Median (+ IQR error bars) of `field`, per variant, aggregated within each group."""
    series = []
    for variant in variants:
        medians, err_lo, err_hi = [], [], []
        for group in groups:
            vals = [
                v
                for r in runs
                if group_key(r) == group
                for v in [_variant_value(r, variant, field)]
                if v is not None
            ]
            stats = _median_iqr(vals)
            m = stats["median"] if stats["median"] is not None else 0.0
            medians.append(m)
            err_lo.append(m - stats["p25"] if stats["p25"] is not None else 0.0)
            err_hi.append(stats["p75"] - m if stats["p75"] is not None else 0.0)
        series.append(
            (_VARIANT_LABEL[variant], medians, err_lo, err_hi, _VARIANT_COLOR[variant])
        )
    return series


def _fig_rho_by_config(runs: list[dict]) -> Plot | None:
    """Figure 7a: Spearman rho distribution per clustering configuration (full sweep)."""

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
    """Figure 7b: Spearman rho against the number of clusters per run (full sweep)."""
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
    """Figure 11: feature-family importance, cluster- vs class-level (full sweep)."""
    acc: dict[str, list[float]] = {}
    for r in runs:
        for row in r["results"].get("feature_importances", []):
            acc.setdefault(row["feature"], []).append(row["importance"])
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


def _fig_spearman_by_classifier(runs: list[dict]) -> Plot | None:
    """Figure 8: cluster-level Spearman rho (median, IQR) of the 5 variants, per classifier
    (kmeans-euclidean subset, 90 runs)."""
    subset = [r for r in runs if _is_kmeans_euclidean(r)]
    clfs = sorted({r["clf"] for r in subset})
    if not clfs:
        return None
    series = _variant_series(subset, clfs, lambda r: r["clf"], "spearman")
    return grouped_bar_plot(
        [_CLF_LABEL.get(c, c) for c in clfs],
        series,
        y_label=r"Spearman $\rho$ (median, IQR)",
        y_lim=(-0.05, 1.05),
    )


def _fig_mse_by_classifier(runs: list[dict]) -> Plot | None:
    """Figure 9: cluster-rate MSE (median, IQR) of the 3 rate variants, per classifier
    (kmeans-euclidean subset, 90 runs)."""
    subset = [r for r in runs if _is_kmeans_euclidean(r)]
    clfs = sorted({r["clf"] for r in subset})
    if not clfs:
        return None
    series = _variant_series(
        subset, clfs, lambda r: r["clf"], "cluster_rate_mse", _RATE_VARIANT_ORDER
    )
    return grouped_bar_plot(
        [_CLF_LABEL.get(c, c) for c in clfs],
        series,
        y_label="MSE (median, IQR)",
    )


def _fig_spearman_by_dataset(runs: list[dict]) -> Plot | None:
    """Figure 13: cluster-level Spearman rho (median, IQR) of the 5 variants, per dataset
    (kmeans-euclidean subset)."""
    subset = [r for r in runs if _is_kmeans_euclidean(r)]
    datasets = sorted({_dataset_base(r["dataset"]) for r in subset})
    if not datasets:
        return None
    series = _variant_series(
        subset, datasets, lambda r: _dataset_base(r["dataset"]), "spearman"
    )
    return grouped_bar_plot(
        [_DATASET_LABEL.get(d, d) for d in datasets],
        series,
        y_label=r"Spearman $\rho$ (median, IQR)",
        y_lim=(-0.05, 1.05),
    )


def _fig_oracle_benefit_by_variant(runs: list[dict]) -> Plot | None:
    """Figure 10: per-sample oracle benefit recovered (%) for the 5 baseline variants
    (kmeans-euclidean subset)."""
    subset = [r for r in runs if _is_kmeans_euclidean(r)]
    acc: dict[str, list[float]] = {v: [] for v in _ORACLE_BENEFIT_ORDER}
    for r in subset:
        for variant in _ORACLE_BENEFIT_ORDER:
            v = _variant_value(r, variant, "oracle_benefit_recovered")
            if v is not None:
                acc[variant].append(100.0 * v)
    if not any(acc.values()):
        return None
    labels = [_VARIANT_LABEL[v] for v in _ORACLE_BENEFIT_ORDER]
    values = [np.asarray(acc[v], dtype=float) for v in _ORACLE_BENEFIT_ORDER]
    colors = [_VARIANT_COLOR[v] for v in _ORACLE_BENEFIT_ORDER]
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
    """Table 4: number of clusters per configuration, sorted by median within each distance."""
    groups: dict[tuple[str, str], list[int]] = {}
    for r in runs:
        if r["results"].get("spearman") is None:
            continue
        n = r["results"].get("n_clusters_used")
        if n is None:
            continue
        groups.setdefault((r["distance"], r["algorithm"]), []).append(n)

    rows = []
    for distance in sorted({d for d, _ in groups}):
        algos = [algo for (d, algo) in groups if d == distance]
        algos.sort(key=lambda algo: -float(np.median(groups[(distance, algo)])))
        for algorithm in algos:
            arr = np.array(groups[(distance, algorithm)], dtype=float)
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


def _table_datasets(runs: list[dict]) -> dict:
    """Table 3: per-dataset instance/feature/class/imbalance counts from the raw, pre-filter
    df_info.json (not the post-`rare_category_filter` metadata)."""
    seen: dict[str, dict] = {}
    for r in runs:
        ds = _dataset_base(r["dataset"])
        if ds in seen:
            continue
        info_path = r["ds_dir"] / "shared/metadata/df_info.json"
        if not info_path.exists():
            continue
        info = load_from_json(info_path)
        label_dist = info.get("label_distribution") or {}
        counts = sorted(label_dist.values(), reverse=True)
        seen[ds] = {
            "dataset": ds,
            "n_instances": info["shape"][0],
            "n_features": r.get("n_features"),
            "n_classes": len(label_dist),
            "imbalance_ratio": (
                counts[0] / counts[-1] if len(counts) >= 2 and counts[-1] else None
            ),
        }
    return {"rows": [seen[ds] for ds in sorted(seen)]}


def _table_variant_spearman(runs: list[dict]) -> dict:
    """Table 5: cluster-level Spearman rho for the 5 baseline variants (kmeans-euclidean, 90 runs)."""
    subset = [r for r in runs if _is_kmeans_euclidean(r)]
    rows = [
        {
            "variant": variant,
            **_median_iqr([_variant_value(r, variant, "spearman") for r in subset]),
        }
        for variant in _VARIANT_ORDER
    ]
    return {"n_runs": len(subset), "rows": rows}


def _table_variant_cluster_mse(runs: list[dict]) -> dict:
    """Table 6: cluster-rate MSE for the 3 rate-scale variants (kmeans-euclidean, 90 runs)."""
    subset = [r for r in runs if _is_kmeans_euclidean(r)]
    rows = [
        {
            "variant": variant,
            **_median_iqr(
                [_variant_value(r, variant, "cluster_rate_mse") for r in subset]
            ),
        }
        for variant in _RATE_VARIANT_ORDER
    ]
    return {"n_runs": len(subset), "rows": rows}


def _table_variant_spearman_by_dataset(runs: list[dict]) -> dict:
    """Table 7: per-dataset median Spearman rho for the 5 variants (kmeans-euclidean, across
    the 10 classifiers)."""
    subset = [r for r in runs if _is_kmeans_euclidean(r)]
    datasets = sorted({_dataset_base(r["dataset"]) for r in subset})
    rows = []
    for ds in datasets:
        ds_runs = [r for r in subset if _dataset_base(r["dataset"]) == ds]
        row: dict = {"dataset": ds}
        for variant in _VARIANT_ORDER:
            vals = [
                v
                for r in ds_runs
                for v in [_variant_value(r, variant, "spearman")]
                if v is not None
            ]
            row[variant] = float(np.median(vals)) if vals else None
        rows.append(row)
    return {"rows": rows}


def _render_comparisons(root: Path, fmt: str = "pdf", out: Path | None = None) -> None:
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
        "figure/spearman_by_classifier": _fig_spearman_by_classifier(runs),
        "figure/mse_by_classifier": _fig_mse_by_classifier(runs),
        "figure/oracle_benefit_by_variant": _fig_oracle_benefit_by_variant(runs),
        "figure/spearman_by_dataset": _fig_spearman_by_dataset(runs),
    }
    figures = {k: v for k, v in figures.items() if v is not None}

    tables = {
        "json/perconfig_table": _table_perconfig(runs),
        "json/nclusters_table": _table_nclusters(runs),
        "json/datasets_table": _table_datasets(runs),
        "json/variant_spearman_table": _table_variant_spearman(runs),
        "json/variant_cluster_mse_table": _table_variant_cluster_mse(runs),
        "json/variant_spearman_by_dataset_table": _table_variant_spearman_by_dataset(
            runs
        ),
    }

    figures_base = out or root
    bus = LogDispatcher()
    bus.subscribe(FilesystemFigureSubscriber(figures_base))
    bus.subscribe(JSONSubscriber(root))
    bus.publish(LogBundle.from_dict({**figures, **tables}))
    logger.info(
        "Comparison figures (%s) -> %s",
        ", ".join(sorted(k.split("/")[-1] for k in figures)),
        figures_base,
    )
    logger.info(
        "Comparison tables (%s) -> %s",
        ", ".join(sorted(k.split("/")[-1] for k in tables)),
        root,
    )


def _parse_args(argv: list[str]) -> tuple[Path, str, Path | None]:
    """Parse `sweep=<path> [format=..] [out=..]` from argv."""
    kv = dict(a.split("=", 1) for a in argv if "=" in a)
    if "sweep" not in kv:
        raise ValueError("comparisons requires sweep=<path>.")
    return (
        Path(kv["sweep"]),
        kv.get("format", "pdf"),
        Path(kv["out"]) if kv.get("out") else None,
    )


def main() -> None:
    """Entry point for the cross-run comparisons stage."""
    root, fmt, out = _parse_args(sys.argv[1:])
    _render_comparisons(root, fmt=fmt, out=out)


if __name__ == "__main__":
    main()
