import logging
import sys
from pathlib import Path

import numpy as np

from src.core.io import save_figures
from src.core.log import setup_logger
from src.core.paths import DATASET_STAGES, RunPaths
from src.core.record import clear_dir, read_record
from src.core.utils import load_from_json, save_to_json
from src.domain.analysis.failure_regressor import (
    BASELINE_VARIANTS,
    RATE_BASELINE_VARIANTS,
)
from src.domain.plot.base import Plot, set_figure_format
from src.domain.plot.comparison_charts import (
    box_plot,
    grouped_bar_plot,
    line_whisker_plot,
    stacked_bar_plot,
)
from src.domain.plot.style import (
    BASELINE_COLOR,
    BASELINE_LABEL,
    NOISE_COLOR,
    PALETTE,
    apply_plot_style,
)

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
        "region-geometry",
        (
            "max_dispersion",
            "p95_dispersion",
            "dist_to_nearest_rival",
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
_VARIANT_ORDER = list(BASELINE_VARIANTS)
_RATE_VARIANT_ORDER = [v for v in _VARIANT_ORDER if v in RATE_BASELINE_VARIANTS]
# Top-to-bottom row order of the oracle-benefit figure.
_ORACLE_BENEFIT_ORDER = _VARIANT_ORDER[::-1]


def _load_sweep_runs(root: Path) -> list[dict]:
    """Collect one record per `<config>/<dataset>/<classifier>` run under `root`.

    `clf` is the run directory name, taken verbatim: runs saved under a name a classifier
    no longer uses are reported as a classifier of their own.
    """
    runs: list[dict] = []
    for cfg_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for ds_dir in sorted(p for p in cfg_dir.iterdir() if p.is_dir()):
            if not (ds_dir / "regions/record.json").exists():
                continue
            regions_config = read_record(ds_dir / "regions")["config"]
            meta = load_from_json(ds_dir / "split/meta.json")
            algorithm = next(iter(regions_config["clustering"]["algorithms"]), None)
            for clf_dir in sorted(
                p
                for p in ds_dir.iterdir()
                if p.is_dir() and p.name not in DATASET_STAGES
            ):
                base = clf_dir / "regress"
                results_path = base / "results.json"
                if not results_path.exists():
                    continue
                # A run is described by what it was built from, not by what the folders
                # beside it hold now.
                run_paths = RunPaths(dataset=ds_dir, classifier=clf_dir)
                for source, source_id in read_record(base)["inputs"].items():
                    if read_record(run_paths.of(source))["id"] != source_id:
                        raise ValueError(
                            f"{base} was built from another {source} than the one on "
                            "disk: re-run `make regress`."
                        )
                # Absent when the failure regressor skipped a degenerate target.
                baselines_path = base / "baselines.json"
                instance = (
                    load_from_json(baselines_path) if baselines_path.exists() else None
                )
                if instance is not None:
                    variants = {r["variant"] for r in instance["baselines"]}
                    if variants != set(_VARIANT_ORDER):
                        raise ValueError(
                            f"{baselines_path} has variants {sorted(variants)}; "
                            f"compare displays {_VARIANT_ORDER}."
                        )
                runs.append(
                    {
                        "config": cfg_dir.name,
                        "dataset": ds_dir.name,
                        "clf": clf_dir.name,
                        "distance": regions_config["distance"],
                        "algorithm": algorithm,
                        "n_features": len(meta["num_cols"]) + len(meta["cat_cols"]),
                        "meta": meta,
                        "results": load_from_json(results_path),
                        "instance": instance,
                    }
                )
    return runs


def _dataset_base(name: str) -> str:
    """Strip a trailing `_<seed>` suffix from a run's dataset directory name."""
    head, _, tail = name.rpartition("_")
    return head if tail.isdigit() else name


def _dist_color(distance: str) -> str:
    return _COS if distance == "cosine" else _EUC


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
    if run["instance"] is None:
        return None
    entry = next(r for r in run["instance"]["baselines"] if r["variant"] == variant)
    v = entry[field]
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return float(v)


def _bar_series(
    name: str, values_by_group: list[list[float | None]], color: str
) -> tuple[str, list[float], list[float], list[float], str]:
    """Median (+ IQR error bars) of each group's values; NaN, so no bar, for an empty group."""
    medians, err_lo, err_hi = [], [], []
    for values in values_by_group:
        stats = _median_iqr(values)
        m = stats["median"] if stats["median"] is not None else np.nan
        medians.append(m)
        err_lo.append(m - stats["p25"] if stats["p25"] is not None else 0.0)
        err_hi.append(stats["p75"] - m if stats["p75"] is not None else 0.0)
    return name, medians, err_lo, err_hi, color


def _variant_series(
    runs: list[dict],
    *,
    groups: list[str],
    group_key,
    field: str,
    variants: list[str],
) -> list[tuple[str, list[float], list[float], list[float], str]]:
    """Median (+ IQR error bars) of `field`, per variant, aggregated within each group."""
    return [
        _bar_series(
            BASELINE_LABEL[variant],
            [
                [
                    _variant_value(r, variant, field)
                    for r in runs
                    if group_key(r) == group
                ]
                for group in groups
            ],
            BASELINE_COLOR[variant],
        )
        for variant in variants
    ]


def _fig_rho_by_config(runs: list[dict]) -> Plot | None:
    """Spearman rho distribution per clustering configuration."""

    def sort_key(cfg: str) -> tuple[int, int]:
        meta = next(r for r in runs if r["config"] == cfg)
        algo = meta["algorithm"]
        return (
            0 if meta["distance"] == "cosine" else 1,
            _ALGO_ORDER.index(algo) if algo in _ALGO_ORDER else 99,
        )

    labels, values, colors = [], [], []
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
    return box_plot(
        labels,
        values,
        colors=colors,
        x_label=r"Spearman $\rho$",
        x_lim=(-1.05, 1.05),
        axvline=0.0,
        legend={"cosine": _COS, "euclidean": _EUC},
    )


def _fig_rho_vs_regions(runs: list[dict]) -> Plot | None:
    """Spearman rho against the number of regions per run."""
    series: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    for distance in ("cosine", "euclidean"):
        xs, ys = [], []
        for r in runs:
            if r["distance"] != distance:
                continue
            rho = r["results"].get("spearman")
            if rho is not None:
                xs.append(r["results"]["n_regions_used"])
                ys.append(rho)
        series[distance] = (
            np.asarray(xs, float),
            np.asarray(ys, float),
            _dist_color(distance),
        )
    return line_whisker_plot(
        series,
        x_label="number of regions per run",
        y_label=r"Spearman $\rho$",
        log_x=True,
        y_lim=(-1.05, 1.05),
        vline=10,
        vline_label="unstable regime",
        hline=0.0,
    )


def _fig_family_importance(runs: list[dict]) -> Plot | None:
    """Feature-family importance, region- vs class-level."""
    acc: dict[str, list[float]] = {}
    for r in runs:
        for row in r["results"].get("feature_importances", []):
            acc.setdefault(row["feature"], []).append(row["importance"])
    if not acc:
        return None
    mean_imp = {k: float(np.mean(v)) for k, v in acc.items()}

    def in_family(key: str, prefix: str, members: tuple[str, ...]) -> bool:
        return key.startswith(prefix) and any(
            key[len(prefix) :].startswith(m) for m in members
        )

    def part(prefix: str, members: tuple[str, ...]) -> float:
        return 100.0 * sum(
            v for k, v in mean_imp.items() if in_family(k, prefix, members)
        )

    for prefix in ("region_", "class_"):
        unmatched = sorted(
            k
            for k in mean_imp
            if k.startswith(prefix)
            and not any(
                in_family(k, prefix, members) for _, members in _FEATURE_FAMILIES
            )
        )
        if unmatched:
            raise ValueError(
                f"feature_importances has {prefix}-scoped keys _FEATURE_FAMILIES "
                f"does not list: {unmatched}. A run predates a rename: re-run it."
            )

    names = [name for name, _ in _FEATURE_FAMILIES]
    region = [part("region_", members) for _, members in _FEATURE_FAMILIES]
    klass = [part("class_", members) for _, members in _FEATURE_FAMILIES]
    return stacked_bar_plot(
        names,
        [("region-level", region, _COS), ("class-level", klass, _EUC)],
        x_label="mean importance (% of total)",
        total_format="{:.1f}%",
    )


def _fig_variant_by_group(
    runs: list[dict],
    *,
    group_key,
    label_map: dict[str, str],
    field: str,
    y_label: str,
    variants: list[str] = _VARIANT_ORDER,
    y_lim: tuple[float, float] | None = None,
    log_y: bool = False,
) -> Plot | None:
    """Grouped bar of each variant's median (+IQR) `field`, grouped by `group_key`."""
    groups = sorted({group_key(r) for r in runs})
    if not groups:
        return None
    series = _variant_series(
        runs, groups=groups, group_key=group_key, field=field, variants=variants
    )
    return grouped_bar_plot(
        [label_map.get(g, g) for g in groups],
        series,
        y_label=y_label,
        y_lim=y_lim,
        log_y=log_y,
    )


def _fig_oracle_benefit_by_variant(runs: list[dict]) -> Plot | None:
    """Per-sample oracle benefit recovered (%) of the variants."""
    acc: dict[str, list[float]] = {v: [] for v in _ORACLE_BENEFIT_ORDER}
    for r in runs:
        for variant in _ORACLE_BENEFIT_ORDER:
            v = _variant_value(r, variant, "oracle_benefit_recovered")
            if v is not None:
                acc[variant].append(100.0 * v)
    if not any(acc.values()):
        return None
    labels = [BASELINE_LABEL[v] for v in _ORACLE_BENEFIT_ORDER]
    values = [np.asarray(acc[v], dtype=float) for v in _ORACLE_BENEFIT_ORDER]
    colors = [BASELINE_COLOR[v] for v in _ORACLE_BENEFIT_ORDER]
    return box_plot(
        labels,
        values,
        colors=colors,
        figsize=(5.4, 0.6 * len(labels)),
        x_label="oracle benefit recovered (%)",
        x_lim=(-25.0, 105.0),
        axvline=0.0,
    )


# The predictions the by-size figures compare, in the order of their bars.
_SIZE_VARIANTS = ("region", "train_rate_region", "val_rate_region")


def _size_values(runs: list[dict], size_bin: int, variant: str, field: str) -> list:
    """`field` of one variant in one size bin, from every run that has it."""
    return [
        row[field]
        for r in runs
        if r["instance"] is not None
        for row in r["instance"]["error_by_size"]
        if row["size_bin"] == size_bin and row["variant"] == variant
    ]


def _size_bins(runs: list[dict]) -> list[int]:
    return sorted(
        {
            row["size_bin"]
            for r in runs
            if r["instance"] is not None
            for row in r["instance"]["error_by_size"]
        }
    )


def _fig_error_by_size(
    runs: list[dict], *, field: str, y_label: str, noise: bool = False
) -> Plot | None:
    """Median (+IQR) of `field` per bin of region size, one bar per prediction."""
    bins = _size_bins(runs)
    if not bins:
        return None
    series = [
        _bar_series(
            BASELINE_LABEL[variant],
            [_size_values(runs, b, variant, field) for b in bins],
            BASELINE_COLOR[variant],
        )
        for variant in _SIZE_VARIANTS
    ]
    if noise:
        series.append(
            (
                "test noise",
                _bar_series(
                    "",
                    [_size_values(runs, b, "region", "test_noise") for b in bins],
                    "",
                )[1],
                None,
                None,
                NOISE_COLOR,
            )
        )
    labels = [f"Q{b + 1}" for b in bins]
    labels[0] += " (smallest)"
    labels[-1] += " (largest)"
    return grouped_bar_plot(
        labels,
        series,
        x_label="region size (train rows), quantile",
        y_label=y_label,
        hline=None if noise else 0.0,
        log_y=noise,
    )


def _table_error_by_size(runs: list[dict]) -> dict:
    """Median (+IQR) squared and signed error per bin of region size and prediction."""
    rows = []
    for b in _size_bins(runs):
        for variant in _SIZE_VARIANTS:
            mse = _median_iqr(_size_values(runs, b, variant, "mse"))
            bias = _median_iqr(_size_values(runs, b, variant, "bias"))
            noise = _median_iqr(_size_values(runs, b, variant, "test_noise"))
            rows.append(
                {
                    "size_bin": b,
                    "variant": variant,
                    "mse_median": mse["median"],
                    "mse_p25": mse["p25"],
                    "mse_p75": mse["p75"],
                    "bias_median": bias["median"],
                    "bias_p25": bias["p25"],
                    "bias_p75": bias["p75"],
                    "test_noise_median": noise["median"],
                    "n_runs": mse["n"],
                }
            )
    return {"rows": rows}


def _table_perconfig(runs: list[dict]) -> dict:
    """Spearman rho by clustering configuration."""
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
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "pct_gt_0_7": 100.0 * float((arr > 0.7).mean()),
                "pct_lt_0": 100.0 * float((arr < 0.0).mean()),
                "n_runs": len(arr),
            }
        )
    return {"rows": rows}


def _table_nregions(runs: list[dict]) -> dict:
    """Number of regions per configuration, sorted by median within each distance."""
    groups: dict[tuple[str, str], list[int]] = {}
    for r in runs:
        if r["results"].get("spearman") is None:
            continue
        groups.setdefault((r["distance"], r["algorithm"]), []).append(
            r["results"]["n_regions_used"]
        )

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
    """Per-dataset row and class counts and imbalance, from the raw class counts."""
    seen: dict[str, dict] = {}
    for r in runs:
        ds = _dataset_base(r["dataset"])
        if ds in seen:
            continue
        meta = r["meta"]
        counts = sorted((c["n_rows"] for c in meta["raw_classes"]), reverse=True)
        seen[ds] = {
            "dataset": ds,
            "n_rows": meta["n_raw_rows"],
            "n_features": r["n_features"],
            "n_classes": len(counts),
            "imbalance_ratio": (
                counts[0] / counts[-1] if len(counts) >= 2 and counts[-1] else None
            ),
        }
    return {"rows": [seen[ds] for ds in sorted(seen)]}


def _table_variant_field(
    runs: list[dict], *, field: str, variants: list[str] = _VARIANT_ORDER
) -> dict:
    """Median (+IQR) of `field` per baseline variant."""
    rows = [
        {
            "variant": variant,
            **_median_iqr([_variant_value(r, variant, field) for r in runs]),
        }
        for variant in variants
    ]
    return {"n_runs": len(runs), "rows": rows}


def _table_variant_spearman_by_dataset(runs: list[dict]) -> dict:
    """Per-dataset median Spearman rho of the variants."""
    datasets = sorted({_dataset_base(r["dataset"]) for r in runs})
    rows = []
    for ds in datasets:
        ds_runs = [r for r in runs if _dataset_base(r["dataset"]) == ds]
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


def _render_comparisons(root: Path, *, fmt: str, out: Path | None) -> None:
    """Aggregate the sweep under `root` into the cross-run figures and result tables."""
    set_figure_format(fmt)
    runs = _load_sweep_runs(root)
    if not runs:
        raise FileNotFoundError(f"No sweep runs found under {root}.")
    n_hdb = sum(1 for r in runs if r["algorithm"] == "hdbscan")
    logger.info(
        "Sweep results from %d runs (%d main, %d hdbscan) under %s",
        len(runs),
        len(runs) - n_hdb,
        n_hdb,
        root,
    )
    kmeans_euclidean = [
        r for r in runs if r["algorithm"] == "kmeans" and r["distance"] == "euclidean"
    ]

    figures = {
        "rho_by_config": _fig_rho_by_config(runs),
        "rho_vs_regions": _fig_rho_vs_regions(runs),
        "family_importance": _fig_family_importance(runs),
        "spearman_by_classifier": _fig_variant_by_group(
            kmeans_euclidean,
            group_key=lambda r: r["clf"],
            label_map=_CLF_LABEL,
            field="spearman",
            y_label=r"Spearman $\rho$ (median, IQR)",
            y_lim=(-0.05, 1.05),
        ),
        "mse_by_classifier": _fig_variant_by_group(
            kmeans_euclidean,
            group_key=lambda r: r["clf"],
            label_map=_CLF_LABEL,
            field="region_rate_mse",
            y_label="MSE (median, IQR)",
            variants=_RATE_VARIANT_ORDER,
            log_y=True,
        ),
        "oracle_benefit_by_variant": _fig_oracle_benefit_by_variant(kmeans_euclidean),
        "mse_by_region_size": _fig_error_by_size(
            kmeans_euclidean,
            field="mse",
            y_label="MSE (median, IQR)",
            noise=True,
        ),
        "bias_by_region_size": _fig_error_by_size(
            kmeans_euclidean,
            field="bias",
            y_label="predicted − observed rate (median, IQR)",
        ),
        "spearman_by_dataset": _fig_variant_by_group(
            kmeans_euclidean,
            group_key=lambda r: _dataset_base(r["dataset"]),
            label_map=_DATASET_LABEL,
            field="spearman",
            y_label=r"Spearman $\rho$ (median, IQR)",
            y_lim=(-0.05, 1.05),
        ),
    }
    figures = {k: v for k, v in figures.items() if v is not None}

    tables = {
        "perconfig_table": _table_perconfig(runs),
        "nregions_table": _table_nregions(runs),
        "datasets_table": _table_datasets(runs),
        "variant_spearman_table": _table_variant_field(
            kmeans_euclidean, field="spearman"
        ),
        "variant_region_mse_table": _table_variant_field(
            kmeans_euclidean, field="region_rate_mse", variants=_RATE_VARIANT_ORDER
        ),
        "variant_spearman_by_dataset_table": _table_variant_spearman_by_dataset(
            kmeans_euclidean
        ),
        "error_by_size_table": _table_error_by_size(kmeans_euclidean),
    }

    tables_dir = root / "compare"
    clear_dir(tables_dir)
    figures_dir = out or tables_dir
    save_figures(figures, figures_dir)
    for name, table in tables.items():
        save_to_json(table, tables_dir / f"{name}.json")
    logger.info(
        "Comparison figures (%s) -> %s", ", ".join(sorted(figures)), figures_dir
    )
    logger.info("Comparison tables (%s) -> %s", ", ".join(sorted(tables)), tables_dir)


def _parse_args(argv: list[str]) -> tuple[Path, str, Path | None]:
    """Parse `sweep=<path> [format=..] [out=..]` from argv."""
    unknown = [
        a
        for a in argv
        if "=" not in a or a.split("=", 1)[0] not in ("sweep", "format", "out")
    ]
    if unknown:
        raise ValueError(
            f"compare: unknown argument(s) {unknown}; "
            "expected sweep=<path> [format=pdf|png] [out=<dir>]."
        )
    kv = dict(a.split("=", 1) for a in argv)
    if "sweep" not in kv:
        raise ValueError("compare requires sweep=<path>.")
    return (
        Path(kv["sweep"]),
        kv.get("format", "pdf"),
        Path(kv["out"]) if kv.get("out") else None,
    )


def main() -> None:
    """Entry point for the compare stage."""
    root, fmt, out = _parse_args(sys.argv[1:])
    _render_comparisons(root, fmt=fmt, out=out)


if __name__ == "__main__":
    main()
