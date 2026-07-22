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
from src.domain.analysis.selective_prediction import risk_coverage_curve
from src.domain.plot.base import Plot, set_figure_format
from src.domain.plot.charts import band_curve_plot, box_strip_plot, line_whisker_plot, stacked_bar_plot
from src.domain.plot.style import PALETTE, apply_plot_style

setup_logger(log_file="resources/logs.txt")
apply_plot_style()
logger = logging.getLogger(__name__)

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
_BASELINE_LABEL = {"mcp_risk": "MCP", "margin_risk": "Margin", "entropy_risk": "Entropy"}
_BASELINE_COLOR = {"mcp_risk": PALETTE[3], "margin_risk": PALETTE[4], "entropy_risk": PALETTE[5]}


def _load_sweep_runs(root: Path) -> list[dict]:
    """Collect one record per `<config>/<dataset>/<classifier>` run under `root`.

    `distance` and `algorithm` come from each run's composed config, not parsed
    from directory names.
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
        label = f"{meta['distance']} {_ALGO_LABEL.get(meta['algorithm'], meta['algorithm'])}"
        labels.append(label)
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
        series, x_label="number of clusters per run",
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
    """Figure B: accuracy-coverage curves aggregated (median + predictor IQR band) over the full sweep.

    The native-confidence baselines (MCP/margin/entropy) are added as plain median
    curves (no band, to keep the figure readable) wherever `cluster_summary.json`
    carries that score — absent for runs that predate the confidence-baseline feature.
    """
    grid = np.linspace(0.0, 1.0, 101)
    predictor, oracle, randoms = [], [], []
    baselines: dict[str, list[np.ndarray]] = {name: [] for name in _BASELINE_LABEL}
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
        for name in _BASELINE_LABEL:
            if not all(name in summary[c] for c in ids):
                continue
            risk = np.array([summary[c][name] for c in ids], dtype=float)
            cov_b, acc_b = risk_coverage_curve(risk, fail, support)
            baselines[name].append(np.interp(grid, cov_b, acc_b))
    if not predictor:
        return None

    # Mirror coverage -> fraction rejected (riskiest first), so the curves rise
    # left-to-right and match the single-run selective figure (Fig.~\ref{fig:cic_selective}).
    pred_curves = np.vstack(predictor)[:, ::-1]
    oracle_curves = np.vstack(oracle)[:, ::-1]
    lines = [
        ("Oracle", np.median(oracle_curves, axis=0), _ORACLE),
        ("Predictor", np.median(pred_curves, axis=0), _COS),
    ]
    for name, label in _BASELINE_LABEL.items():
        if not baselines[name]:
            continue
        curves = np.vstack(baselines[name])[:, ::-1]
        lines.append((label, np.median(curves, axis=0), _BASELINE_COLOR[name]))
    return band_curve_plot(
        grid,
        lines,
        band=(np.percentile(pred_curves, 25, axis=0),
              np.percentile(pred_curves, 75, axis=0), _COS, "Predictor (IQR across runs)"),
        baseline=float(np.median(randoms)),
        x_label="fraction rejected (riskiest first)",
        y_label="accuracy on retained points",
    )


def _fig_baseline_redundancy(runs: list[dict]) -> Plot | None:
    """Figure F: Spearman rho between the RF predicted rate and each confidence baseline,
    aggregated over the full sweep — how redundant the predictor is with native confidence.

    Reads the per-run `baseline_redundancy` block (`fit_failure_classifier.py`); absent
    for runs that predate the confidence-baseline feature, so groups may be smaller than
    `_fig_rho_by_config`'s.
    """
    labels, values, colors = [], [], []
    for name, label in _BASELINE_LABEL.items():
        vals = [
            r["results"]["baseline_redundancy"][name]
            for r in runs
            if r["results"].get("baseline_redundancy", {}).get(name) is not None
        ]
        if not vals:
            continue
        labels.append(label)
        values.append(np.asarray(vals, dtype=float))
        colors.append(_BASELINE_COLOR[name])
    if not labels:
        return None
    return box_strip_plot(
        labels, values, colors=colors, show_points=True,
        x_label=r"Spearman $\rho$(RF predicted rate, baseline risk)",
        x_lim=(-1.05, 1.05), axvline=0.0,
    )


def _std(values: np.ndarray) -> float:
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
        rows.append({
            "distance": distance,
            "algorithm": algorithm,
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": _std(arr),
            "pct_gt_0_7": 100.0 * float((arr > 0.7).mean()),
            "pct_lt_0": 100.0 * float((arr < 0.0).mean()),
            "n_runs": len(arr),
        })
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
        rows.append({
            "distance": distance,
            "algorithm": algorithm,
            "median": float(np.median(arr)),
            "min": int(arr.min()),
            "max": int(arr.max()),
            "n_runs": len(arr),
        })
    return {"rows": rows}


def _table_perclf_perdataset(runs: list[dict]) -> dict:
    """Table: Spearman rho by classifier and by dataset (tab:perclf_perdataset)."""
    def _group(key: str) -> list[dict]:
        groups: dict[str, list[float]] = {}
        for r in runs:
            rho = r["results"].get("spearman")
            if rho is None:
                continue
            groups.setdefault(r[key], []).append(rho)
        rows = []
        for name, vals in sorted(groups.items()):
            arr = np.array(vals, dtype=float)
            rows.append({
                key: name,
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": _std(arr),
                "min": float(arr.min()),
                "n_runs": len(arr),
            })
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
    """Table: selective-prediction lift by configuration at tau=0.8 (tab:selective).

    Reads the `risk_coverage` block already persisted by `fit_failure_classifier.py`
    (`selective_prediction_metrics`, default `coverage_target=0.8`) — no recomputation.
    Also reports the native-classifier-confidence baselines (`mcp_risk`/`margin_risk`/
    `entropy_risk`, from `confidence_baselines`) alongside the predictor, when present —
    absent (columns omitted) for runs that predate that field.

    `{name}_sig_better_pct` / `{name}_sig_worse_pct` read the paired cluster-level
    bootstrap in `significance` (no re-training, no extra seeds): the % of runs where
    the predictor is significantly better / worse than that baseline (95% CI on
    `lift_diff` excludes 0) — a numeric lift can be noise, this asks whether it's
    provably real, in either direction. `bootstrap_compare`'s `lift_diff` is defined as
    baseline − predictor (`reference="predictor"`), so "better" is `mean < 0`, not `> 0`.
    """
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
                row[f"{name}_sig_better_pct"] = 100.0 * float(np.mean([
                    s["lift_significant"] and s["lift_diff"]["mean"] < 0 for s in sig_vs
                ]))
                row[f"{name}_sig_worse_pct"] = 100.0 * float(np.mean([
                    s["lift_significant"] and s["lift_diff"]["mean"] > 0 for s in sig_vs
                ]))
        rows.append(row)
    return {"rows": rows}


def _render_sweep_results(root: Path, fmt: str = "pdf", out: Path | None = None) -> None:
    """Aggregate the sweep under `root` into the five cross-run paper figures (written to
    `out`) and the four Results-section JSON tables (written to `root`, next to the raw
    per-run trees they were aggregated from): tab:perconfig, tab:nclusters,
    tab:perclf_perdataset, tab:selective.

    All clustering configurations, including HDBSCAN, contribute to every output;
    HDBSCAN's higher variability is reported rather than hidden.
    """
    set_figure_format(fmt)
    runs = _load_sweep_runs(root)
    if not runs:
        logger.warning("No sweep runs found under %s; nothing to render.", root)
        return
    n_hdb = sum(1 for r in runs if r["algorithm"] == "hdbscan")
    logger.info(
        "Sweep results from %d runs (%d main, %d hdbscan) under %s",
        len(runs), len(runs) - n_hdb, n_hdb, root,
    )

    figures = {
        "figure/rho_by_config": _fig_rho_by_config(runs),
        "figure/rho_vs_clusters": _fig_rho_vs_clusters(runs),
        "figure/family_importance": _fig_family_importance(runs),
        "figure/selective_sweep": _fig_selective_sweep(runs),
        "figure/baseline_redundancy": _fig_baseline_redundancy(runs),
    }
    figures = {k: v for k, v in figures.items() if v is not None}

    tables = {
        "json/perconfig_table": _table_perconfig(runs),
        "json/nclusters_table": _table_nclusters(runs),
        "json/perclf_perdataset_table": _table_perclf_perdataset(runs),
        "json/selective_table": _table_selective(runs),
    }

    figures_base = out or root
    bus = LogDispatcher()
    bus.subscribe(FilesystemFigureSubscriber(figures_base))
    bus.subscribe(JSONSubscriber(root))
    bus.publish(LogBundle.from_dict({**figures, **tables}))
    logger.info(
        "Sweep figures (%s) -> %s",
        ", ".join(sorted(k.split("/")[-1] for k in figures)), figures_base,
    )
    logger.info(
        "Sweep tables (%s) -> %s",
        ", ".join(sorted(k.split("/")[-1] for k in tables)), root,
    )


def _parse_args(argv: list[str]) -> tuple[Path, str, Path | None]:
    """Parse `sweep=<path> [format=..] [out=..]` from argv."""
    kv = dict(a.split("=", 1) for a in argv if "=" in a)
    if "sweep" not in kv:
        raise ValueError("sweep_results requires sweep=<path>.")
    return Path(kv["sweep"]), kv.get("format", "pdf"), Path(kv["out"]) if kv.get("out") else None


def main():
    """Main entry point: aggregate a sweep tree into cross-run paper figures + result tables."""
    root, fmt, out = _parse_args(sys.argv[1:])
    _render_sweep_results(root, fmt=fmt, out=out)


if __name__ == "__main__":
    main()
