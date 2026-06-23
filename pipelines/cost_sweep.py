import logging
import re
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from omegaconf import OmegaConf
from scipy.stats import spearmanr

from src.core.config import load_config, to_container
from src.core.io import load_df
from src.core.utils import load_from_json, save_to_json
from src.domain.analysis.complexity import prepare_complexity_graph
from pipelines.common import paths_from_cfg
from pipelines.compute_complexity import (
    cluster_class_map,
    compute_class_complexity,
    compute_cluster_complexity,
)
from pipelines.fit_failure_classifier import (
    build_cluster_summary,
    fit_failure_classifier,
)

logger = logging.getLogger(__name__)

# Cap grid densified around the cost/quality knee (~25-50k observed across cells);
# 5000 dropped — it floor-bound-saturates to the same subsample as 10000 (degenerate
# duplicate), while 10000 is kept as the low/floor-bound anchor.
DEFAULT_CAPS = [10000, 15000, 25000, 35000, 50000, 75000, 100000, 200000, 300000]

# Measure-name → complexity family. Order matters: geometry keys like
# `frac_at_risk` start with "f" but are not F-family.
_GEOMETRY = {
    "max_dispersion",
    "p95_dispersion",
    "dist_to_nearest_centroid",
    "p5_silhouette",
    "frac_at_risk",
}


def _family(measure: str) -> str | None:
    """Map a complexity measure name to its family (F/N/T/Net/G), or None to skip."""
    if measure in _GEOMETRY:
        return "G"
    if measure.startswith("network_density") or measure in ("cls_coef", "hub"):
        return "Net"
    if re.match(r"f[1-4]_", measure):
        return "F"
    if re.match(r"n[1-4]_", measure):
        return "N"
    if measure in ("t2", "t3", "t4"):
        return "T"
    return None


# True-size bins (cluster members in train) for stratifying intrinsic fidelity.
# Half-open [lo, hi); the last bin is open-ended (hi=None).
_SIZE_BINS = [(0, 20), (20, 50), (50, 200), (200, None)]


def _size_bin_label(lo: int, hi: int | None) -> str:
    return f">={lo}" if hi is None else f"{lo}-{hi}"


def _per_family_rankcorr(
    cluster_ids: list[str], complexity_k: dict, complexity_ref: dict
) -> dict[str, float]:
    """Mean per-family Spearman of per-cluster measure values over `cluster_ids`."""
    measures = {
        m
        for cid in cluster_ids
        for m in complexity_k.get(cid, {})
        if _family(m) is not None
    }
    per_family: dict[str, list[float]] = {}
    for m in measures:
        a, b = [], []
        for cid in cluster_ids:
            va = complexity_k.get(cid, {}).get(m)
            vb = complexity_ref.get(cid, {}).get(m)
            if va is not None and vb is not None:
                a.append(va)
                b.append(vb)
        if len(a) < 2:
            continue
        rho = spearmanr(a, b).statistic
        if not np.isnan(rho):
            per_family.setdefault(_family(m), []).append(float(rho))
    return {fam: float(np.mean(vals)) for fam, vals in sorted(per_family.items())}


def _shared_ids(complexity_k: dict, complexity_ref: dict) -> list[str]:
    return [c for c in complexity_k if c in complexity_ref]


def _family_rankcorr(complexity_k: dict, complexity_ref: dict) -> dict[str, float]:
    """Mean per-family Spearman rank-corr of per-cluster measure values, K vs k_ref."""
    return _per_family_rankcorr(
        _shared_ids(complexity_k, complexity_ref), complexity_k, complexity_ref
    )


def _family_rankcorr_by_size(
    complexity_k: dict, complexity_ref: dict, cluster_size: dict[str, int]
) -> dict[str, dict]:
    """`_family_rankcorr` stratified by true cluster size (members in train).

    Isolates whether low fidelity is concentrated in small clusters — the ones
    governed by `min_subsample_per_cluster`. Noise rows (absent from
    `cluster_size`) are dropped.
    """
    shared = _shared_ids(complexity_k, complexity_ref)
    out: dict[str, dict] = {}
    for lo, hi in _SIZE_BINS:
        ids = [
            c
            for c in shared
            if c in cluster_size
            and cluster_size[c] >= lo
            and (hi is None or cluster_size[c] < hi)
        ]
        out[_size_bin_label(lo, hi)] = {
            "n_clusters": len(ids),
            "rankcorr": _per_family_rankcorr(ids, complexity_k, complexity_ref),
        }
    return out


def _floor_bound_frac(counts: np.ndarray, n_total: int, cap: int, floor: int) -> float:
    """Fraction of clusters whose subsample size is set by the floor, not the cap.

    A cluster is floor-bound when its proportional share round(cap·n_c/n_total) ≤ floor;
    below that the cap is inert for it. When this →1 the cap no longer drives n_subsample
    (the degenerate tail of the sweep).
    """
    prop = np.round(cap * counts / n_total).astype(int)
    return float((prop <= floor).mean())


# Cost analysis on the per-cap records dict; replayable on a saved cost_sweep.json.
_STABLE_FAMILIES = ("F", "G")
_FIDELITY_TAU = 0.80
_MAX_FLOOR_BOUND = 0.90


def _fit_cost_model(records: dict[int, dict]) -> dict:
    """Fit T(m) = c·m^alpha to (n_subsample, graph_build_s) in log-log space."""
    pts = {int(r["n_subsample"]): float(r["graph_build_s"])
           for r in records.values()
           if r.get("n_subsample") and (r.get("graph_build_s") or 0) > 0}
    if len(pts) < 3:
        return {"alpha": None, "c": None, "r2": None, "n_points": len(pts)}
    m = np.array(sorted(pts), dtype=float)
    lm, lt = np.log(m), np.log([pts[int(x)] for x in m])
    alpha, log_c = np.polyfit(lm, lt, 1)
    ss_res = float(np.sum((lt - (alpha * lm + log_c)) ** 2))
    ss_tot = float(np.sum((lt - lt.mean()) ** 2))
    return {
        "alpha": float(alpha),
        "c": float(np.exp(log_c)),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else None,
        "n_points": len(pts),
    }


def _spearman_ci(rho: float | None, n: int | None, crit: float = 1.96) -> tuple:
    """Fisher-z confidence interval for a Spearman rho on n samples."""
    if rho is None or n is None or n < 4:
        return (None, None)
    z = np.arctanh(np.clip(rho, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(n - 3)
    return float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se))


def _indistinguishable(rho_a, n_a, rho_b, n_b, crit: float = 1.96) -> bool:
    """Independent Fisher-z test that two Spearman rho's match (conservative:
    caps share clustering, so the paired bootstrap is the rigorous version)."""
    if None in (rho_a, n_a, rho_b, n_b) or n_a < 4 or n_b < 4:
        return False
    za = np.arctanh(np.clip(rho_a, -0.999999, 0.999999))
    zb = np.arctanh(np.clip(rho_b, -0.999999, 0.999999))
    se = np.sqrt(1.0 / (n_a - 3) + 1.0 / (n_b - 3))
    return bool(abs(za - zb) / se < crit)


def _iso_quality_cap(records: dict[int, dict], k_ref: int) -> dict:
    """Smallest cap indistinguishable from k_ref, with every larger cap too."""
    ref = records[k_ref]
    rho_ref, n_ref = ref.get("spearman"), ref.get("n_clusters_used")
    build_ref = ref.get("graph_build_s") or float("nan")
    chosen = k_ref
    for cap in sorted(records, reverse=True):
        r = records[cap]
        if _indistinguishable(r.get("spearman"), r.get("n_clusters_used"), rho_ref, n_ref):
            chosen = cap
        else:
            break
    r = records[chosen]
    build = r.get("graph_build_s") or float("nan")
    return {
        "cap": chosen,
        "rho": r.get("spearman"),
        "rho_ref": rho_ref,
        "speedup": build_ref / build if build else None,
        "build_s": build,
    }


def _select_cap(
    records: dict[int, dict],
    k_ref: int,
    families: tuple = _STABLE_FAMILIES,
    tau: float = _FIDELITY_TAU,
    max_floor_bound: float = _MAX_FLOOR_BOUND,
) -> dict:
    """Cheapest cap keeping stable-family fidelity >= tau and not floor-bound."""
    build_ref = records[k_ref].get("graph_build_s") or float("nan")
    params = {"families": list(families), "tau": tau, "max_floor_bound": max_floor_bound}
    for cap in sorted(records):
        r = records[cap]
        fam = r.get("intrinsic_family_rankcorr", {})
        vals = [fam[f] for f in families if fam.get(f) is not None]
        fb = r.get("floor_bound_frac", 1.0)
        if vals and min(vals) >= tau and fb < max_floor_bound:
            build = r.get("graph_build_s") or float("nan")
            return {
                "cap": cap,
                "rho": r.get("spearman"),
                "min_stable_fidelity": float(min(vals)),
                "floor_bound_frac": fb,
                "speedup": build_ref / build if build else None,
                "binding": "fidelity" if min(vals) <= tau + 0.05 else "floor_bound",
                "params": params,
            }
    return {"cap": None, "params": params}


def _cost_units(m: int, k: int, d: int) -> dict:
    """Hardware-neutral k-NN build cost: m² distance evals over d features, m·k edges."""
    return {
        "distance_evals": m * (m - 1),
        "feature_ops": m * (m - 1) * d,
        "knn_edges": m * k,
    }


def _family_importances(importances: dict[str, float]) -> dict[str, float]:
    """Sum failure-classifier feature importances by complexity family."""
    out: dict[str, float] = {}
    for col, imp in importances.items():
        measure = col.split("_", 1)[1] if col.startswith(("cluster_", "class_")) else col
        fam = _family(measure)
        if fam is not None:
            out[fam] = out.get(fam, 0.0) + float(imp)
    return out


def _paired_bootstrap_iso(
    records: dict[int, dict], k_ref: int, n_boot: int = 2000, seed: int = 42
) -> dict:
    """Paired bootstrap on Δρ = ρ_cap − ρ_ref over clusters common to cap and k_ref.

    Needs per-cap `oof_pairs`; returns {"available": False} on legacy sweeps.
    """
    ref = records[k_ref].get("oof_pairs")
    if not ref:
        return {"available": False}
    rng = np.random.default_rng(seed)
    per_cap: dict[str, dict] = {}
    for cap in sorted(records, reverse=True):
        pairs = records[cap].get("oof_pairs")
        if not pairs:
            continue
        common = [c for c in pairs if c in ref]
        if len(common) < 4:
            per_cap[str(cap)] = {"n_common": len(common)}
            continue
        pc = np.array([pairs[c] for c in common], dtype=float)
        rc = np.array([ref[c] for c in common], dtype=float)
        n = len(common)
        deltas = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            deltas[b] = (spearmanr(pc[idx, 0], pc[idx, 1]).statistic
                         - spearmanr(rc[idx, 0], rc[idx, 1]).statistic)
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        per_cap[str(cap)] = {
            "n_common": n,
            "delta_rho_mean": float(np.nanmean(deltas)),
            "delta_ci_low": float(lo),
            "delta_ci_high": float(hi),
            "indistinguishable": bool(lo <= 0.0 <= hi),
        }
    chosen = k_ref
    for cap in sorted(records, reverse=True):
        e = per_cap.get(str(cap), {})
        if e.get("indistinguishable"):
            chosen = cap
        elif "indistinguishable" in e:
            break
    return {"available": True, "n_boot": n_boot, "iso_cap": chosen, "per_cap": per_cap}


_COMPLEXITY_FNS = {
    "build_knn_graph", "prepare_complexity_graph",
    "compute_f_measures", "compute_n_measures", "compute_network_measures",
    "compute_t_measures", "compute_cluster_geometry",
    "compute_complexity_from_graph", "compute_cluster_complexity",
    "compute_class_complexity",
}


def _stage_seconds(timing_path: Path, exclude: set[str] | None = None) -> float:
    """Sum durations in a timing.json, skipping `exclude`d function names."""
    try:
        rows = load_from_json(timing_path)
    except (FileNotFoundError, OSError):
        return 0.0
    exclude = exclude or set()
    return float(
        sum(r.get("duration_s", 0.0) for r in rows if r.get("function") not in exclude)
    )


def _cost_analysis(records: dict[int, dict], k_ref: int) -> dict:
    """Bundle the cost model, per-cap quality CIs, iso-quality cap and selection."""
    ref = records[k_ref]
    build_ref = ref.get("graph_build_s") or float("nan")
    quality_ci: dict[str, dict] = {}
    for cap in sorted(records, reverse=True):
        r = records[cap]
        lo, hi = _spearman_ci(r.get("spearman"), r.get("n_clusters_used"))
        build = r.get("graph_build_s") or float("nan")
        quality_ci[str(cap)] = {
            "rho": r.get("spearman"),
            "ci_low": lo,
            "ci_high": hi,
            "n_clusters_used": r.get("n_clusters_used"),
            "speedup_vs_ref": build_ref / build if build else None,
            "indistinguishable_from_ref": _indistinguishable(
                r.get("spearman"), r.get("n_clusters_used"),
                ref.get("spearman"), ref.get("n_clusters_used"),
            ),
        }
    return {
        "cost_model": _fit_cost_model(records),
        "iso_quality_cap": _iso_quality_cap(records, k_ref),
        "iso_quality_bootstrap": _paired_bootstrap_iso(records, k_ref),
        "selected_cap": _select_cap(records, k_ref),
        "quality_ci": quality_ci,
    }


def main():
    """Sweep the complexity sample-cap for one cell and write shared/cost_sweep.json."""
    cfg = load_config(
        config_path=Path(__file__).parent.parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )
    paths = paths_from_cfg(cfg)

    caps = OmegaConf.select(cfg, "capsweep.caps")
    caps = [int(k) for k in caps] if caps else list(DEFAULT_CAPS)
    caps = sorted(set(caps), reverse=True)
    k_ref = caps[0]

    # Load the cell once (cap-invariant): train geometry, clusters, RF predictions.
    num_cols = list(cfg.data.num_cols) if cfg.data.num_cols else []
    cat_cols = list(cfg.data.cat_cols) if cfg.data.cat_cols else []
    train_df = load_df(str(paths.processed_data / f"train.{cfg.data.extension}"))
    X_num = (
        train_df[num_cols].to_numpy(dtype=np.float64)
        if num_cols
        else np.empty((len(train_df), 0))
    )
    X_cat = train_df[cat_cols].to_numpy() if cat_cols else None
    y_class = train_df[f"encoded_{cfg.data.label_col}"].to_numpy(dtype=np.int64)
    y_cluster = train_df["cluster"].to_numpy(dtype=np.int64)
    # Full map (noise included) for the flag-only noise rows injected downstream.
    cluster_to_class = cluster_class_map(y_cluster, y_class)

    clusters_meta = load_from_json(paths.shared / "metadata/clusters_meta.json")
    noise_cluster_ids = clusters_meta.get("noise_cluster_ids", [])

    # Exclude noise pseudo-clusters from the graph (they re-enter downstream only
    # as flag-only rows), matching pipelines/compute_complexity.py.
    if noise_cluster_ids:
        genuine = ~np.isin(y_cluster, noise_cluster_ids)
        X_num, y_class, y_cluster = X_num[genuine], y_class[genuine], y_cluster[genuine]
        X_cat = X_cat[genuine] if X_cat is not None else None

    predictions = load_from_json(paths.outputs / "analysis/predictions/test.json")
    param_grid = to_container(cfg.failure_classifier.param_grid)

    # True genuine-cluster sizes: drive intrinsic size-stratification and the
    # floor-bound fraction (cap-invariant, so computed once).
    unique_clusters, counts = np.unique(y_cluster, return_counts=True)
    n_total = int(counts.sum())
    cluster_size = {str(c): int(n) for c, n in zip(unique_clusters, counts)}
    floor = int(cfg.complexity.min_subsample_per_cluster)
    d_feat = X_num.shape[1] + (X_cat.shape[1] if X_cat is not None else 0)

    logger.info(
        "Cap-sweep on %s [%s], caps=%s, k_ref=%d",
        cfg.data.file_name,
        cfg.complexity.distance,
        caps,
        k_ref,
    )

    records: dict[int, dict] = {}
    complexity_by_cap: dict[int, dict] = {}
    for k in caps:
        t0 = perf_counter()
        graph = prepare_complexity_graph(
            X_num,
            X_cat,
            y_class,
            y_cluster,
            k=cfg.complexity.k,
            max_samples=k,
            min_per_cluster=cfg.complexity.min_subsample_per_cluster,
            metric=cfg.complexity.distance,
            random_state=cfg.seed,
        )
        graph_build_s = perf_counter() - t0

        cluster_complexity = compute_cluster_complexity(
            graph,
            noise_cluster_ids,
            cluster_to_class,
            top_k_clusters=cfg.complexity.top_k_clusters,
            metric=cfg.complexity.distance,
            random_state=cfg.seed,
        )
        class_complexity = compute_class_complexity(
            graph,
            top_k_clusters=cfg.complexity.top_k_clusters,
            metric=cfg.complexity.distance,
            random_state=cfg.seed,
        )
        cluster_summary = build_cluster_summary(
            cluster_complexity, class_complexity, predictions
        )
        results = fit_failure_classifier(
            cluster_summary,
            param_grid,
            n_outer_splits=cfg.failure_classifier.n_outer_splits,
            n_inner_splits=cfg.failure_classifier.n_inner_splits,
            min_test_support=cfg.failure_classifier.min_test_support,
            random_state=cfg.seed,
            analysis_bus=None,
        )

        # Per-cluster (predicted, observed) pairs for the paired bootstrap.
        oof_pred = results.get("oof_predicted_rate", {})
        oof_pairs = {
            cid: [float(pred), float(cluster_summary[cid]["failure_rate"])]
            for cid, pred in oof_pred.items()
            if cluster_summary.get(cid, {}).get("failure_rate") is not None
        }

        m = int(graph.y_cluster.shape[0])
        complexity_by_cap[k] = cluster_complexity
        records[k] = {
            "spearman": results.get("spearman"),
            "spearman_pvalue": results.get("spearman_pvalue"),
            "r2": results.get("r2"),
            "mae": results.get("mae"),
            "n_clusters_used": results.get("n_clusters_used"),
            "n_subsample": m,
            "floor_bound_frac": _floor_bound_frac(counts, n_total, min(k, n_total), floor),
            "graph_build_s": graph_build_s,
            "cost_units": _cost_units(m, cfg.complexity.k, d_feat),
            "family_importances": _family_importances(results.get("feature_importances", {})),
            "oof_pairs": oof_pairs,
        }
        logger.info(
            "K=%d: rho=%.4f, n_subsample=%d, build=%.1fs",
            k,
            (
                records[k]["spearman"]
                if records[k]["spearman"] is not None
                else float("nan")
            ),
            records[k]["n_subsample"],
            graph_build_s,
        )

    # Intrinsic fidelity: per-family stability of the measure vectors vs k_ref,
    # overall and stratified by true cluster size.
    for k in caps:
        records[k]["intrinsic_family_rankcorr"] = _family_rankcorr(
            complexity_by_cap[k], complexity_by_cap[k_ref]
        )
        records[k]["intrinsic_by_size"] = _family_rankcorr_by_size(
            complexity_by_cap[k], complexity_by_cap[k_ref], cluster_size
        )

    out = {
        "dataset": cfg.data.file_name,
        "distance": cfg.complexity.distance,
        "classifier": cfg.classifier.name,
        "k_ref": k_ref,
        "caps": {str(k): records[k] for k in caps},
        "cost_analysis": _cost_analysis(records, k_ref),
    }

    # Complexity-build share of the end-to-end pipeline, at k_ref vs selected cap.
    prep_clustering_s = _stage_seconds(paths.shared / "timing.json", exclude=_COMPLEXITY_FNS)
    classify_s = _stage_seconds(paths.outputs / "timing.json")
    non_complexity_s = prep_clustering_s + classify_s
    sel = out["cost_analysis"]["selected_cap"]["cap"]

    def _share(build_s: float) -> float | None:
        total = build_s + non_complexity_s
        return build_s / total if total else None

    out["pipeline_cost"] = {
        "prep_clustering_s": prep_clustering_s,
        "classify_s": classify_s,
        "non_complexity_s": non_complexity_s,
        "complexity_build_s_kref": records[k_ref]["graph_build_s"],
        "complexity_share_kref": _share(records[k_ref]["graph_build_s"]),
        "complexity_build_s_selected": records[sel]["graph_build_s"] if sel else None,
        "complexity_share_selected": _share(records[sel]["graph_build_s"]) if sel else None,
        "selected_cap": sel,
    }
    save_to_json(out, paths.shared / "cost_sweep.json")
    logger.info("Cap-sweep written -> %s", paths.shared / "cost_sweep.json")


if __name__ == "__main__":
    main()
