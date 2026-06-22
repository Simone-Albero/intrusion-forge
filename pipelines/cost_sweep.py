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

DEFAULT_CAPS = [5000, 10000, 25000, 50000, 100000, 200000, 300000]

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

        complexity_by_cap[k] = cluster_complexity
        records[k] = {
            "spearman": results.get("spearman"),
            "spearman_pvalue": results.get("spearman_pvalue"),
            "r2": results.get("r2"),
            "mae": results.get("mae"),
            "n_clusters_used": results.get("n_clusters_used"),
            "n_subsample": int(graph.y_cluster.shape[0]),
            "floor_bound_frac": _floor_bound_frac(counts, n_total, min(k, n_total), floor),
            "graph_build_s": graph_build_s,
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
    }
    save_to_json(out, paths.shared / "cost_sweep.json")
    logger.info("Cap-sweep written -> %s", paths.shared / "cost_sweep.json")


if __name__ == "__main__":
    main()
