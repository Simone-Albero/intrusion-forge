"""Stress test for per-class ensemble clustering — measures wall time + peak RSS.

Usage:
    python scripts/stress_ensemble.py [N=50000] [F=30] [CENTERS=10]

Synthesizes a single class of N points × F features, runs `_cluster_per_class`
with the current `configs/clustering/ensemble.yaml`, reports total time,
peak RSS, and the consensus diagnostics.
"""

from __future__ import annotations

import gc
import logging
import resource
import sys
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from sklearn.datasets import make_blobs

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipelines.prepare_data import _cluster_per_class  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
    f = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    centers = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    print(f"[setup] N={n}  F={f}  centers={centers}")
    rss0 = peak_rss_mb()

    x, _ = make_blobs(
        n_samples=n, centers=centers, n_features=f, cluster_std=1.2, random_state=42
    )
    x = x.astype(np.float64)
    y = np.array(["A"] * n)
    classes = ["A"]
    print(f"[setup] data ready, RSS={peak_rss_mb():.0f} MB (+{peak_rss_mb()-rss0:.0f})")

    cfg = OmegaConf.load(ROOT / "configs/clustering/ensemble.yaml")
    algorithms = OmegaConf.to_container(cfg.algorithms, resolve=True)
    mcs = (
        OmegaConf.to_container(cfg.min_consensus_size, resolve=True)
        if not isinstance(cfg.min_consensus_size, (int, float))
        else cfg.min_consensus_size
    )

    print(f"[setup] algorithms: {list(algorithms.keys())}")
    print(f"[setup] consensus_threshold={cfg.consensus_threshold} min_consensus_size={mcs}")
    print(f"[setup] max_fit_samples={cfg.max_fit_samples}")

    gc.collect()
    t0 = time.perf_counter()
    labels, centroids, noise_ids, report = _cluster_per_class(
        x, y, classes,
        algorithms=algorithms,
        consensus_threshold=cfg.consensus_threshold,
        max_fit_samples=cfg.max_fit_samples,
        min_consensus_size=mcs,
        random_state=0,
        metric="euclidean",
    )
    elapsed = time.perf_counter() - t0

    print()
    print(f"=== RESULTS ===")
    print(f"  wall time:     {elapsed:.1f}s  ({elapsed/60:.2f} min)")
    print(f"  peak RSS:      {peak_rss_mb():.0f} MB")
    print(f"  n_clusters:    {len(set(labels.tolist()) - {-1})}")
    print(f"  n_noise:       {int((labels == -1).sum())}")
    print()
    print(f"=== algo bests ===")
    for name, algo_r in report["A"]["algorithms"].items():
        b = algo_r["best"]
        print(f"  {name:9s}: combo={b['combo']} n_clusters={b['n_clusters']} score={b['score']:.3f} dur={b['duration_s']:.1f}s")
    print()
    print(f"=== consensus ===")
    for k, v in report["A"]["consensus"].items():
        if isinstance(v, (int, float, type(None))):
            print(f"  {k}: {v}")
        elif isinstance(v, dict) and len(v) <= 12:
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: <list of {len(v)}>")


if __name__ == "__main__":
    main()
