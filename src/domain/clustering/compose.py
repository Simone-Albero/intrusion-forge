import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from src.domain.clustering import ClusteringFactory
from src.domain.clustering.base import ClusterFn, grid_search
from src.domain.clustering.ensemble import make_ensemble_cluster_fn

if TYPE_CHECKING:
    from src.domain.clustering.ensemble import ConsensusReporter

Reporter = Callable[[str, dict], None]  # (algo_name, grid_search_result) -> None


def _is_grid_spec(value) -> bool:
    """True if value is an abs list or a `{rel/rel_inv/sqrt: [...]}` dict."""
    if isinstance(value, list):
        return True
    if isinstance(value, dict) and any(
        isinstance(value.get(k), list) for k in ("rel", "rel_inv", "sqrt")
    ):
        return True
    return False


def _split_grid_fixed(params: dict) -> tuple[dict, dict]:
    """Split params into ({grid-spec → grid}, {scalar → fixed})."""
    grid = {k: v for k, v in params.items() if _is_grid_spec(v)}
    fixed = {k: v for k, v in params.items() if not _is_grid_spec(v)}
    return grid, fixed


def _clip(x: int, lo, hi) -> int:
    if lo is not None:
        x = max(int(lo), x)
    if hi is not None:
        x = min(int(hi), x)
    return x


def _resolve_grid(grid: dict, effective_n: int) -> dict[str, list]:
    """Expand each grid entry to a concrete list of values.

    Abs lists pass through unchanged.
    `{rel: [...], min?, max?}` → `[clip(round(effective_n * r), min, max) for r in rel]`.
    `{rel_inv: [...], min?, max?}` → `[clip(round(1 / r), min, max) for r in rel_inv]`.
    `{sqrt: [...], min?, max?}` → `[clip(round(c * sqrt(effective_n)), min, max) for c in sqrt]`.
    Deduplicated preserving insertion order; values are always int.
    """
    out: dict[str, list] = {}
    for key, spec in grid.items():
        if isinstance(spec, list):
            out[key] = spec
            continue
        lo = spec.get("min")
        hi = spec.get("max")
        if "rel" in spec:
            raw = (int(round(effective_n * float(r))) for r in spec["rel"])
        elif "sqrt" in spec:
            raw = (int(round(float(c) * math.sqrt(effective_n))) for c in spec["sqrt"])
        else:
            raw = (int(round(1.0 / float(r))) for r in spec["rel_inv"])
        seen: set[int] = set()
        values: list[int] = []
        for x in raw:
            x = _clip(x, lo, hi)
            if x not in seen:
                seen.add(x)
                values.append(x)
        out[key] = values
    return out


def _resolve_scalar_rel(value, effective_n: int) -> int:
    """Resolve int/float scalar or `{rel/sqrt: float, min?, max?}` to a single int.

    Scalar values pass through (cast to int). `rel` scales linearly with
    `effective_n`; `sqrt` scales with its square root (steadier cluster counts
    across class sizes). `min`/`max` clip the result.
    """
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, dict) and "rel" in value:
        x = int(round(effective_n * float(value["rel"])))
        return _clip(x, value.get("min"), value.get("max"))
    if isinstance(value, dict) and "sqrt" in value:
        x = int(round(float(value["sqrt"]) * math.sqrt(effective_n)))
        return _clip(x, value.get("min"), value.get("max"))
    raise TypeError(f"unsupported scalar-rel spec: {value!r}")


def _make_single_cluster_fn(
    name: str,
    params: dict,
    max_fit_samples: int,
    random_state: int,
    reporter: Reporter | None = None,
) -> ClusterFn:
    """Build a ClusterFn for a single registered algorithm."""
    fit_fn = ClusteringFactory.get(name)
    grid_raw, fixed = _split_grid_fixed(params)

    def _fn(X_num: np.ndarray, X_cat: np.ndarray | None = None) -> np.ndarray:
        effective_n = min(X_num.shape[0], max_fit_samples)
        grid = _resolve_grid(grid_raw, effective_n)
        common = {
            "max_fit_samples": max_fit_samples,
            "random_state": random_state,
            **fixed,
        }
        try:
            if grid:
                result = grid_search(X_num, X_cat, fit_fn, grid, **common)
                if reporter is not None:
                    reporter(name, result)
                return fit_fn(X_num, X_cat=X_cat, **result["best"]["combo"], **common)
            return fit_fn(X_num, X_cat=X_cat, **common)
        except Exception as e:
            # Graceful degradation: this algorithm abstains from the ensemble vote.
            # Co-association (noise = no-vote, variable denominator) handles -1 cleanly,
            # so the consensus continues with M-1 effective voters.
            if reporter is not None:
                reporter(name, {"best": None, "sweep": [], "error": f"{type(e).__name__}: {e}"})
            return np.full(X_num.shape[0], -1, dtype=np.int64)

    return _fn


def build_cluster_fn(
    algorithms: dict[str, dict],
    consensus_threshold: float,
    max_fit_samples: int,
    random_state: int,
    min_consensus_size=1,
    reporter: Reporter | None = None,
    consensus_reporter: "ConsensusReporter | None" = None,
    propagation_confidence_floor: float = 0.0,
    weight_voters: bool = True,
    refine_geometry: bool = True,
    refine_margin: float = 0.8,
) -> ClusterFn:
    """Build a ClusterFn from {algorithm_name: params}; ensembles when >1 key.

    `min_consensus_size` is the HDBSCAN(precomputed) min_cluster_size on the
    co-association matrix; may be `int` or `{rel/sqrt: float, min?, max?}`. Ignored
    for a single algorithm. `reporter` is invoked once per algorithm with its
    grid-search result; `consensus_reporter` receives the consensus diagnostics
    on each ensemble call. `weight_voters`, `refine_geometry` and `refine_margin`
    configure the ensemble's voter weighting and feature-space refinement.
    """
    if not algorithms:
        raise ValueError("build_cluster_fn: algorithms is empty")
    fns = [
        _make_single_cluster_fn(
            name, params, max_fit_samples, random_state, reporter=reporter
        )
        for name, params in algorithms.items()
    ]
    if len(fns) == 1:
        return fns[0]
    return make_ensemble_cluster_fn(
        fns,
        threshold=consensus_threshold,
        min_consensus_size=min_consensus_size,
        max_fit_samples=max_fit_samples,
        random_state=random_state,
        consensus_reporter=consensus_reporter,
        propagation_confidence_floor=propagation_confidence_floor,
        weight_voters=weight_voters,
        refine_geometry=refine_geometry,
        refine_margin=refine_margin,
    )
