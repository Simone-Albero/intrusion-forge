from collections.abc import Callable

import hdbscan
import numpy as np
from sklearn.metrics import adjusted_rand_score

from src.domain.clustering.base import ClusterFn

ConsensusReporter = Callable[[dict], None]


def _coassociation_matrix(labels_arr_sub: np.ndarray) -> tuple[np.ndarray, dict]:
    """Symmetric (s, s) co-association in [0, 1] + pairwise per-algo agreement.

    Noise = no-vote at variable denominator. For a pair (i, j):
        num[i,j] = sum_m [label_m(i) == label_m(j) AND both != -1]
        den[i,j] = sum_m [both != -1]
        co[i,j]  = num/den if den > 0 else 0.0

    Returns `(co_assoc, pairwise)` where pairwise is `{(i,j): agreement}` for
    each algo pair (i < j), computed as fraction of point-pairs on which the
    two algorithms agree on "in-same-cluster" (XNOR), restricted to pairs both
    algorithms classified (no-noise).
    """
    M, s = labels_arr_sub.shape
    valid = labels_arr_sub != -1
    num = np.zeros((s, s), dtype=np.float32)
    den = np.zeros((s, s), dtype=np.float32)
    same_cluster: list[np.ndarray] = []
    for m in range(M):
        l = labels_arr_sub[m]
        v = valid[m]
        pair_valid = v[:, None] & v[None, :]
        sc = (l[:, None] == l[None, :]) & pair_valid
        same_cluster.append(sc)
        den += pair_valid.astype(np.float32)
        num += sc.astype(np.float32)
    den_safe = np.where(den == 0, 1.0, den)
    co = (num / den_safe).astype(np.float32)
    co[den == 0] = 0.0
    np.fill_diagonal(co, 1.0)

    triu = np.triu_indices(s, k=1)
    pairwise: dict[tuple[int, int], float] = {}
    for i in range(M):
        for j in range(i + 1, M):
            both_valid = valid[i][:, None] & valid[i][None, :] & valid[j][:, None] & valid[j][None, :]
            both_pairs = both_valid[triu]
            agree = (same_cluster[i] == same_cluster[j])[triu]
            agree_on_valid = agree[both_pairs]
            pairwise[(i, j)] = float(agree_on_valid.mean()) if agree_on_valid.size else 0.0
    return co, pairwise


def _agreement_with_sub(labels_arr: np.ndarray, sub_idx: np.ndarray, j_idx: np.ndarray) -> np.ndarray:
    """Co-association agreement of points j_idx vs all sub_idx points.

    Same noise = no-vote convention. Returns (len(j_idx), len(sub_idx)) float32.
    """
    sub_labels = labels_arr[:, sub_idx]
    j_labels = labels_arr[:, j_idx]
    sub_valid = sub_labels != -1
    j_valid = j_labels != -1
    match = (
        (j_labels[:, :, None] == sub_labels[:, None, :])
        & j_valid[:, :, None]
        & sub_valid[:, None, :]
    )
    both_valid = j_valid[:, :, None] & sub_valid[:, None, :]
    num = match.sum(axis=0).astype(np.float32)
    den = both_valid.sum(axis=0).astype(np.float32)
    return np.where(den > 0, num / np.where(den == 0, 1.0, den), 0.0).astype(np.float32)


def _propagate_labels(
    labels_arr: np.ndarray,
    sub_idx: np.ndarray,
    sub_labels: np.ndarray,
    non_sub: np.ndarray,
    floor: float,
    full_labels: np.ndarray,
    batch_size: int = 1000,
) -> dict:
    """Assign non-sub points to the cluster of the sub-point with highest agreement."""
    max_agreements: list[float] = []
    low_conf = 0
    for start in range(0, len(non_sub), batch_size):
        batch = non_sub[start : start + batch_size]
        agree = _agreement_with_sub(labels_arr, sub_idx, batch)
        best = agree.argmax(axis=1)
        max_vals = agree[np.arange(len(batch)), best]
        max_agreements.extend(max_vals.tolist())
        best_labels = sub_labels[best]
        noise_mask = (best_labels == -1) | (max_vals == 0) | (max_vals < floor)
        full_labels[batch] = np.where(noise_mask, -1, best_labels)
        low_conf += int(noise_mask.sum())
    if not max_agreements:
        return {"mean_max": 0.0, "low_conf_ratio": 0.0}
    return {
        "mean_max": float(np.mean(max_agreements)),
        "low_conf_ratio": low_conf / len(max_agreements),
    }


def _build_diagnostics(
    labels_arr: np.ndarray,
    sub_idx: np.ndarray,
    sub_labels: np.ndarray,
    co: np.ndarray,
    pairwise_idx: dict,
    s: int,
    n: int,
    M: int,
) -> dict:
    """Compute intra/inter agreement, histogram, ARIs from the post-HDBSCAN sub partition."""
    triu = np.triu_indices(s, k=1)
    co_triu = co[triu]
    sub_pairs = sub_labels[triu[0]] == sub_labels[triu[1]]
    non_noise_pairs = (sub_labels[triu[0]] != -1) & (sub_labels[triu[1]] != -1)
    intra_mask = sub_pairs & non_noise_pairs
    inter_mask = (~sub_pairs) & non_noise_pairs
    mean_intra = float(co_triu[intra_mask].mean()) if intra_mask.any() else 0.0
    mean_inter = float(co_triu[inter_mask].mean()) if inter_mask.any() else 0.0

    histo_int = np.clip(np.round(co_triu * M).astype(int), 0, M)
    counts = np.bincount(histo_int, minlength=M + 1).tolist()

    return {
        "subsample_fraction": s / n if n > 0 else 1.0,
        "mean_intra_cluster_agreement": mean_intra,
        "mean_inter_cluster_agreement": mean_inter,
        "consensus_separation": mean_intra - mean_inter,
        "agreement_histogram": counts,
        "pairwise_algo_agreement_idx": {
            f"{i}-{j}": v for (i, j), v in pairwise_idx.items()
        },
        "algorithm_consensus_ari_idx": [
            float(adjusted_rand_score(sub_labels, labels_arr[m, sub_idx]))
            for m in range(M)
        ],
    }


def compute_coassociation_labels(
    labels_list: list[np.ndarray],
    *,
    threshold: float,
    min_cluster_size: int,
    max_fit_samples: int,
    random_state: int = 0,
    propagation_confidence_floor: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Consensus clustering via co-association matrix + HDBSCAN(precomputed).

    On a subsample of `max_fit_samples` points: build co-association, run
    HDBSCAN with `cluster_selection_epsilon = 1 - threshold`, then propagate to
    the remaining points via 1-NN highest agreement with the sub-points.
    """
    if not labels_list:
        raise ValueError("compute_coassociation_labels: empty labels_list")
    n = labels_list[0].shape[0]
    if any(l.shape[0] != n for l in labels_list):
        raise ValueError(
            f"inconsistent label-array lengths: {[l.shape[0] for l in labels_list]}"
        )

    labels_arr = np.stack(labels_list)
    M = labels_arr.shape[0]

    s = min(n, max_fit_samples)
    if n > max_fit_samples:
        rng = np.random.default_rng(random_state)
        sub_idx = np.sort(rng.choice(n, s, replace=False))
    else:
        sub_idx = np.arange(n)

    sub = labels_arr[:, sub_idx]
    co, pairwise_idx = _coassociation_matrix(sub)

    dist = (1.0 - co).astype(np.float64)
    np.fill_diagonal(dist, 0.0)
    eps = max(0.0, float(1.0 - threshold))
    mcs = max(2, int(min_cluster_size))
    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=mcs,
        cluster_selection_epsilon=eps,
    )
    sub_labels = clusterer.fit_predict(dist).astype(np.int64)

    diagnostics = _build_diagnostics(labels_arr, sub_idx, sub_labels, co, pairwise_idx, s, n, M)

    full_labels = np.full(n, -1, dtype=np.int64)
    full_labels[sub_idx] = sub_labels
    if n > s:
        non_sub = np.setdiff1d(np.arange(n), sub_idx, assume_unique=True)
        prop_stats = _propagate_labels(
            labels_arr, sub_idx, sub_labels, non_sub,
            propagation_confidence_floor, full_labels,
        )
        diagnostics["propagation_mean_max_agreement"] = prop_stats["mean_max"]
        diagnostics["propagation_low_confidence_ratio"] = prop_stats["low_conf_ratio"]
    else:
        diagnostics["propagation_mean_max_agreement"] = None
        diagnostics["propagation_low_confidence_ratio"] = None

    return full_labels, diagnostics


def make_ensemble_cluster_fn(
    cluster_fns: list[ClusterFn],
    *,
    threshold: float = 0.5,
    min_consensus_size=1,
    max_fit_samples: int = 10_000,
    random_state: int = 0,
    consensus_reporter: ConsensusReporter | None = None,
    propagation_confidence_floor: float = 0.0,
) -> ClusterFn:
    """Compose multiple ClusterFns via co-association + HDBSCAN(precomputed).

    `min_consensus_size` may be `int` or `{rel: float, min?, max?}`; resolved
    against `effective_n = min(N, max_fit_samples)` per call. `consensus_reporter`
    receives the diagnostics dict after each consensus computation.
    """

    def _fn(X_num: np.ndarray, X_cat: np.ndarray | None = None) -> np.ndarray:
        from src.domain.clustering.compose import _resolve_scalar_rel  # lazy to avoid cycle

        labels_list = [fn(X_num, X_cat) for fn in cluster_fns]
        effective_n = min(X_num.shape[0], max_fit_samples)
        mcs = _resolve_scalar_rel(min_consensus_size, effective_n)
        labels, diagnostics = compute_coassociation_labels(
            labels_list,
            threshold=threshold,
            min_cluster_size=mcs,
            max_fit_samples=max_fit_samples,
            random_state=random_state,
            propagation_confidence_floor=propagation_confidence_floor,
        )
        if consensus_reporter is not None:
            consensus_reporter(diagnostics)
        return labels

    return _fn
