import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from scipy.spatial.distance import cdist
from tqdm import tqdm

from src.core.utils import timed


def aggregate_min_mean_max(
    vals: list[float],
) -> tuple[float | None, float | None, float | None]:
    """Aggregate a list of values into (min, mean, max). Returns Nones if empty."""
    if not vals:
        return None, None, None
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.min()), float(arr.mean()), float(arr.max())


def make_null_row(metric_keys: tuple[str, ...]) -> dict[str, float | None]:
    """Null pairwise-output row: `f"{metric}_{stat}": None` for stats min/mean/max."""
    return {f"{m}_{stat}": None for m in metric_keys for stat in ("min", "mean", "max")}


def l2_normalize(X_num: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Row-wise L2-normalize so that Euclidean on unit vectors maps to cosine."""
    norms = np.linalg.norm(X_num, axis=1, keepdims=True)
    return X_num / np.maximum(norms, eps)


def _scale_for_metric(X_num: np.ndarray, metric: str) -> np.ndarray:
    """Numerics in the space `hybrid_row_batch` expects: unit rows, or range-scaled."""
    if metric == "cosine":
        return l2_normalize(X_num)
    feat_ranges = np.maximum(X_num.max(axis=0) - X_num.min(axis=0), 1e-8)
    return X_num / feat_ranges


def hybrid_row_batch(
    X_num_ref: np.ndarray,
    X_cat: np.ndarray | None,
    query_num_ref: np.ndarray,
    query_cat: np.ndarray | None,
    d_num: int,
    d_cat: int,
    *,
    metric: str,
) -> np.ndarray:
    """Gower-hybrid distance on numerics scaled by `_scale_for_metric`, plus Hamming.

    `cdist` sums in a different order than a per-feature loop, so a result can drift
    by a few ulp — N2 is the one downstream measure that reads it, not just order.
    """
    if metric == "cosine":
        dist = cdist(query_num_ref, X_num_ref, metric="sqeuclidean")
        dist *= d_num / 2.0
        # Cosine distance reaches 2 for opposed vectors; the Gower average needs each
        # numeric term in [0, 1], which is d_num after the scaling above.
        np.clip(dist, 0.0, d_num, out=dist)
    else:
        # Range-scaled inputs turn the Gower-Euclidean average into plain Manhattan.
        dist = cdist(query_num_ref, X_num_ref, metric="cityblock")

    if X_cat is not None and d_cat > 0:
        # uint16 costs little next to `dist`, and keeps d_cat (a column count, at
        # most in the tens) far from wrapping, unlike uint8's 256.
        mismatch = np.zeros(dist.shape, dtype=np.uint16)
        for f in range(d_cat):
            mismatch += query_cat[:, f : f + 1] != X_cat[:, f]
        dist += mismatch

    dist /= d_num + d_cat
    return dist


def _thread_budget() -> int:
    """Worker count for `build_knn_graph`: the cgroup/affinity quota, not the host."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


@timed
def build_knn_graph(
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    *,
    k: int,
    metric: str,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a k-NN graph with Gower-hybrid distances, batches spread across threads."""
    n, d_num = X_num.shape
    d_cat = X_cat.shape[1] if X_cat is not None else 0
    effective_k = min(k, n - 1)
    X_ref = _scale_for_metric(X_num, metric)

    indices = np.empty((n, effective_k), dtype=np.int64)
    distances = np.empty((n, effective_k), dtype=np.float64)

    def fill_batch(start: int) -> None:
        end = min(start + batch_size, n)
        q_cat = X_cat[start:end] if X_cat is not None else None
        dists = hybrid_row_batch(
            X_ref, X_cat, X_ref[start:end], q_cat, d_num, d_cat, metric=metric
        )
        batch_idx = np.arange(end - start)
        dists[batch_idx, start + batch_idx] = np.inf

        part = np.argpartition(dists, effective_k, axis=1)[:, :effective_k]
        part_d = np.take_along_axis(dists, part, axis=1)
        order = np.argsort(part_d, axis=1)
        indices[start:end] = np.take_along_axis(part, order, axis=1)
        distances[start:end] = np.take_along_axis(part_d, order, axis=1)

    starts = list(range(0, n, batch_size))
    # cdist and argpartition release the GIL, so threads overlap real work; each
    # fills its own slice of `indices`/`distances`, with nothing to lock between them.
    with ThreadPoolExecutor(max_workers=_thread_budget()) as pool:
        futures = [pool.submit(fill_batch, start) for start in starts]
        try:
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="k-NN graph",
                unit="batch",
                leave=False,
            ):
                future.result()
        except BaseException:
            # Batches not yet started can still be dropped; ones already running finish
            # regardless, since they only write to their own slice.
            for pending in futures:
                pending.cancel()
            raise

    return indices, distances


def _to_sparse_csr(
    indices: np.ndarray, distances: np.ndarray, n: int
) -> scipy.sparse.csr_matrix:
    """Symmetrise the directed k-NN graph keeping the minimum distance per edge."""
    k = indices.shape[1]
    rows = np.repeat(np.arange(n), k)
    cols = indices.ravel()
    data = distances.ravel()

    sym_rows = np.concatenate([rows, cols])
    sym_cols = np.concatenate([cols, rows])
    sym_data = np.concatenate([data, data])

    order = np.lexsort((sym_data, sym_cols, sym_rows))
    sym_rows = sym_rows[order]
    sym_cols = sym_cols[order]
    sym_data = sym_data[order]

    unique_mask = np.ones(len(sym_rows), dtype=bool)
    unique_mask[1:] = (sym_rows[1:] != sym_rows[:-1]) | (sym_cols[1:] != sym_cols[:-1])

    mat = scipy.sparse.csr_matrix(
        (sym_data[unique_mask], (sym_rows[unique_mask], sym_cols[unique_mask])),
        shape=(n, n),
    )
    mat.setdiag(0)
    mat.eliminate_zeros()
    return mat


def _bridge_disconnected(
    mat: scipy.sparse.csr_matrix,
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    d_num: int,
    d_cat: int,
    *,
    metric: str,
) -> scipy.sparse.csr_matrix:
    """Add one bridge edge per disconnected component of the k-NN graph."""
    n_comp, comp_labels = scipy.sparse.csgraph.connected_components(mat, directed=False)
    if n_comp == 1:
        return mat

    mat = mat.tolil()
    ref = int(np.where(comp_labels == 0)[0][0])
    X_ref = _scale_for_metric(X_num, metric)
    q_cat = X_cat[ref : ref + 1] if X_cat is not None else None
    dists_row = hybrid_row_batch(
        X_ref, X_cat, X_ref[ref : ref + 1], q_cat, d_num, d_cat, metric=metric
    )[0]

    for ci in range(1, n_comp):
        nodes_ci = np.where(comp_labels == ci)[0]
        best_j = int(nodes_ci[dists_row[nodes_ci].argmin()])
        d = max(float(dists_row[best_j]), 1e-10)
        mat[ref, best_j] = d
        mat[best_j, ref] = d
        comp_labels[nodes_ci] = 0
    return mat.tocsr()


def build_approx_mst(
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    *,
    metric: str,
) -> np.ndarray:
    """Approximate MST on the sparse k-NN graph, bridging disconnected components first."""
    n, d_num = X_num.shape
    d_cat = X_cat.shape[1] if X_cat is not None else 0

    graph = _to_sparse_csr(knn_indices, knn_distances, n)
    graph = _bridge_disconnected(graph, X_num, X_cat, d_num, d_cat, metric=metric)
    mst = scipy.sparse.csgraph.minimum_spanning_tree(graph).tocoo()
    if mst.nnz == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.column_stack((mst.row, mst.col)).astype(np.int64, copy=False)


def topk_adversarial_clusters(
    centroid_matrix: np.ndarray,
    cluster_ids: list[str],
    id_to_class: dict[str, int],
    *,
    top_k: int,
    metric: str,
) -> dict[str, list[str]]:
    """Top-K nearest cluster ids of a different class, by ascending centroid distance."""
    pw = cdist(centroid_matrix, centroid_matrix, metric=metric)
    np.fill_diagonal(pw, np.inf)
    classes = np.array([id_to_class[cid] for cid in cluster_ids], dtype=np.int64)
    out: dict[str, list[str]] = {}
    for i, cid in enumerate(cluster_ids):
        adv_idx = np.where(classes != classes[i])[0]
        order = np.argsort(pw[i, adv_idx])
        out[cid] = [cluster_ids[int(adv_idx[j])] for j in order[:top_k]]
    return out
