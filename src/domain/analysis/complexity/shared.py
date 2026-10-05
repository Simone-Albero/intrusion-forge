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


def l2_normalize(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Row-wise L2-normalize so that Euclidean on unit vectors maps to cosine."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def scale_for_metric(X: np.ndarray, metric: str) -> np.ndarray:
    """Points as the graph measures them: unit rows under cosine, as they are otherwise."""
    return l2_normalize(X) if metric == "cosine" else X


def _thread_budget() -> int:
    """Worker count for `build_knn_graph`: the cgroup/affinity quota, not the host."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def nearest_neighbors(
    nodes: np.ndarray,
    node_rows: np.ndarray,
    points: np.ndarray,
    point_rows: np.ndarray,
    *,
    k: int,
    metric: str,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """The k nearest graph nodes of every point, as indices into `nodes`."""
    # A point that is a node, told by its row, is not its own neighbour.
    n = points.shape[0]
    effective_k = min(k, nodes.shape[0] - 1)
    X_nodes = scale_for_metric(nodes, metric)
    X_points = scale_for_metric(points, metric)

    indices = np.empty((n, effective_k), dtype=np.int64)
    distances = np.empty((n, effective_k), dtype=np.float64)

    def fill_batch(start: int) -> None:
        end = min(start + batch_size, n)
        dists = cdist(X_points[start:end], X_nodes, metric="euclidean")
        dists[node_rows[None, :] == point_rows[start:end, None]] = np.inf

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
                desc="k-NN",
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


@timed
def build_knn_graph(
    X: np.ndarray, *, k: int, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """The euclidean k-NN graph of a point cloud: every point's k nearest others."""
    rows = np.arange(X.shape[0])
    return nearest_neighbors(X, rows, X, rows, k=k, metric=metric)


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
    mat: scipy.sparse.csr_matrix, X: np.ndarray, *, metric: str
) -> scipy.sparse.csr_matrix:
    """Add one bridge edge per disconnected component of the k-NN graph."""
    n_comp, comp_labels = scipy.sparse.csgraph.connected_components(mat, directed=False)
    if n_comp == 1:
        return mat

    mat = mat.tolil()
    ref = int(np.where(comp_labels == 0)[0][0])
    X_ref = scale_for_metric(X, metric)
    dists_row = cdist(X_ref[ref : ref + 1], X_ref, metric="euclidean")[0]

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
    X: np.ndarray,
    *,
    metric: str,
) -> np.ndarray:
    """Approximate MST on the sparse k-NN graph, bridging disconnected components first."""
    graph = _to_sparse_csr(knn_indices, knn_distances, X.shape[0])
    graph = _bridge_disconnected(graph, X, metric=metric)
    mst = scipy.sparse.csgraph.minimum_spanning_tree(graph).tocoo()
    if mst.nnz == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.column_stack((mst.row, mst.col)).astype(np.int64, copy=False)


def nearest_rivals(
    centroids: dict[int, np.ndarray],
    population_class: dict[int, int],
    *,
    top_k: int,
    metric: str,
) -> dict[int, list[int]]:
    """Each population's `top_k` nearest rivals, of another class, by centroid."""
    ids = list(population_class)
    matrix = np.stack([np.asarray(centroids[pid], dtype=np.float64) for pid in ids])
    pw = cdist(matrix, matrix, metric=metric)
    np.fill_diagonal(pw, np.inf)
    classes = np.array([population_class[pid] for pid in ids], dtype=np.int64)
    out: dict[int, list[int]] = {}
    for i, pid in enumerate(ids):
        rival_idx = np.where(classes != classes[i])[0]
        order = np.argsort(pw[i, rival_idx])
        out[pid] = [ids[int(rival_idx[j])] for j in order[:top_k]]
    return out
