import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from scipy.spatial.distance import cdist

from ...common.utils import timed


def _gower_row_batch(
    X_num_scaled: np.ndarray,
    X_cat: np.ndarray | None,
    query_num_scaled: np.ndarray,
    query_cat: np.ndarray | None,
    d_total: int,
) -> np.ndarray:
    """Batched Gower distance: L1 on pre-scaled numerics, 0/1 for categoricals.

    Numerics computed via scipy cdist (C-level L1 on pre-scaled arrays).
    Categoricals iterated feature-by-feature to keep memory at O(b × n).
    Returns shape (b, n) distance matrix.
    """
    dist = cdist(query_num_scaled, X_num_scaled, metric="cityblock")

    if X_cat is not None and X_cat.shape[1] > 0:
        for f in range(X_cat.shape[1]):
            dist += (query_cat[:, f : f + 1] != X_cat[:, f]).astype(np.float64)

    return dist / d_total


@timed
def build_knn_graph(
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    k: int,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Build k-NN graph via batched Gower distance.

    Returns (indices, distances), both shape (n, k).
    Never materialises the full n×n matrix.
    Numerics are pre-scaled once before the batch loop.
    """
    n = X_num.shape[0]
    effective_k = min(k, n - 1)

    num_ranges = X_num.max(axis=0) - X_num.min(axis=0)
    safe_ranges = np.where(num_ranges > 0, num_ranges, 1.0)
    X_num_scaled = X_num / safe_ranges  # precompute once outside the loop
    d_cat = X_cat.shape[1] if X_cat is not None else 0
    d_total = X_num.shape[1] + d_cat

    indices = np.empty((n, effective_k), dtype=np.int64)
    distances = np.empty((n, effective_k), dtype=np.float64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        q_num_scaled = X_num_scaled[start:end]
        q_cat = X_cat[start:end] if X_cat is not None else None

        dists = _gower_row_batch(X_num_scaled, X_cat, q_num_scaled, q_cat, d_total)

        # exclude self-distances
        batch_idx = np.arange(end - start)
        dists[batch_idx, start + batch_idx] = np.inf

        part = np.argpartition(dists, effective_k, axis=1)[:, :effective_k]
        part_d = np.take_along_axis(dists, part, axis=1)
        order = np.argsort(part_d, axis=1)

        indices[start:end] = np.take_along_axis(part, order, axis=1)
        distances[start:end] = np.take_along_axis(part_d, order, axis=1)

    return indices, distances


def to_sparse_csr(
    indices: np.ndarray,
    distances: np.ndarray,
    n: int,
) -> scipy.sparse.csr_matrix:
    """Symmetrize k-NN graph to CSR matrix for HDBSCAN metric='precomputed'.

    For each pair (i, j) appearing in the graph, keeps the minimum distance.
    Diagonal is 0.
    """
    k = indices.shape[1]
    rows = np.repeat(np.arange(n), k)
    cols = indices.ravel()
    data = distances.ravel()

    # include both directed edges; sort ascending by data so first duplicate = min
    sym_rows = np.concatenate([rows, cols])
    sym_cols = np.concatenate([cols, rows])
    sym_data = np.concatenate([data, data])

    order = np.lexsort((sym_data, sym_cols, sym_rows))
    sym_rows = sym_rows[order]
    sym_cols = sym_cols[order]
    sym_data = sym_data[order]

    # keep only first (minimum) entry per (row, col) pair
    unique_mask = np.ones(len(sym_rows), dtype=bool)
    unique_mask[1:] = (sym_rows[1:] != sym_rows[:-1]) | (sym_cols[1:] != sym_cols[:-1])

    mat = scipy.sparse.csr_matrix(
        (sym_data[unique_mask], (sym_rows[unique_mask], sym_cols[unique_mask])),
        shape=(n, n),
    )
    mat.setdiag(0)
    mat.eliminate_zeros()
    return mat


def build_approx_mst(
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
) -> list[tuple[int, int, float]]:
    """Approximate MST on the sparse k-NN graph. Used by N1."""
    n = knn_indices.shape[0]
    graph = to_sparse_csr(knn_indices, knn_distances, n)
    mst = scipy.sparse.csgraph.minimum_spanning_tree(graph)
    cx = mst.tocoo()
    return list(zip(cx.row.tolist(), cx.col.tolist(), cx.data.tolist()))


def build_sparse_knn_matrix(
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    k: int,
    batch_size: int = 256,
) -> scipy.sparse.csr_matrix:
    """Build sparse CSR kNN matrix for HDBSCAN. Wrapper: build_knn_graph → to_sparse_csr.

    Ensures graph connectivity (required by HDBSCAN metric='precomputed').
    When categorical features partition the space, k neighbors may all belong to
    the same category, leaving other categories unreachable. Strategy:
      1. Build with requested k. If connected, return immediately.
      2. Retry with k *= 2 up to 3 times (k capped at n-1).
      3. If still disconnected, add one bridge edge per component (last resort).
    """
    n = X_num.shape[0]
    current_k = k

    for _ in range(4):  # up to 3 doublings + original
        knn_indices, knn_distances = build_knn_graph(
            X_num, X_cat, current_k, batch_size
        )
        mat = to_sparse_csr(knn_indices, knn_distances, n)
        n_comp, comp_labels = scipy.sparse.csgraph.connected_components(
            mat, directed=False
        )
        if n_comp == 1:
            return mat
        next_k = min(current_k * 2, n - 1)
        if next_k == current_k:
            break  # already at max k, no point retrying
        current_k = next_k

    # last resort: add one cross-component bridge edge per disconnected component
    num_ranges = X_num.max(axis=0) - X_num.min(axis=0)
    safe_ranges = np.where(num_ranges > 0, num_ranges, 1.0)
    X_num_scaled = X_num / safe_ranges
    d_cat = X_cat.shape[1] if X_cat is not None else 0
    d_total = X_num.shape[1] + d_cat
    mat = mat.tolil()
    for ci in range(1, n_comp):
        nodes_ci = np.where(comp_labels == ci)[0]
        ref = int(np.where(comp_labels == 0)[0][0])
        q_num_scaled = X_num_scaled[ref : ref + 1]
        q_cat = X_cat[ref : ref + 1] if X_cat is not None else None
        dists_row = _gower_row_batch(X_num_scaled, X_cat, q_num_scaled, q_cat, d_total)[
            0
        ]
        best_j = int(nodes_ci[dists_row[nodes_ci].argmin()])
        d = max(float(dists_row[best_j]), 1e-10)
        mat[ref, best_j] = d
        mat[best_j, ref] = d
        comp_labels[nodes_ci] = 0
    return mat.tocsr()
