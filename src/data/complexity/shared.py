import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from scipy.spatial.distance import cdist

from ...common.utils import timed


def _iqr_scaling(X_num: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale numerics by IQR (Q3 - Q1) for outlier-robust Gower distance.

    Replaces the classic max-min range so that heavy-tailed features (byte
    counts, durations) are not collapsed by a few outliers.
    """
    q1 = np.quantile(X_num, 0.25, axis=0)
    q3 = np.quantile(X_num, 0.75, axis=0)
    return X_num / np.maximum(q3 - q1, eps)


def aggregate_min_mean_max(
    vals: list[float],
) -> tuple[float | None, float | None, float | None]:
    """Aggregate a list of values into (min, mean, max). Returns Nones if empty."""
    if not vals:
        return None, None, None
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.min()), float(arr.mean()), float(arr.max())


def make_null_row(metric_keys: tuple[str, ...]) -> dict[str, float | None]:
    """Build a dict of `f"{metric}_{scope}_{stat}": None` for the standard
    pairwise output schema (scopes class+cluster, stats min/mean/max)."""
    return {
        f"{m}_{scope}_{stat}": None
        for m in metric_keys
        for scope in ("class", "cluster")
        for stat in ("min", "mean", "max")
    }


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
    Numerics are pre-scaled by IQR once before the batch loop.
    """
    n = X_num.shape[0]
    effective_k = min(k, n - 1)

    X_num_scaled = _iqr_scaling(X_num)
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
) -> scipy.sparse.csr_matrix:
    """Add one cross-component bridge edge per disconnected component.

    Used when the k-NN graph is disconnected (categoricals can partition the
    space such that the k neighbours of every node sit in the same category).
    """
    n_comp, comp_labels = scipy.sparse.csgraph.connected_components(
        mat, directed=False
    )
    if n_comp == 1:
        return mat

    X_num_scaled = _iqr_scaling(X_num)
    d_cat = X_cat.shape[1] if X_cat is not None else 0
    d_total = X_num.shape[1] + d_cat
    mat = mat.tolil()
    ref = int(np.where(comp_labels == 0)[0][0])
    q_num_scaled = X_num_scaled[ref : ref + 1]
    q_cat = X_cat[ref : ref + 1] if X_cat is not None else None
    dists_row = _gower_row_batch(X_num_scaled, X_cat, q_num_scaled, q_cat, d_total)[0]
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
) -> np.ndarray:
    """Approximate MST on the sparse k-NN graph. Returns an (E, 2) int64 array
    of MST edges (row, col). One bridge edge is added per disconnected
    component so the MST connects every cluster.
    """
    n = knn_indices.shape[0]
    graph = _to_sparse_csr(knn_indices, knn_distances, n)
    graph = _bridge_disconnected(graph, X_num, X_cat)
    mst = scipy.sparse.csgraph.minimum_spanning_tree(graph).tocoo()
    if mst.nnz == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.column_stack((mst.row, mst.col)).astype(np.int64, copy=False)


def topk_adversarial_clusters(
    centroid_matrix: np.ndarray,
    cluster_ids: list[str],
    id_to_class: dict[str, int],
    top_k: int,
) -> dict[str, list[str]]:
    """For each cluster, return the top-K nearest cluster IDs of a different class.

    Uses Euclidean distances on the precomputed centroid matrix. Result is
    sorted by ascending centroid distance and capped at `top_k` (or all
    available adversarial clusters if fewer).
    """
    if centroid_matrix.shape[0] == 0:
        return {}
    pw = cdist(centroid_matrix, centroid_matrix, metric="euclidean")
    np.fill_diagonal(pw, np.inf)
    classes = np.array(
        [id_to_class.get(cid, -1) for cid in cluster_ids], dtype=np.int64
    )
    out: dict[str, list[str]] = {}
    for i, cid in enumerate(cluster_ids):
        cls_c = id_to_class.get(cid)
        if cls_c is None:
            out[cid] = []
            continue
        adv_idx = np.where(classes != cls_c)[0]
        if adv_idx.size == 0:
            out[cid] = []
            continue
        order = np.argsort(pw[i, adv_idx])
        take = min(top_k, adv_idx.size)
        out[cid] = [cluster_ids[int(adv_idx[j])] for j in order[:take]]
    return out
