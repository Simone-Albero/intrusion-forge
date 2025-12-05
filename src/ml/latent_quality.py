import numpy as np
from sklearn.neighbors import NearestNeighbors


def trustworthiness(
    X: np.ndarray,
    X_embedded: np.ndarray,
    n_neighbors: int = 5,
    metric: str = "euclidean",
) -> float:
    """
    Compute the trustworthiness score of a dimensionality reduction.

    Trustworthiness measures whether points that are close in the embedded space
    are also close in the original space. High trustworthiness means the embedded
    space doesn't introduce false neighbors.

    Args:
        X: Original high-dimensional data of shape (n_samples, n_features)
        X_embedded: Embedded low-dimensional data of shape (n_samples, n_components)
        n_neighbors: Number of neighbors to consider
        metric: Distance metric to use

    Returns:
        Trustworthiness score in range [0, 1], where 1 is perfect
    """
    n_samples = X.shape[0]

    # Find neighbors in original space
    nbrs_original = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
    nbrs_original.fit(X)
    indices_original = nbrs_original.kneighbors(X, return_distance=False)[:, 1:]

    # Find neighbors in embedded space
    nbrs_embedded = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
    nbrs_embedded.fit(X_embedded)
    indices_embedded = nbrs_embedded.kneighbors(X_embedded, return_distance=False)[
        :, 1:
    ]

    # Compute ranks in original space
    ranks = np.zeros((n_samples, n_samples), dtype=int)
    for i in range(n_samples):
        ranks[i, indices_original[i]] = np.arange(1, n_neighbors + 1)

    # Compute trustworthiness
    trust = 0.0
    for i in range(n_samples):
        for j in indices_embedded[i]:
            if j not in indices_original[i]:
                trust += max(0, ranks[i, j] - n_neighbors)

    normalizer = n_samples * n_neighbors * (2 * n_samples - 3 * n_neighbors - 1) / 2
    if normalizer > 0:
        trust = 1.0 - (2.0 * trust / normalizer)
    else:
        trust = 1.0

    return trust


def continuity(
    X: np.ndarray,
    X_embedded: np.ndarray,
    n_neighbors: int = 5,
    metric: str = "euclidean",
) -> float:
    """
    Compute the continuity score of a dimensionality reduction.

    Continuity measures whether points that are close in the original space
    are also close in the embedded space. High continuity means the embedded
    space doesn't tear apart neighborhoods.

    Args:
        X: Original high-dimensional data of shape (n_samples, n_features)
        X_embedded: Embedded low-dimensional data of shape (n_samples, n_components)
        n_neighbors: Number of neighbors to consider
        metric: Distance metric to use

    Returns:
        Continuity score in range [0, 1], where 1 is perfect
    """
    n_samples = X.shape[0]

    # Find neighbors in original space
    nbrs_original = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
    nbrs_original.fit(X)
    indices_original = nbrs_original.kneighbors(X, return_distance=False)[:, 1:]

    # Find neighbors in embedded space
    nbrs_embedded = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
    nbrs_embedded.fit(X_embedded)
    indices_embedded = nbrs_embedded.kneighbors(X_embedded, return_distance=False)[
        :, 1:
    ]

    # Compute ranks in embedded space
    ranks = np.zeros((n_samples, n_samples), dtype=int)
    for i in range(n_samples):
        ranks[i, indices_embedded[i]] = np.arange(1, n_neighbors + 1)

    # Compute continuity
    cont = 0.0
    for i in range(n_samples):
        for j in indices_original[i]:
            if j not in indices_embedded[i]:
                cont += max(0, ranks[i, j] - n_neighbors)

    normalizer = n_samples * n_neighbors * (2 * n_samples - 3 * n_neighbors - 1) / 2
    if normalizer > 0:
        cont = 1.0 - (2.0 * cont / normalizer)
    else:
        cont = 1.0

    return cont


def local_distance_consistency(
    X: np.ndarray,
    X_embedded: np.ndarray,
    n_neighbors: int = 10,
    metric: str = "euclidean",
) -> float:
    """
    Compute local distance consistency between original and embedded spaces.

    Measures how well the relative distances between neighbors are preserved
    in the latent space. Returns the Spearman correlation between distances.

    Args:
        X: Original high-dimensional data of shape (n_samples, n_features)
        X_embedded: Embedded low-dimensional data of shape (n_samples, n_components)
        n_neighbors: Number of neighbors to consider
        metric: Distance metric to use

    Returns:
        Distance consistency score (Spearman correlation), range [-1, 1]
    """
    from scipy.stats import spearmanr

    n_samples = X.shape[0]

    # Find neighbors in original space
    nbrs_original = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
    nbrs_original.fit(X)
    distances_orig, indices_orig = nbrs_original.kneighbors(X)

    # Get corresponding distances in embedded space
    distances_list_orig = []
    distances_list_embed = []

    for i in range(n_samples):
        # Get distances to neighbors in original space
        orig_dists = distances_orig[i, 1:]  # Exclude self

        # Get distances to same points in embedded space
        neighbor_indices = indices_orig[i, 1:]
        embed_dists = np.linalg.norm(
            X_embedded[i] - X_embedded[neighbor_indices], axis=1
        )

        distances_list_orig.extend(orig_dists)
        distances_list_embed.extend(embed_dists)

    # Compute Spearman correlation
    correlation, _ = spearmanr(distances_list_orig, distances_list_embed)

    return correlation


def reconstruction_latent_correlation(
    X_embedded: np.ndarray,
    reconstruction_errors: np.ndarray,
    metric: str = "euclidean",
) -> float:
    """
    Measure correlation between latent space density and reconstruction quality.

    Ideally, points in dense regions of latent space should have better
    reconstruction (lower error). This metric measures this relationship.

    Args:
        X_embedded: Embedded latent representations of shape (n_samples, n_components)
        reconstruction_errors: Per-sample reconstruction errors of shape (n_samples,)
        metric: Distance metric for computing latent space density

    Returns:
        Pearson correlation coefficient between local density and reconstruction error
    """
    from scipy.stats import pearsonr

    # Compute local density in latent space (inverse of mean distance to k neighbors)
    n_neighbors = min(10, len(X_embedded) - 1)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
    nbrs.fit(X_embedded)
    distances, _ = nbrs.kneighbors(X_embedded)

    # Local density (inverse of average distance to neighbors)
    local_density = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-10)

    # Compute correlation (negative correlation is good: high density -> low error)
    correlation, _ = pearsonr(local_density, reconstruction_errors)

    # Return negative correlation (so positive is better)
    return -correlation
