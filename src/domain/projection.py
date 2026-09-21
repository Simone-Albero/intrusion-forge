import numpy as np
from sklearn.manifold import TSNE

_TSNE_MIN_PERPLEXITY = 5
_TSNE_MAX_PERPLEXITY = 30


def stratified_subsample(
    labels: np.ndarray,
    *,
    n_samples: int | None = None,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Sample up to `n_samples` indices, capped equally per group."""
    n = len(labels)
    if n_samples is None or n_samples >= n:
        return np.arange(n)

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    unique_groups, counts = np.unique(labels, return_counts=True)
    per_group = min(n_samples // len(unique_groups), int(counts.min()))

    parts = []
    for g in unique_groups:
        pool = np.where(labels == g)[0]
        parts.append(rng.choice(pool, min(per_group, len(pool)), replace=False))

    return np.concatenate(parts) if parts else np.array([], dtype=int)


def tsne_projection(
    data: np.ndarray,
    *,
    n_components: int = 2,
    perplexity: int | None = None,
    random_state: int = 42,
) -> np.ndarray:
    """Project data to lower dimensions using t-SNE with adaptive perplexity."""
    if perplexity is None:
        perplexity = max(
            _TSNE_MIN_PERPLEXITY,
            min(_TSNE_MAX_PERPLEXITY, (len(data) - 1) // 3),
        )
    return TSNE(
        n_components=n_components, perplexity=perplexity, random_state=random_state
    ).fit_transform(data)
