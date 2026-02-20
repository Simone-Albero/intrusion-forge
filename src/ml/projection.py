import numpy as np
from sklearn.manifold import TSNE


def create_subsample_mask(
    labels: np.ndarray,
    n_samples: int | None = None,
    stratify: bool = True,
) -> np.ndarray:
    """Generate a boolean mask for subsampling data.

    If n_samples is None or >= dataset size, all samples are kept.
    If stratify=True, samples proportionally to class distribution.
    If stratify=False, samples an equal number of samples per class.
    """
    if n_samples is None or n_samples >= len(labels):
        return np.ones(len(labels), dtype=bool)

    mask = np.zeros(len(labels), dtype=bool)
    unique_labels = np.unique(labels)

    if stratify:
        label_counts = np.array([np.sum(labels == l) for l in unique_labels])
        samples_per_class = np.round(label_counts / len(labels) * n_samples).astype(int)
        # Fix rounding so total == n_samples
        samples_per_class[np.argmax(samples_per_class)] += (
            n_samples - samples_per_class.sum()
        )
    else:
        per_class = n_samples // len(unique_labels)
        samples_per_class = [per_class] * len(unique_labels)

    for label, k in zip(unique_labels, samples_per_class):
        class_indices = np.where(labels == label)[0]
        selected = np.random.choice(
            class_indices, size=min(k, len(class_indices)), replace=False
        )
        mask[selected] = True

    return mask


def tsne_projection(
    data: np.ndarray, n_components: int = 2, perplexity: int | None = None
) -> np.ndarray:
    """Project data to 2D using t-SNE with adaptive perplexity."""
    if perplexity is None:
        perplexity = max(5, min(30, (len(data) - 1) // 3))
    return TSNE(
        n_components=n_components, perplexity=perplexity, random_state=42
    ).fit_transform(data)
