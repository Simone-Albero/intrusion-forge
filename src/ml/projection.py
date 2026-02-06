from typing import Optional

import numpy as np
from sklearn.manifold import TSNE


def create_subsample_mask(
    labels: np.ndarray,
    n_samples: Optional[int] = None,
    stratify: bool = True,
) -> np.ndarray:
    """
    Generate a boolean mask for subsampling data based on labels.
    If n_samples is None or greater than the dataset size, return a mask with all True values.

    If stratify is True, samples proportionally to class distribution.
    If stratify is False, samples the same number of samples per class.

    Returns:
        np.ndarray: Boolean mask indicating which samples to keep.
    """
    mask = np.zeros(len(labels), dtype=bool)

    if n_samples is None or n_samples >= len(labels):
        mask[:] = True
        return mask

    if stratify:
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        label_proportions = label_counts / len(labels)
        samples_per_class = np.round(label_proportions * n_samples).astype(int)

        diff = n_samples - samples_per_class.sum()
        if diff != 0:
            largest_class_idx = np.argmax(samples_per_class)
            samples_per_class[largest_class_idx] += diff

        for label, n_class_samples in zip(unique_labels, samples_per_class):
            class_indices = np.where(labels == label)[0]
            if n_class_samples > len(class_indices):
                selected = class_indices
            else:
                selected = np.random.choice(
                    class_indices, size=n_class_samples, replace=False
                )
            mask[selected] = True
    else:
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        n_classes = len(unique_labels)
        samples_per_class = n_samples // n_classes

        for label in unique_labels:
            class_indices = np.where(labels == label)[0]
            n_class_samples = min(samples_per_class, len(class_indices))
            selected = np.random.choice(
                class_indices, size=n_class_samples, replace=False
            )
            mask[selected] = True

    return mask


def tsne_projection(data, perplexity=None):
    """Project data using t-SNE with adaptive perplexity."""
    n_samples = len(data)

    if perplexity is None:
        perplexity = min(30, (n_samples - 1) // 3)
        perplexity = max(5, perplexity)

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    projected_data = tsne.fit_transform(data)
    return projected_data
