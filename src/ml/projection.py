from typing import Optional

import numpy as np
from sklearn.manifold import TSNE


def subsample_data_and_labels(
    data: np.ndarray,
    labels: np.ndarray,
    n_samples: Optional[int] = None,
    stratify: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Subsample the data and labels to the specified number of samples.
    If n_samples is None or greater than the dataset size, return the original data and labels.
    """
    if n_samples is None or n_samples >= data.shape[0]:
        return data, labels

    if stratify:
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        label_proportions = label_counts / len(labels)

        # Calculate samples per class
        samples_per_class = np.round(label_proportions * n_samples).astype(int)

        # Adjust to ensure we get exactly n_samples
        diff = n_samples - samples_per_class.sum()
        if diff != 0:
            # Add/remove samples from the largest class
            largest_class_idx = np.argmax(samples_per_class)
            samples_per_class[largest_class_idx] += diff

        # Sample from each class
        indices = []
        for label, n_class_samples in zip(unique_labels, samples_per_class):
            class_indices = np.where(labels == label)[0]
            if n_class_samples > len(class_indices):
                # If requesting more samples than available, take all
                selected = class_indices
            else:
                selected = np.random.choice(
                    class_indices, size=n_class_samples, replace=False
                )
            indices.extend(selected)

        indices = np.array(indices)
        np.random.shuffle(indices)
        return data[indices], labels[indices]
    else:
        indices = np.random.choice(data.shape[0], size=n_samples, replace=False)
        return data[indices], labels[indices]


def tsne_projection(
    data: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    """
    Apply t-SNE projection to reduce the dimensionality of the data.
    """
    tsne = TSNE(n_components=n_components, random_state=random_state)
    projected_data = tsne.fit_transform(data)
    return projected_data
