import random
from collections import defaultdict
from typing import Dict, Iterator, List, Optional

import numpy as np
from torch.utils.data import Sampler

from . import SamplerFactory


@SamplerFactory.register()
class ClusterBatchSampler(Sampler[List[int]]):
    """Samples clusters and takes samples per cluster for contrastive learning.

    Positives are defined as samples from the same cluster. Optionally includes
    ambiguous samples (cluster == -1) for cross-entropy loss.
    """

    def __init__(
        self,
        clusters: np.ndarray,
        labels: Optional[np.ndarray] = None,
        batch_size: int = 128,
        clusters_per_batch: int = 3,
        samples_per_cluster: int = 2,
        include_ambiguous: int = 0,
        ambiguous_tag: int = -1,
        drop_last: bool = True,
        min_cluster_size: int = 2,
        seed: int = 0,
    ):
        self.clusters = np.asarray(clusters)
        self.labels = None if labels is None else np.asarray(labels)
        self.batch_size = batch_size
        self.clusters_per_batch = clusters_per_batch
        self.samples_per_cluster = samples_per_cluster
        self.include_ambiguous = include_ambiguous
        self.drop_last = drop_last
        self.seed = seed

        # Build cluster index
        cluster_to_idx: Dict[int, List[int]] = defaultdict(list)
        ambiguous_idx: List[int] = []

        for i, c in enumerate(self.clusters):
            if c == ambiguous_tag:
                ambiguous_idx.append(i)
            else:
                cluster_to_idx[int(c)].append(i)

        self.cluster_to_idx = {
            c: idxs
            for c, idxs in cluster_to_idx.items()
            if len(idxs) >= min_cluster_size
        }
        self.ambiguous_idx = ambiguous_idx

        if not self.cluster_to_idx:
            raise ValueError(
                f"No clusters with at least {min_cluster_size} samples found."
            )

        # Build class-to-cluster mapping if labels provided
        self.class_to_clusters: Dict[int, List[int]] = defaultdict(list)
        if self.labels is not None:
            for c, idxs in self.cluster_to_idx.items():
                majority_class = np.bincount(self.labels[idxs]).argmax()
                self.class_to_clusters[majority_class].append(c)

        # Validate batch size
        core_size = self.clusters_per_batch * self.samples_per_cluster
        if core_size + self.include_ambiguous > self.batch_size:
            raise ValueError(
                f"Batch size {self.batch_size} too small for "
                f"{core_size} core + {self.include_ambiguous} ambiguous samples."
            )

        self.core_size = core_size
        self.leftover = self.batch_size - core_size - self.include_ambiguous

    def __len__(self) -> int:
        total_samples = sum(len(idxs) for idxs in self.cluster_to_idx.values())
        estimated_batches = total_samples // self.core_size
        return estimated_batches if self.drop_last else estimated_batches + 1

    def _sample_or_choice(
        self, rng: random.Random, population: List[int], k: int
    ) -> List[int]:
        """Sample k items from population, using replacement if needed."""
        if len(population) >= k:
            return rng.sample(population, k=k)
        return rng.choices(population, k=k)

    def _pick_clusters(self, rng: random.Random, k: int) -> List[int]:
        """Pick k clusters, optionally balancing across classes."""
        cluster_ids = list(self.cluster_to_idx.keys())

        if not self.class_to_clusters:
            return self._sample_or_choice(rng, cluster_ids, k)

        # Balance clusters across classes
        chosen = []
        classes = list(self.class_to_clusters.keys())
        per_class = k // len(classes)

        for cls in classes:
            cls_clusters = self.class_to_clusters[cls]
            chosen.extend(self._sample_or_choice(rng, cls_clusters, per_class))

        # Fill remaining slots
        if len(chosen) < k:
            remaining = [c for c in cluster_ids if c not in chosen]
            chosen.extend(self._sample_or_choice(rng, remaining, k - len(chosen)))

        return chosen

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed)
        all_core_indices = [
            idx for idxs in self.cluster_to_idx.values() for idx in idxs
        ]

        while True:
            batch: List[int] = []

            # Sample from clusters
            chosen_clusters = self._pick_clusters(rng, self.clusters_per_batch)
            for c in chosen_clusters:
                idxs = self.cluster_to_idx[c]
                batch.extend(
                    self._sample_or_choice(rng, idxs, self.samples_per_cluster)
                )

            # Fill remaining slots
            if self.leftover > 0:
                batch.extend(
                    self._sample_or_choice(rng, all_core_indices, self.leftover)
                )

            # Add ambiguous samples
            if self.include_ambiguous > 0 and self.ambiguous_idx:
                batch.extend(
                    self._sample_or_choice(
                        rng, self.ambiguous_idx, self.include_ambiguous
                    )
                )

            yield batch
