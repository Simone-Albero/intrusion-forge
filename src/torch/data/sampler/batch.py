import random
from collections import defaultdict
from typing import Dict, Iterator, List, Optional

import numpy as np
from torch.utils.data import Sampler

from . import SamplerFactory


@SamplerFactory.register()
class ClusterBatchSampler(Sampler[List[int]]):
    """
    Samples clusters and takes samples per cluster for contrastive learning.

    Batch composition:
      core_size = clusters_per_batch * samples_per_cluster
      + include_ambiguous

    Ambiguous samples are those with cluster == ambiguous_tag (default: -1).
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
        min_cluster_size: Optional[int] = None,
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
        self._iter_count = 0

        self.min_cluster_size = (
            samples_per_cluster if min_cluster_size is None else min_cluster_size
        )
        if self.min_cluster_size < self.samples_per_cluster:
            raise ValueError(
                f"min_cluster_size ({self.min_cluster_size}) must be >= "
                f"samples_per_cluster ({self.samples_per_cluster}) to avoid replacement."
            )

        # Validate batch size: core + ambiguous must exactly fit
        self.core_size = self.clusters_per_batch * self.samples_per_cluster
        expected = self.core_size + self.include_ambiguous
        if expected != self.batch_size:
            raise ValueError(
                f"Set batch_size = core_size + include_ambiguous = {expected}. "
                f"Got batch_size={self.batch_size}, core_size={self.core_size}, "
                f"include_ambiguous={self.include_ambiguous}."
            )

        # Build cluster index
        cluster_to_idx: Dict[int, List[int]] = defaultdict(list)
        ambiguous_idx: List[int] = []

        for i, c in enumerate(self.clusters):
            if int(c) == int(ambiguous_tag):
                ambiguous_idx.append(i)
            else:
                cluster_to_idx[int(c)].append(i)

        # Keep only clusters large enough to sample WITHOUT replacement
        self.cluster_to_idx = {
            c: idxs
            for c, idxs in cluster_to_idx.items()
            if len(idxs) >= self.min_cluster_size
        }
        self.ambiguous_idx = ambiguous_idx

        if not self.cluster_to_idx:
            raise ValueError(
                f"No clusters with at least {self.min_cluster_size} samples found."
            )

        # Build class-to-cluster mapping if labels provided (optional class balancing)
        self.class_to_clusters: Dict[int, List[int]] = defaultdict(list)
        if self.labels is not None:
            for c, idxs in self.cluster_to_idx.items():
                majority_class = np.bincount(self.labels[idxs]).argmax()
                self.class_to_clusters[int(majority_class)].append(c)

        # Precompute list of cluster IDs
        self.cluster_ids = list(self.cluster_to_idx.keys())
        if len(self.cluster_ids) < self.clusters_per_batch:
            raise ValueError(
                f"Not enough eligible clusters ({len(self.cluster_ids)}) to sample "
                f"{self.clusters_per_batch} clusters per batch without replacement. "
            )

    def __len__(self) -> int:
        total_core_samples = sum(len(idxs) for idxs in self.cluster_to_idx.values())
        est_batches = total_core_samples // self.core_size
        return est_batches if self.drop_last else est_batches + 1

    def _pick_clusters(
        self,
        rng: random.Random,
        k: int,
        cluster_ids: List[int],
        cluster_idx: int,
    ) -> tuple[List[int], int]:
        """Pick k clusters, optionally balancing across classes."""
        if not self.class_to_clusters:
            # Round-robin without class balancing
            chosen = []
            for _ in range(k):
                if cluster_idx >= len(cluster_ids):
                    rng.shuffle(cluster_ids)
                    cluster_idx = 0
                chosen.append(cluster_ids[cluster_idx])
                cluster_idx += 1
            return chosen, cluster_idx

        # Class-balanced sampling across available classes
        classes = list(self.class_to_clusters.keys())
        rng.shuffle(classes)

        per_class, remainder = divmod(k, len(classes))
        chosen: List[int] = []
        for i, cls in enumerate(classes):
            n = per_class + (1 if i < remainder else 0)
            pool = self.class_to_clusters[cls]
            if not pool:
                continue
            chosen.extend(
                rng.choices(pool, k=n) if len(pool) < n else rng.sample(pool, k=n)
            )

        # Ensure we have k unique clusters in the batch
        # If duplicates happen, fill from global cluster_ids.
        chosen_unique = []
        seen = set()
        for c in chosen:
            if c not in seen:
                chosen_unique.append(c)
                seen.add(c)

        # Fill up if needed
        if len(chosen_unique) < k:
            # iterate through shuffled cluster_ids to add missing ones
            fill_pool = cluster_ids[:]
            rng.shuffle(fill_pool)
            for c in fill_pool:
                if c not in seen:
                    chosen_unique.append(c)
                    seen.add(c)
                    if len(chosen_unique) == k:
                        break

        return chosen_unique[:k], cluster_idx

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self._iter_count)
        self._iter_count += 1

        cluster_ids = self.cluster_ids[:]
        rng.shuffle(cluster_ids)

        cluster_idx = 0
        for _ in range(len(self)):
            # Sample clusters
            chosen_clusters, cluster_idx = self._pick_clusters(
                rng, self.clusters_per_batch, cluster_ids, cluster_idx
            )

            batch: List[int] = []
            used = set()

            for cluster in chosen_clusters:
                idxs = self.cluster_to_idx[cluster]
                samples = rng.sample(idxs, k=self.samples_per_cluster)
                batch.extend(samples)
                used.update(samples)

            if self.include_ambiguous > 0:
                if not self.ambiguous_idx:
                    raise ValueError(
                        "include_ambiguous > 0 but no ambiguous samples found."
                    )

                amb_pool = [i for i in self.ambiguous_idx if i not in used]
                if len(amb_pool) < self.include_ambiguous:
                    raise ValueError(
                        f"Not enough unique ambiguous samples to draw {self.include_ambiguous} "
                        f"(available={len(amb_pool)}). Reduce include_ambiguous or increase data."
                    )

                batch.extend(rng.sample(amb_pool, k=self.include_ambiguous))

            if len(batch) != self.batch_size:
                raise RuntimeError(
                    f"Internal error: batch size {len(batch)} != expected {self.batch_size}"
                )

            yield batch
