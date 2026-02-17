import random
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Sequence

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
        self.class_to_clusters: Dict[int, set[int]] = defaultdict(set)
        if self.labels is not None:
            for label, cluster in zip(self.labels, self.clusters):
                cluster_int = int(cluster)
                if (
                    cluster_int != int(ambiguous_tag)
                    and cluster_int in self.cluster_to_idx
                ):
                    self.class_to_clusters[int(label)].add(cluster_int)

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
            pool = list(self.class_to_clusters[cls])
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


@SamplerFactory.register()
class RivalClusterBatchSampler(Sampler[List[int]]):
    """
    Cluster sampler that prioritizes rival clusters in the same batch.

    Batch composition (no leftover):
      core_size = clusters_per_batch * samples_per_cluster
      + include_ambiguous

    - Samples are drawn WITHOUT replacement inside each cluster.
    - Clusters must have at least samples_per_cluster samples.
    - If cluster_rivals is provided, rival pairs (opposite classes) are prioritized.
    - Maintains class balance across batches.

    cluster_rivals: dict {cluster_id: [rival1, rival2, ...]} (cross-class rivals)
    """

    def __init__(
        self,
        clusters: np.ndarray,
        labels: Optional[np.ndarray] = None,
        batch_size: int = 128,
        clusters_per_batch: int = 8,
        samples_per_cluster: int = 4,
        include_ambiguous: int = 0,
        ambiguous_tag: int = -1,
        drop_last: bool = True,
        cluster_rivals: Optional[Dict[int, Sequence[int]]] = None,
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

        self.cluster_rivals = cluster_rivals if cluster_rivals is not None else {}

        # Min cluster size is always samples_per_cluster
        self.min_cluster_size = self.samples_per_cluster

        # Validate batch size: core + ambiguous must exactly fit (no leftover)
        self.core_size = self.clusters_per_batch * self.samples_per_cluster
        expected = self.core_size + self.include_ambiguous
        if expected != self.batch_size:
            raise ValueError(
                "This sampler uses no leftover. "
                f"Set batch_size = clusters_per_batch*samples_per_cluster + include_ambiguous = {expected}. "
                f"Got batch_size={self.batch_size}."
            )

        # Build cluster index
        cluster_to_idx: Dict[int, List[int]] = defaultdict(list)
        ambiguous_idx: List[int] = []

        for i, c in enumerate(self.clusters):
            ci = int(c)
            if ci == int(ambiguous_tag):
                ambiguous_idx.append(i)
            else:
                cluster_to_idx[ci].append(i)

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

        # Optional: build class->clusters (for balanced anchor choice / cross-class checks)
        self.class_to_clusters: Dict[int, List[int]] = defaultdict(list)
        self.cluster_to_class: Dict[int, int] = {}

        if self.labels is not None:
            # majority label per cluster
            for c, idxs in self.cluster_to_idx.items():
                maj = int(np.bincount(self.labels[idxs]).argmax())
                self.cluster_to_class[c] = maj
                self.class_to_clusters[maj].append(c)

        # Precompute eligible cluster IDs
        self.cluster_ids = list(self.cluster_to_idx.keys())
        if len(self.cluster_ids) < self.clusters_per_batch:
            raise ValueError(
                f"Not enough eligible clusters ({len(self.cluster_ids)}) to sample "
                f"{self.clusters_per_batch} per batch."
            )

        # Clean rivals list: keep only rivals that are eligible clusters (and optionally cross-class)
        if self.cluster_rivals:
            cleaned = {}
            for c, rivals in self.cluster_rivals.items():
                if c not in self.cluster_to_idx:
                    continue
                rr = [int(r) for r in rivals if int(r) in self.cluster_to_idx]
                if self.labels is not None and c in self.cluster_to_class:
                    # keep only cross-class rivals
                    cc = self.cluster_to_class[c]
                    rr = [
                        r
                        for r in rr
                        if self.cluster_to_class.get(r, None) is None
                        or self.cluster_to_class.get(r) != cc
                    ]
                if rr:
                    cleaned[c] = rr
            self.cluster_rivals = cleaned

    def __len__(self) -> int:
        total_core_samples = sum(len(idxs) for idxs in self.cluster_to_idx.values())
        est_batches = total_core_samples // self.core_size
        return est_batches if self.drop_last else est_batches + 1

    def _pick_anchor_clusters(self, rng: random.Random, n: int) -> List[int]:
        """Pick n anchor clusters (optionally class-balanced)."""
        if not self.class_to_clusters:
            # simple random anchors
            return rng.sample(self.cluster_ids, k=n)

        classes = list(self.class_to_clusters.keys())
        rng.shuffle(classes)
        per_class, rem = divmod(n, len(classes))

        anchors: List[int] = []
        for i, cls in enumerate(classes):
            need = per_class + (1 if i < rem else 0)
            pool = self.class_to_clusters[cls]
            if not pool:
                continue
            if len(pool) >= need:
                anchors.extend(rng.sample(pool, k=need))
            else:
                # fall back to sampling with replacement across anchors (rare)
                anchors.extend(rng.choices(pool, k=need))

        # ensure unique; if duplicates, fill randomly
        anchors_unique = []
        seen = set()
        for c in anchors:
            if c not in seen:
                anchors_unique.append(c)
                seen.add(c)
        while len(anchors_unique) < n:
            c = rng.choice(self.cluster_ids)
            if c not in seen:
                anchors_unique.append(c)
                seen.add(c)
        return anchors_unique[:n]

    def _pick_rival_for(
        self, rng: random.Random, c: int, forbidden: set[int]
    ) -> Optional[int]:
        """Pick one rival for cluster c not in forbidden."""
        rivals = self.cluster_rivals.get(c, [])
        candidates = [r for r in rivals if r not in forbidden]
        if not candidates:
            return None
        # choose among top-k rivals randomly (keeps stochasticity but stays hard)
        return rng.choice(candidates)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self._iter_count)
        self._iter_count += 1

        # Initialize per-cluster available index pools (exhaustive sampling)
        cluster_available: Dict[int, List[int]] = {}
        for c, idxs in self.cluster_to_idx.items():
            pool = list(idxs)
            rng.shuffle(pool)
            cluster_available[c] = pool

        # Initialize ambiguous available pool
        ambiguous_available = list(self.ambiguous_idx)
        rng.shuffle(ambiguous_available)

        for _ in range(len(self)):
            chosen_clusters: List[int] = []
            chosen_set: set[int] = set()

            # Prioritize rival clusters if available
            if self.cluster_rivals:
                # Get all anchor clusters that have rivals and are eligible
                anchors_with_rivals = [
                    c for c in self.cluster_rivals.keys() if c in self.cluster_to_idx
                ]
                rng.shuffle(anchors_with_rivals)

                # Try to fill batch with anchor-rival pairs
                for anchor in anchors_with_rivals:
                    if len(chosen_clusters) >= self.clusters_per_batch:
                        break
                    if anchor in chosen_set:
                        continue

                    # Try to add rival (should be opposite class)
                    rival = self._pick_rival_for(rng, anchor, chosen_set)

                    # Add anchor
                    chosen_clusters.append(anchor)
                    chosen_set.add(anchor)

                    # Add rival if available and space permits
                    if (
                        rival is not None
                        and len(chosen_clusters) < self.clusters_per_batch
                    ):
                        chosen_clusters.append(rival)
                        chosen_set.add(rival)

            # Fill remaining slots with class-balanced random selection
            remaining = self.clusters_per_batch - len(chosen_clusters)
            if remaining > 0:
                fillers = self._pick_anchor_clusters(rng, remaining)
                for c in fillers:
                    if c not in chosen_set:
                        chosen_clusters.append(c)
                        chosen_set.add(c)
                    if len(chosen_clusters) >= self.clusters_per_batch:
                        break

            # Ensure we have exactly clusters_per_batch clusters
            chosen_clusters = chosen_clusters[: self.clusters_per_batch]
            # print(f"Chosen clusters for batch [{self._iter_count}]: {chosen_clusters}")

            # Sample indices from each chosen cluster WITHOUT replacement from available pool
            batch: List[int] = []
            used: set[int] = set()
            for c in chosen_clusters:
                # Refill cluster pool if exhausted
                if len(cluster_available[c]) < self.samples_per_cluster:
                    pool = list(self.cluster_to_idx[c])
                    rng.shuffle(pool)
                    cluster_available[c] = pool

                # Sample from available pool
                samples = cluster_available[c][: self.samples_per_cluster]
                cluster_available[c] = cluster_available[c][self.samples_per_cluster :]
                batch.extend(samples)
                used.update(samples)

            # Add ambiguous samples (optional), avoiding duplicates
            if self.include_ambiguous > 0:
                if not self.ambiguous_idx:
                    raise ValueError(
                        "include_ambiguous > 0 but no ambiguous samples found."
                    )

                # Refill ambiguous pool if needed
                if len(ambiguous_available) < self.include_ambiguous:
                    pool = list(self.ambiguous_idx)
                    rng.shuffle(pool)
                    ambiguous_available = pool

                # Sample from available pool, avoiding already used indices in this batch
                amb_samples = []
                temp_pool = []
                for idx in ambiguous_available:
                    if idx not in used:
                        if len(amb_samples) < self.include_ambiguous:
                            amb_samples.append(idx)
                        else:
                            temp_pool.append(idx)
                    else:
                        temp_pool.append(idx)

                if len(amb_samples) < self.include_ambiguous:
                    raise ValueError(
                        f"Not enough unique ambiguous samples to draw {self.include_ambiguous} "
                        f"(available={len([i for i in ambiguous_available if i not in used])}). "
                        f"Reduce include_ambiguous or increase data."
                    )

                batch.extend(amb_samples)
                ambiguous_available = temp_pool

            if len(batch) != self.batch_size:
                raise RuntimeError(
                    f"Internal error: batch size {len(batch)} != expected {self.batch_size}"
                )

            yield batch
