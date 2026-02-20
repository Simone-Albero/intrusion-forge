import random
from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler

from . import SamplerFactory


@SamplerFactory.register()
class RivalClusterBatchSampler(Sampler[list[int]]):
    """Cluster sampler that prioritizes rival (cross-class) clusters in the same batch.

    Batch composition: batch_size = clusters_per_batch * samples_per_cluster + include_ambiguous.
    Clusters are sampled without replacement within each epoch. Ambiguous samples are drawn
    independently. Rival pairs are prioritized when building each batch.

    Args:
        cluster_rivals: {cluster_id: [rival_id, ...]} cross-class rival mapping.
    """

    def __init__(
        self,
        clusters: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        batch_size: int = 128,
        clusters_per_batch: int = 8,
        samples_per_cluster: int = 4,
        include_ambiguous: int = 0,
        ambiguous_tag: int = -1,
        drop_last: bool = True,
        cluster_rivals: dict[int, list[int]] | None = None,
        seed: int = 0,
    ):
        self.batch_size = batch_size
        self.clusters_per_batch = clusters_per_batch
        self.samples_per_cluster = samples_per_cluster
        self.include_ambiguous = include_ambiguous
        self.drop_last = drop_last
        self.seed = seed
        self._iter_count = 0

        self.core_size = clusters_per_batch * samples_per_cluster
        expected = self.core_size + include_ambiguous
        if expected != batch_size:
            raise ValueError(
                f"batch_size must equal clusters_per_batch * samples_per_cluster + include_ambiguous "
                f"(= {expected}), got {batch_size}."
            )

        # Cluster state — populated by init_clusters()
        self.cluster_to_idx: dict[int, list[int]] | None = None
        self.ambiguous_idx: list[int] = []
        self.class_to_clusters: dict[int, list[int]] = defaultdict(list)
        self.cluster_to_class: dict[int, int] = {}
        self.cluster_ids: list[int] = []
        self.cluster_rivals: dict[int, list[int]] = {}

        if clusters is not None:
            self.init_clusters(clusters, labels, cluster_rivals, ambiguous_tag)

    def init_clusters(
        self,
        clusters: np.ndarray,
        labels: np.ndarray | None = None,
        cluster_rivals: dict[int, list[int]] | None = None,
        ambiguous_tag: int = -1,
    ) -> None:
        """Initialize cluster structures. Must be called before iteration if clusters was not passed to __init__."""
        self.clusters = np.asarray(clusters)
        self.labels = None if labels is None else np.asarray(labels)

        # Build cluster and ambiguous index pools
        cluster_to_idx: dict[int, list[int]] = defaultdict(list)
        ambiguous_idx: list[int] = []

        for i, c in enumerate(self.clusters):
            ci = int(c)
            if ci == int(ambiguous_tag):
                ambiguous_idx.append(i)
            else:
                cluster_to_idx[ci].append(i)

        self.cluster_to_idx = {
            c: idxs
            for c, idxs in cluster_to_idx.items()
            if len(idxs) >= self.samples_per_cluster
        }
        self.ambiguous_idx = ambiguous_idx

        if not self.cluster_to_idx:
            raise ValueError(
                f"No clusters with at least {self.samples_per_cluster} samples found."
            )

        # Build class <-> cluster mappings (majority label per cluster)
        self.class_to_clusters = defaultdict(list)
        self.cluster_to_class = {}

        if self.labels is not None:
            for c, idxs in self.cluster_to_idx.items():
                maj = int(np.bincount(self.labels[idxs]).argmax())
                self.cluster_to_class[c] = maj
                self.class_to_clusters[maj].append(c)

        self.cluster_ids = list(self.cluster_to_idx.keys())
        if len(self.cluster_ids) < self.clusters_per_batch:
            raise ValueError(
                f"Not enough eligible clusters ({len(self.cluster_ids)}) "
                f"to sample {self.clusters_per_batch} per batch."
            )

        # Clean rivals: keep only eligible cross-class rivals
        self.cluster_rivals = {}
        for c, rivals in (cluster_rivals or {}).items():
            if c not in self.cluster_to_idx:
                continue
            valid = [int(r) for r in rivals if int(r) in self.cluster_to_idx]
            if self.cluster_to_class:
                cc = self.cluster_to_class.get(c)
                valid = [r for r in valid if self.cluster_to_class.get(r) != cc]
            if valid:
                self.cluster_rivals[c] = valid

    def __len__(self) -> int:
        if self.cluster_to_idx is None:
            raise RuntimeError("Clusters not initialized. Call init_clusters() first.")
        total = sum(len(idxs) for idxs in self.cluster_to_idx.values())
        est = total // self.core_size
        return est if self.drop_last else est + 1

    def _pick_anchor_clusters(self, rng: random.Random, n: int) -> list[int]:
        """Pick n clusters, class-balanced when labels are available."""
        if not self.class_to_clusters:
            return rng.sample(self.cluster_ids, k=n)

        classes = list(self.class_to_clusters.keys())
        rng.shuffle(classes)
        per_class, rem = divmod(n, len(classes))

        anchors: list[int] = []
        for i, cls in enumerate(classes):
            need = per_class + (1 if i < rem else 0)
            pool = self.class_to_clusters[cls]
            anchors.extend(rng.sample(pool, k=min(need, len(pool))))

        # Fill any shortfall due to class imbalance
        seen = set(anchors)
        while len(anchors) < n:
            c = rng.choice(self.cluster_ids)
            if c not in seen:
                anchors.append(c)
                seen.add(c)
        return anchors[:n]

    @staticmethod
    def compute_rivals(
        z: np.ndarray,
        y: np.ndarray,
        c: np.ndarray,
        invalid_classes: list[int] | None = None,
    ) -> dict[int, list[int]]:
        """Compute cluster rivals from normalized embeddings, labels, and cluster ids.

        Args:
            z: Normalized embeddings, shape (N, D).
            y: Class labels, shape (N,).
            c: Cluster ids, shape (N,). Use -1 for ambiguous/invalid samples.
            invalid_classes: Class ids to exclude from anchors and rivals.

        Returns:
            Dict mapping anchor cluster id -> list of rival cluster ids.
        """
        invalid_classes = set(invalid_classes or [])
        valid_mask = c != -1
        if not valid_mask.any():
            return {}

        z_v, y_v, c_v = z[valid_mask], y[valid_mask], c[valid_mask]

        cluster_centroids: dict[int, np.ndarray] = {}
        cluster_to_class: dict[int, int] = {}
        cluster_purity: dict[int, float] = {}
        for cid in np.unique(c_v):
            cid = int(cid)
            mask = c_v == cid
            z_c, y_c = z_v[mask], y_v[mask]
            centroid = z_c.mean(axis=0)
            cluster_centroids[cid] = centroid / (np.linalg.norm(centroid) + 1e-12)
            counts = np.bincount(y_c.astype(int))
            cluster_to_class[cid] = int(counts.argmax())
            cluster_purity[cid] = float(counts.max() / counts.sum())

        anchor_clusters = [
            cid
            for cid, p in cluster_purity.items()
            if p >= 0.90 and cluster_to_class[cid] not in invalid_classes
        ]
        if not anchor_clusters:
            return {}

        centroid_ids = list(cluster_centroids.keys())
        centroids_matrix = np.stack([cluster_centroids[cid] for cid in centroid_ids])
        sim_matrix = centroids_matrix @ centroids_matrix.T
        idx_map = {cid: i for i, cid in enumerate(centroid_ids)}

        max_rivals, min_similarity, similarity_delta = 5, 0.7, 0.10
        rivals: dict[int, list[int]] = {}

        for cid in anchor_clusters:
            anchor_class = cluster_to_class[cid]
            candidates = sorted(
                [
                    (float(sim_matrix[idx_map[cid], idx_map[other]]), int(other))
                    for other in centroid_ids
                    if other != cid
                    and cluster_to_class.get(other) != anchor_class
                    and cluster_to_class.get(other) not in invalid_classes
                ],
                reverse=True,
            )
            if not candidates:
                continue
            best_sim = candidates[0][0]
            filtered = [
                (s, o)
                for s, o in candidates
                if s >= min_similarity and s >= best_sim - similarity_delta
            ]
            if filtered:
                rivals[cid] = [o for _, o in filtered[:max_rivals]]

        return rivals

    def _pick_rival(
        self, rng: random.Random, c: int, forbidden: set[int]
    ) -> int | None:
        """Pick one eligible rival for cluster c not in forbidden."""
        candidates = [r for r in self.cluster_rivals.get(c, []) if r not in forbidden]
        return rng.choice(candidates) if candidates else None

    def __iter__(self):
        if self.cluster_to_idx is None:
            raise RuntimeError("Clusters not initialized. Call init_clusters() first.")
        rng = random.Random(self.seed + self._iter_count)
        self._iter_count += 1

        # Per-cluster pools, consumed without replacement across the epoch
        cluster_available: dict[int, list[int]] = {}
        for c, idxs in self.cluster_to_idx.items():
            pool = list(idxs)
            rng.shuffle(pool)
            cluster_available[c] = pool

        # Ambiguous pool
        ambiguous_pool = list(self.ambiguous_idx)
        rng.shuffle(ambiguous_pool)

        for _ in range(len(self)):
            # --- Select clusters ---
            chosen: list[int] = []
            chosen_set: set[int] = set()

            if self.cluster_rivals:
                anchors = [c for c in self.cluster_rivals if c in self.cluster_to_idx]
                rng.shuffle(anchors)
                for anchor in anchors:
                    if len(chosen) >= self.clusters_per_batch:
                        break
                    if anchor in chosen_set:
                        continue
                    chosen.append(anchor)
                    chosen_set.add(anchor)
                    if len(chosen) < self.clusters_per_batch:
                        rival = self._pick_rival(rng, anchor, chosen_set)
                        if rival is not None:
                            chosen.append(rival)
                            chosen_set.add(rival)

            remaining = self.clusters_per_batch - len(chosen)
            if remaining > 0:
                for c in self._pick_anchor_clusters(rng, remaining):
                    if c not in chosen_set and len(chosen) < self.clusters_per_batch:
                        chosen.append(c)
                        chosen_set.add(c)

            chosen = chosen[: self.clusters_per_batch]

            # --- Sample from each cluster ---
            batch: list[int] = []
            for c in chosen:
                if len(cluster_available[c]) < self.samples_per_cluster:
                    pool = list(self.cluster_to_idx[c])
                    rng.shuffle(pool)
                    cluster_available[c] = pool
                batch.extend(cluster_available[c][: self.samples_per_cluster])
                cluster_available[c] = cluster_available[c][self.samples_per_cluster :]

            # --- Sample ambiguous ---
            if self.include_ambiguous > 0:
                if not self.ambiguous_idx:
                    raise ValueError(
                        "include_ambiguous > 0 but no ambiguous samples found."
                    )
                if len(ambiguous_pool) < self.include_ambiguous:
                    ambiguous_pool = list(self.ambiguous_idx)
                    rng.shuffle(ambiguous_pool)
                batch.extend(ambiguous_pool[: self.include_ambiguous])
                ambiguous_pool = ambiguous_pool[self.include_ambiguous :]

            yield batch
