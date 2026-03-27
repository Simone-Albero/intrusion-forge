# Cluster Features

Per-cluster features used in the failure-prediction pipeline.
All features are computed at cluster level (one row per cluster) and assembled
by `build_cluster_summary` in `src/data/analyze.py`.

---

## Intra-cluster dispersion

Measure how spread out the samples inside a cluster are, relative to its centroid.
High dispersion means the cluster is loose; low dispersion means it is tight.

| Feature | Description |
|---|---|
| `intra_dispersion` | Mean distance from each sample to the cluster centroid. |
| `std_dispersion` | Standard deviation of sample-to-centroid distances. |
| `median_dispersion` | Median of sample-to-centroid distances. More robust to outliers than the mean. |
| `max_dispersion` | Maximum sample-to-centroid distance. Captures the worst-case reach of the cluster. |
| `p95_dispersion` | 95th percentile of sample-to-centroid distances. |
| `p99_dispersion` | 99th percentile of sample-to-centroid distances. Sensitive to extreme outliers within the cluster. |

---

## Density

| Feature | Description |
|---|---|
| `density` | Sample count divided by the cube of the mean intra-cluster distance (`n / intra³`). A proxy for how densely populated the cluster is relative to its spatial extent. |

---

## Centroid-level distances

Distances between cluster centroids, used to understand isolation at the
cluster-aggregate level rather than sample level.

| Feature | Description |
|---|---|
| `dist_to_class_centroid` | Distance from the cluster centroid to the centroid of all samples belonging to the same class. Large values indicate the cluster is an outlier within its own class. |
| `dist_to_nearest_cluster` | Distance to the nearest other cluster centroid (any class). |
| `dist_to_nearest_foreign_cluster` | Distance to the nearest cluster centroid of a *different* class. |

---

## Separation and overlap margins

These features compare intra-cluster spread to the distance to the nearest
foreign cluster, giving a direct signal for separability.

| Feature | Description |
|---|---|
| `foreign_separation_ratio` | `dist_to_nearest_foreign_cluster / intra_dispersion`. Values < 1 indicate the foreign cluster is closer than the cluster's own average radius. |
| `overlap_margin` | `dist_to_nearest_foreign_cluster - dist_to_nearest_cluster`. Positive means a same-class neighbour is closer than any foreign neighbour; negative indicates potential overlap. |
| `normalized_overlap_margin` | `overlap_margin / dist_to_nearest_foreign_cluster`. Scale-invariant version of `overlap_margin`. |
| `intra_foreign_margin` | `dist_to_nearest_foreign_cluster - intra_dispersion`. How much room there is between the average cluster radius and the nearest foreign cluster. |
| `max_foreign_margin` | `dist_to_nearest_foreign_cluster - max_dispersion`. Margin between the cluster's worst-case reach and the nearest foreign cluster. Negative means at least one sample is closer to a foreign cluster than the centroid is. |
| `foreign_coverage_ratio` | `max_dispersion / dist_to_nearest_foreign_cluster`. Fraction of the gap to the nearest foreign cluster that the cluster's maximum reach covers. Values approaching or exceeding 1 indicate high risk of overlap. |

---

## Silhouette

Silhouette score is computed per sample and then aggregated. Values range
from −1 (misclassified) to +1 (well-separated).

| Feature | Description |
|---|---|
| `silhouette` | Mean silhouette score over all cluster samples. |
| `min_silhouette` | Minimum silhouette score in the cluster. Captures the most at-risk sample. |
| `p5_silhouette` | 5th percentile of silhouette scores. More stable than the minimum; still sensitive to the worst tail. |

---

## Fraction at risk

| Feature | Description |
|---|---|
| `frac_at_risk` | Fraction of samples whose minimum distance to any foreign-class centroid is smaller than their distance to the cluster centroid. A sample is "at risk" when it is geometrically closer to a foreign class than to its own cluster centre. |

---

## Pairwise separability ratios

These features are derived from pair-wise `mean_intra / mean_inter` separability
ratios between clusters (output of `compute_pairwise_separability`). The ratio
measures how much a cluster overlaps with each peer cluster: low = well-separated,
high = overlapping.

| Feature | Description |
|---|---|
| `min_foreign_ratio` | Minimum separability ratio to any cluster of a *different* class. The best-case foreign separation from this cluster's perspective. |
| `max_foreign_ratio` | Maximum separability ratio to any cluster of a different class. Worst-case foreign overlap. |
| `min_self_ratio` | Minimum separability ratio to any cluster of the *same* class. Measures intra-class cohesion: a high value means this cluster is isolated even from its own class. |
| `max_self_ratio` | Maximum separability ratio to any cluster of the same class. |
| `ratio_spread` | `max_foreign_ratio − min_self_ratio`. Positive values mean the worst foreign overlap exceeds the best self-cohesion, signalling likely confusion. Equivalent to measuring whether the cluster is "closer" to foreign peers than to same-class peers (in ratio space). |
| `ratio_scale` | `max_foreign_ratio / min_self_ratio`. Scale-invariant version of `ratio_spread`. Values > 1 indicate the foreign overlap dominates over self-cohesion. Preferred when ratio magnitudes vary widely across datasets. |
