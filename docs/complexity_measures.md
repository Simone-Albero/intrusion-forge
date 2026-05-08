# Complexity Measures

All measures are computed per cluster via `compute_all_complexity_measures`.

The pairwise families (F, N, ND) are aggregated against two scopes:

- **`*_class_*`**: cluster *c* (class `cls_c`) vs each adversarial **class** `j ≠ cls_c`.
  `X_j` is the union of all non-noise samples of class `j`.
- **`*_cluster_*`**: cluster *c* vs each of the **top-K nearest adversarial
  clusters** `c'` (i.e., clusters of any class `≠ cls_c`), ranked by Euclidean
  centroid distance.

For each scope the final value is the **min** (worst case), **mean**, and
**max** (best case) over the populations considered.

The k-NN graph is built once using **Gower distance** (batched, mixed
numerical + categorical). Numerical features are pre-scaled by their **IQR
(Q3 − Q1)** to keep the metric robust to heavy-tailed outliers. Defaults:
`k = 30`, `top_k_clusters = 10`. Both are configurable via
`configs/experiment/supervised.yaml`.

---

## Family F — Feature-based

Measures how well individual features separate a cluster from an adversarial
population. Computed on RobustScaled numerical features only (`X_num`).
Categorical information is captured by the N/ND families.

### `f1_class_{min,mean,max}` / `f1_cluster_{min,mean,max}`

**Fisher discriminant ratio**

$$f_1(c, j) = \frac{1}{1 + \max_f \frac{(\mu_{c,f} - \mu_{j,f})^2}{\sigma^2_{c,f} + \sigma^2_{j,f}}}$$

| | |
|---|---|
| **Complexity** | $O(d)$ per pair |
| **Range** | $(0, 1]$ |
| **→ 0** | Large Fisher ratio on at least one feature → easy to separate |
| **→ 1** | All features have zero discriminative power → very hard to separate |

---

### `f2_class_{min,mean,max}` / `f2_cluster_{min,mean,max}`

**Bounding-box overlap ratio**

Mean fraction of per-feature range shared between cluster $c$ and adversarial
population $j$.

$$f_2(c, j) = \frac{1}{d} \sum_f \frac{\max(0,\, \min(c_{\max,f}, j_{\max,f}) - \max(c_{\min,f}, j_{\min,f}))}{\max(c_{\max,f}, j_{\max,f}) - \min(c_{\min,f}, j_{\min,f})}$$

| | |
|---|---|
| **Complexity** | $O(nd)$ per pair |
| **Range** | $[0, 1]$ |
| **→ 0** | No feature-space overlap → easy to separate |
| **→ 1** | Complete overlap on every feature → very hard to separate |

---

### `f3_class_{min,mean,max}` / `f3_cluster_{min,mean,max}`

**Best single-feature separability (fraction in overlap region)**

For each feature, fraction of cluster-$c$ samples that fall in the overlap
region with the adversarial population. Takes the **minimum over features**
(best discriminating feature).

$$f_3(c, j) = \min_f \frac{|\{x \in c : lo_f \le x_f \le hi_f\}|}{|c|}$$

where $lo_f = \max(c_{\min,f},\, j_{\min,f})$ and $hi_f = \min(c_{\max,f},\, j_{\max,f})$.

| | |
|---|---|
| **Complexity** | $O(nd)$ per pair |
| **Range** | $[0, 1]$ |
| **→ 0** | At least one feature fully separates cluster $c$ from $j$ |
| **→ 1** | Every feature has all cluster-$c$ samples inside the overlap region |

---

### `f4_class_{min,mean,max}` / `f4_cluster_{min,mean,max}`

**Joint-feature overlap fraction**

Fraction of cluster-$c$ samples that fall inside the overlap region on **all
features simultaneously**.

$$f_4(c, j) = \frac{|\{x \in c : \forall f,\; lo_f \le x_f \le hi_f\}|}{|c|}$$

| | |
|---|---|
| **Complexity** | $O(nd)$ per pair |
| **Range** | $[0, 1]$ |
| **→ 0** | No sample simultaneously overlaps on all features → clean joint separation |
| **→ 1** | All samples are jointly inside the overlap region on every feature |

---

## Family N — Neighborhood-based

Measures local overlap in the k-NN graph (Gower distance). Noise points
($y_{cluster} = -1$) are excluded from cluster membership but may appear as
neighbours.

### `n1_class_{min,mean,max}` / `n1_cluster_{min,mean,max}`

**MST boundary fraction**

Fraction of cluster-$c$ samples that have at least one edge in the
approximate minimum spanning tree connecting them to a sample of the
adversarial population. The MST is built on the symmetrised k-NN graph and a
bridge edge is added per disconnected component to guarantee connectivity.

$$n_1(c, j) = \frac{|\{x \in c : \exists\, (x, y) \in \text{MST},\; y \in j\}|}{|c|}$$

| | |
|---|---|
| **Complexity** | $O(n \log n)$ for MST; $O(|E|)$ vectorised boundary check |
| **Range** | $[0, 1]$ |
| **→ 0** | Cluster $c$ is entirely interior, never adjacent to $j$ in the MST |
| **→ 1** | Every sample of cluster $c$ borders $j$ in the MST |

---

### `n2_class_{min,mean,max}` / `n2_cluster_{min,mean,max}`

**Intra/inter nearest-neighbour distance ratio**

For each sample $x \in c$, ratio of nearest same-cluster distance to the sum
of intra and inter distances:

$$n_2(x) = \frac{d_{\text{intra}}}{d_{\text{intra}} + d_{\text{inter}}}$$

Averaged over all samples in $c$ that have both kinds of neighbour within
the k-NN.

| | |
|---|---|
| **Complexity** | $O(|c| \cdot k)$ per pair |
| **Range** | $[0, 1]$ |
| **→ 0** | Intra-cluster neighbours are far closer than $j$ neighbours → well-separated |
| **→ 0.5** | Intra and inter distances are equal → no local separation |
| **→ 1** | $j$ neighbours are much closer than same-cluster neighbours → severe overlap |

---

### `n3_class_{min,mean,max}` / `n3_cluster_{min,mean,max}`

**1-NN error rate restricted to c ∪ j**

For each sample $x \in c$, find its nearest neighbour in the k-NN list that
belongs to $c \cup j$. Misclassified if that neighbour is in $j$.

$$n_3(c, j) = \frac{|\{x \in c : \mathrm{NN}_{c \cup j}(x) \in j\}|}{|c|}$$

| | |
|---|---|
| **Complexity** | $O(|c| \cdot k)$ per pair |
| **Range** | $[0, 1]$ |
| **→ 0** | Nearest neighbour is always from the same cluster → clean local boundary |
| **→ 1** | Nearest neighbour is always from $j$ → cluster $c$ is embedded inside $j$ |

> Practical note: this metric is meaningful only if the k-NN list has enough
> reach to contain at least one $c \cup j$ neighbour for most samples. With
> the new default `k = 30` this is true for the vast majority of samples on
> NB15/CIC-IDS-2018.

---

### `n4_class_{min,mean,max}` / `n4_cluster_{min,mean,max}`

**k-NN majority-vote error rate**

For each sample $x \in c$, count votes from same-cluster vs $j$ neighbours.
Misclassified if $j$-votes > $c$-votes.

$$n_4(c, j) = \frac{|\{x \in c : \text{votes}_j(x) > \text{votes}_c(x)\}|}{|c|}$$

| | |
|---|---|
| **Complexity** | $O(|c| \cdot k)$ per pair |
| **Range** | $[0, 1]$ |
| **→ 0** | Majority of neighbours always from same cluster |
| **→ 1** | Majority of neighbours always from $j$ |

---

## Family ND — Network density

### `network_density_class_{min,mean,max}` / `network_density_cluster_{min,mean,max}`

**Cross-population k-NN density**

Fraction of k-NN edges from cluster $c$ pointing to samples of the
adversarial population:

$$\text{density}(c, j) = \frac{\sum_{x \in c} |\{nb \in NN(x) : nb \in j\}|}{|c| \times k}$$

| | |
|---|---|
| **Complexity** | $O(|c| \cdot k)$ per pair |
| **Range** | $[0, 1]$ |
| **→ 0** | No k-NN edges cross to $j$ → isolated cluster |
| **→ 1** | All k-NN edges from cluster $c$ point to $j$ → fully intermixed |

---

### `cls_coef`

**Local clustering coefficient** within the cluster (intra-cluster
neighbour pairs that are themselves neighbours, averaged across cluster
members).

### `hub`

**Mean in-degree** in the reverse k-NN graph (hubness proxy).

---

## Family T — Dimensionality

Per-cluster dimensionality measures. Computed on the subset of samples
belonging to each cluster.

### `t2`

**Feature-to-sample ratio**

$$T_2 = \frac{d_{\text{num}} + d_{\text{cat}}}{n_c}$$

### `t3`

**PCA intrinsic dimensionality ratio** — fraction of numerical features
needed to explain 95% of variance within the cluster.

### `t4`

**PCA components-to-sample ratio**

$$T_4 = \frac{n_{\text{PCA}_{95\%}}}{n_c}$$

T-family is computed on numerical features only by design (categorical
information is captured by the N/ND families).

---

## Family G — Geometry

Per-cluster geometric properties computed in Euclidean space on `X_num`.
Silhouette is approximated via stratified sampling (up to 10 000 points).
G-family is numerical-only by design.

### `max_dispersion`

Maximum Euclidean distance from any sample in the cluster to its centroid.

### `p95_dispersion`

95th percentile of the sample → centroid distances. Outlier-robust
counterpart to `max_dispersion`.

### `dist_to_nearest_foreign_cluster`

Minimum centroid-to-centroid Euclidean distance from cluster $c$ to any
cluster of a different class.

### `p5_silhouette`

5th percentile of silhouette scores (Euclidean, computed on non-noise points
with cluster labels).

### `frac_at_risk`

Fraction of cluster-$c$ samples with silhouette score $< 0$.

### `min_sibling_centroid_dist`

Minimum centroid-to-centroid Euclidean distance from cluster $c$ to any
other cluster of the **same class**. `None` if the class has only one
cluster.
