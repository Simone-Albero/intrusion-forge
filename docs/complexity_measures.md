# Complexity Measures

All measures are computed per cluster via `compute_all_complexity_measures`.
Each cluster is treated as a binary problem: *cluster c (class `cls_c`)* vs *each adversarial class `j ≠ cls_c`*.
When a measure is pairwise, the final value is the **min** (worst case), **mean**, and **max** (best case) over all adversarial classes.

The k-NN graph is built once using **Gower distance** (batched, mixed numerical + categorical support) with `k = 5` by default.

---

## Family F — Feature-based

Measures how well individual features separate a cluster from adversarial classes.
Computed on RobustScaled numerical features only (`X_num`).

### `f1_min` / `f1_mean` / `f1_max`

**Fisher discriminant ratio**

$$f_1(c, j) = \frac{1}{1 + \max_f \frac{(\mu_{c,f} - \mu_{j,f})^2}{\sigma^2_{c,f} + \sigma^2_{j,f}}}$$

| | |
|---|---|
| **Complexity** | $O(d)$ per pair |
| **Range** | $(0, 1]$ |
| **→ 0** | Large Fisher ratio on at least one feature → easy to separate |
| **→ 1** | All features have zero discriminative power → very hard to separate |

---

### `f2_min` / `f2_mean` / `f2_max`

**Bounding-box overlap ratio**

Mean fraction of per-feature range shared between cluster $c$ and class $j$.

$$f_2(c, j) = \frac{1}{d} \sum_f \frac{\max(0,\, \min(c_{\max,f}, j_{\max,f}) - \max(c_{\min,f}, j_{\min,f}))}{\max(c_{\max,f}, j_{\max,f}) - \min(c_{\min,f}, j_{\min,f})}$$

| | |
|---|---|
| **Complexity** | $O(nd)$ per pair |
| **Range** | $[0, 1]$ |
| **→ 0** | No feature-space overlap → easy to separate |
| **→ 1** | Complete overlap on every feature → very hard to separate |

---

### `f3_min` / `f3_mean` / `f3_max`

**Best single-feature separability (fraction in overlap region)**

For each feature, fraction of cluster-$c$ samples that fall in the overlap region with class $j$.
Takes the **minimum over features** (best discriminating feature).

$$f_3(c, j) = \min_f \frac{|\{x \in c : lo_f \le x_f \le hi_f\}|}{|c|}$$

where $lo_f = \max(c_{\min,f},\, j_{\min,f})$ and $hi_f = \min(c_{\max,f},\, j_{\max,f})$.

| | |
|---|---|
| **Complexity** | $O(nd)$ per pair |
| **Range** | $[0, 1]$ |
| **→ 0** | At least one feature fully separates cluster $c$ from class $j$ |
| **→ 1** | Every feature has all cluster-$c$ samples inside the overlap region |

---

### `f4_min` / `f4_mean` / `f4_max`

**Joint-feature overlap fraction**

Fraction of cluster-$c$ samples that fall inside the overlap region on **all features simultaneously**.

$$f_4(c, j) = \frac{|\{x \in c : \forall f,\; lo_f \le x_f \le hi_f\}|}{|c|}$$

| | |
|---|---|
| **Complexity** | $O(nd)$ per pair |
| **Range** | $[0, 1]$ |
| **→ 0** | No sample simultaneously overlaps on all features → clean joint separation |
| **→ 1** | All samples are jointly inside the overlap region on every feature |

---

## Family N — Neighborhood-based

Measures local overlap in the k-NN graph (Gower distance).
Noise points ($y_{cluster} = -1$) are excluded from cluster membership but may appear as neighbors.

### `n1_min` / `n1_mean` / `n1_max`

**MST boundary fraction**

Fraction of cluster-$c$ samples that have at least one edge in the approximate minimum spanning tree connecting them to a sample of class $j$.

$$n_1(c, j) = \frac{|\{x \in c : \exists\, (x, y) \in \text{MST},\; y \in j\}|}{|c|}$$

| | |
|---|---|
| **Complexity** | $O(n \log n)$ for MST; $O(n)$ per pair for boundary check |
| **Range** | $[0, 1]$ |
| **→ 0** | Cluster $c$ is entirely interior, never adjacent to class $j$ in the MST |
| **→ 1** | Every sample of cluster $c$ borders class $j$ in the MST |

---

### `n2_min` / `n2_mean` / `n2_max`

**Intra/inter nearest-neighbor distance ratio**

For each sample $x \in c$, ratio of nearest same-cluster distance to the sum of intra and inter distances:

$$n_2(x) = \frac{d_{\text{intra}}}{d_{\text{intra}} + d_{\text{inter}}}$$

Averaged over all samples in $c$.

| | |
|---|---|
| **Complexity** | $O(nk)$ per pair |
| **Range** | $[0, 1]$ |
| **→ 0** | Intra-cluster neighbors are far closer than class-$j$ neighbors → well-separated |
| **→ 0.5** | Intra and inter distances are equal → no local separation |
| **→ 1** | Class-$j$ neighbors are much closer than same-cluster neighbors → severe overlap |

---

### `n3_min` / `n3_mean` / `n3_max`

**1-NN error rate**

For each sample $x \in c$, find its nearest neighbor in $c \cup j$ (excluding $x$ itself). Error if that neighbor belongs to class $j$.

$$n_3(c, j) = \frac{|\{x \in c : \mathrm{NN}_{c \cup j}(x) \in j\}|}{|c|}$$

| | |
|---|---|
| **Complexity** | $O(nk)$ per pair |
| **Range** | $[0, 1]$ |
| **→ 0** | Nearest neighbor is always from the same cluster → clean local boundary |
| **→ 1** | Nearest neighbor is always from class $j$ → cluster $c$ is embedded inside class $j$ |

---

### `n4_min` / `n4_mean` / `n4_max`

**k-NN majority-vote error rate**

For each sample $x \in c$, count votes from same-cluster vs class-$j$ neighbors. Misclassified if $j$-votes > $c$-votes.

$$n_4(c, j) = \frac{|\{x \in c : \text{votes}_j(x) > \text{votes}_c(x)\}|}{|c|}$$

where $\text{votes}_j(x) = |\{nb \in NN_k(x) : nb \in j\}|$ and $\text{votes}_c(x) = |\{nb \in NN_k(x) : nb \in c,\; nb \ne x\}|$.

| | |
|---|---|
| **Complexity** | $O(nk)$ per pair |
| **Range** | $[0, 1]$ |
| **→ 0** | Majority of neighbors always from same cluster |
| **→ 1** | Majority of neighbors always from class $j$ |

---

## Family ND — Network density

### `network_density_min` / `network_density_mean` / `network_density_max`

**Cross-class k-NN density**

Fraction of k-NN edges from cluster $c$ pointing to samples of class $j$:

$$\text{density}(c, j) = \frac{\sum_{x \in c} |\{nb \in NN(x) : nb \in j\}|}{|c| \times k}$$

| | |
|---|---|
| **Complexity** | $O(nk)$ per pair |
| **Range** | $[0, 1]$ |
| **→ 0** | No k-NN edges cross to class $j$ → isolated cluster |
| **→ 1** | All k-NN edges from cluster $c$ point to class $j$ → fully intermixed |

---

## Family T — Dimensionality

Per-cluster dimensionality measures. Computed on the subset of samples belonging to each cluster.

### `t2`

**Feature-to-sample ratio**

$$T_2 = \frac{d_{\text{num}} + d_{\text{cat}}}{n_c}$$

| | |
|---|---|
| **Complexity** | $O(1)$ per cluster |
| **Range** | $(0, +\infty)$ (typically $\ll 1$ for large clusters) |
| **→ 0** | Many more samples than features → low dimensionality risk |
| **→ large** | More features than cluster samples → curse of dimensionality |

---

### `t3`

**PCA intrinsic dimensionality ratio**

Fraction of numerical features needed to explain 95% of variance within the cluster:

$$T_3 = \frac{n_{\text{PCA}_{95\%}}}{d_{\text{num}}}$$

| | |
|---|---|
| **Complexity** | $O(n_c d^2)$ PCA fit per cluster |
| **Range** | $(0, 1]$ |
| **→ 0** | High feature redundancy — few PCA components suffice |
| **→ 1** | Features are nearly independent — most components needed for 95% variance |

---

### `t4`

**PCA components-to-sample ratio**

$$T_4 = \frac{n_{\text{PCA}_{95\%}}}{n_c}$$

| | |
|---|---|
| **Complexity** | $O(n_c d^2)$ PCA fit (shared with T3) per cluster |
| **Range** | $(0, 1]$ (typically $\ll 1$) |
| **→ 0** | Cluster is very large relative to its intrinsic dimensionality |
| **→ 1** | Effective dimensionality is comparable to cluster size → underdetermination risk |

---

## Family G — Geometry

Per-cluster geometric properties computed in Euclidean space on `X_num`.
Silhouette is approximated via stratified sampling (up to 10 000 points).

### `max_dispersion`

Maximum Euclidean distance from any sample in the cluster to its centroid.

$$\text{maxDisp}(c) = \max_{x \in c} \|x - \mu_c\|_2$$

| | |
|---|---|
| **Complexity** | $O(|c| \cdot d)$ |
| **Range** | $[0, +\infty)$ |
| **→ 0** | All samples are concentrated at the centroid |
| **→ large** | At least one outlier far from the centroid → diffuse or elongated cluster |

---

### `dist_to_nearest_foreign_cluster`

Minimum centroid-to-centroid Euclidean distance from cluster $c$ to any cluster of a different class.

$$d_{\text{foreign}}(c) = \min_{c'\, :\, \text{class}(c') \ne \text{class}(c)} \|\mu_c - \mu_{c'}\|_2$$

| | |
|---|---|
| **Complexity** | $O(C^2 d)$ pairwise centroids, computed once |
| **Range** | $[0, +\infty)$ |
| **→ 0** | A foreign cluster centroid is almost coincident → high collision risk |
| **→ large** | Well-separated from all foreign clusters |

---

### `p5_silhouette`

5th percentile of silhouette scores (Euclidean, computed on non-noise points with cluster labels).

$$s(x) = \frac{b(x) - a(x)}{\max(a(x), b(x))}$$

where $a(x)$ = mean intra-cluster distance, $b(x)$ = mean distance to nearest foreign cluster.

| | |
|---|---|
| **Complexity** | $O(n^2 d)$ exact; approximated via stratified subsample |
| **Range** | $[-1, 1]$ |
| **→ -1** | The worst 5% of samples are strongly misassigned |
| **→ 1** | Even the worst 5% of samples are well-placed inside their cluster |

---

### `frac_at_risk`

Fraction of cluster-$c$ samples with silhouette score $< 0$ (i.e., closer to a foreign cluster centroid than to their own).

$$\text{fracAtRisk}(c) = \frac{|\{x \in c : s(x) < 0\}|}{|c|}$$

| | |
|---|---|
| **Complexity** | Same as `p5_silhouette` (shared computation) |
| **Range** | $[0, 1]$ |
| **→ 0** | All samples are correctly placed relative to cluster boundaries |
| **→ 1** | Every sample is closer to a foreign cluster than to its own → severely mixed cluster |

---

### `min_sibling_centroid_dist`

Minimum centroid-to-centroid Euclidean distance from cluster $c$ to any other cluster of the **same class** (i.e., a sibling cluster).

$$d_{\text{sibling}}(c) = \min_{c' \ne c\, :\, \text{class}(c') = \text{class}(c)} \|\mu_c - \mu_{c'}\|_2$$

`None` if the class has only one cluster.

| | |
|---|---|
| **Complexity** | $O(C^2 d)$ pairwise centroids, shared with `dist_to_nearest_foreign_cluster` |
| **Range** | $[0, +\infty)$ |
| **→ 0** | Two sub-clusters of the same class nearly overlap → possible over-segmentation |
| **→ large** | Sibling clusters are well spread across the feature space |
