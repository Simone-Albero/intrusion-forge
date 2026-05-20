# Identifying Difficult Regions of the Data Space Prior to Classification

---

## Objective

The goal of this procedure is to identify regions of a labelled dataset that are **inherently difficult to classify before any classifier is trained**. The underlying hypothesis is that the geometric structure of the data space is itself predictive of classification failure: regions where classes overlap in feature space are hard to separate regardless of the learning algorithm. By characterising these regions in advance, it becomes possible to diagnose classification difficulty at its source — the data — rather than attributing failure post-hoc to model capacity or training dynamics.

The procedure is classifier-agnostic and is designed to handle tabular datasets with mixed feature types, imbalanced class distributions, and multimodal class structure. It consists of four sequential stages, depicted schematically in Figure 1:

1. **Intra-class clustering** — decompose the data space into compact, homogeneous sub-regions.
2. **Complexity measurement** — quantify the separability of each sub-region against all adversarial classes.
3. **Classifier training** — train an independent downstream classifier on the full dataset.
4. **Failure–complexity correlation** — assess whether pre-training complexity scores predict post-training failure rates.

![Pipeline overview](pipeline.png)
*Figure 1. Schematic overview of the pipeline. Complexity analysis (Steps 1–2) is fully independent of classifier training (Step 3). The correlation in Step 4 closes the loop and validates the predictive procedure.*

---

## Step 1 — Intra-Class Clustering

The first step partitions each class into a set of compact, internally coherent sub-regions, which become the atomic units of all subsequent analysis.

**Why not analyse at the class level.** Many real-world classes are multimodal: a single class may occupy multiple disconnected or geometrically distinct sub-regions of the feature space, each with different proximity relationships to other classes. Aggregating complexity measures at the class level would conflate these differences and produce a single, uninformative summary. By operating at the cluster level, the procedure can separately characterise the easy sub-regions of a class (well-separated from all other classes) and the hard ones (interleaved with foreign samples), providing a much finer-grained diagnostic.

**Why not analyse at the instance level.** Pushing the granularity to individual samples is equally problematic, for the opposite reason. Instance-level complexity measures are highly sensitive to noise and local density fluctuations: a single outlier or mislabelled sample can appear critically difficult while being entirely unrepresentative of any broader structural problem. More importantly, the complexity measures of interest — silhouette, MST boundary fraction, Fisher ratio — are inherently statistical quantities defined over populations of samples; they are either undefined or degenerate when applied to a single point. Clusters provide the minimum meaningful resolution at which these measures can be estimated reliably, while remaining compact enough to localise difficulty within the data space.

**Algorithm.** Clustering is performed independently per class using **HDBSCAN** (Hierarchical Density-Based Spatial Clustering of Applications with Noise). The choice is motivated by three properties: (1) HDBSCAN does not require a predetermined number of clusters — it infers the cluster structure directly from the data density; (2) it handles irregular cluster shapes and varying local density, which are common in real network traffic data; (3) it explicitly labels low-density samples as **noise** (cluster label $-1$), isolating them from well-defined sub-regions rather than forcing them into a nearest cluster. The two key hyperparameters (`min_cluster_size`, `min_samples`) are selected per class via an internal grid search maximising mean silhouette score.

**Feature scope.** Clustering operates exclusively on the processed numerical features. Including categorical features in the clustering distance would require computing Gower distance over the full mixed-type space for every pair of samples during the HDBSCAN fit — an $O(n^2 d)$ operation that is computationally prohibitive at dataset scale and difficult to approximate without sacrificing the density estimates that HDBSCAN relies on. Numerical features, after log-scaling and robust normalisation, provide a well-defined Euclidean space in which density-based clustering is both tractable and geometrically meaningful. Categorical features are incorporated later, in Step 2, where a sparse neighbourhood graph — rather than a full pairwise matrix — is sufficient.

**Noise handling.** Samples that HDBSCAN cannot assign to any density-connected region receive a provisional label of $-1$. Rather than discarding them, noise points are collected per class and merged into a single per-class pseudo-cluster, which is assigned a fresh globally unique identifier. This ensures that every sample participates in the subsequent analysis: no data is lost, and the pseudo-clusters act as catch-all regions that aggregate the low-density, structurally ambiguous portions of each class.

**Output.** Each sample in the dataset is assigned a globally unique cluster label. Cluster identifiers are offset across classes to avoid collisions, so each label unambiguously identifies both the sub-region and, by construction, the class it belongs to.

---

## Step 2 — Complexity Measures

Each cluster $c$ is characterised by a vector of complexity measures, organised into five families (F, N, ND, T, G). All measures are computed on the processed feature representations (after log-scaling and robust normalisation of numerical features, and hash encoding of categoricals), making them directly comparable across datasets and runs.

For measures that are inherently pairwise — defined with respect to a specific adversarial class $j \ne \text{class}(c)$ — the cluster-level value is reported as the **minimum** (worst-case adversary), **mean** (average difficulty), and optionally **maximum** (easiest adversary) over all adversarial classes. Reporting the minimum is particularly important: a cluster may be easy to separate from most classes but critically exposed to one specific adversary, and this worst case is the one most likely to drive classifier failure.

**Distance metric for neighbourhood families.** The neighbourhood-based families (N, ND) require a dissimilarity measure that covers both numerical and categorical features. A **Gower-hybrid distance** is used: the numerical component is configurable (cosine or Euclidean, set via `complexity.distance`) and is combined with a Hamming-style indicator contribution on categoricals, with each contribution normalised to $[0, 1]$:

$$d_\text{hybrid}(x, x') = \frac{1}{d_\text{num} + d_\text{cat}} \left( \sum_{f=1}^{d_\text{num}} \delta_\text{num}(x_f, x'_f) + \sum_{f=1}^{d_\text{cat}} \mathbb{1}[x_f \ne x'_f] \right)$$

Computing the full pairwise distance matrix would be $O(n^2 d)$, which is prohibitive at dataset scale. Instead, only a sparse $k$-nearest-neighbour graph (default `complexity.k = 30`) is materialised — distances are computed in row batches and only the $k$ closest neighbours per sample are retained. This graph is constructed once and shared across all neighbourhood-based families, amortising its cost.

### Family F — Feature-Based Separability

These four measures assess how well the numerical features discriminate cluster $c$ from adversarial classes. They are computed on robust-scaled numerical features only, without any neighbourhood structure.

**F1 — Fisher discriminant ratio.** Normalised inverse of the maximum per-feature Fisher ratio. A value near zero means at least one feature provides strong linear separation; near one means no feature separates the two groups.

$$f_1(c, j) = \frac{1}{1 + \max_f \frac{(\mu_{c,f} - \mu_{j,f})^2}{\sigma^2_{c,f} + \sigma^2_{j,f}}}$$

**F2 — Bounding-box overlap ratio.** Mean fraction of per-feature value range shared between cluster $c$ and class $j$. Zero means no marginal overlap; one means complete overlap on every feature.

$$f_2(c, j) = \frac{1}{d} \sum_f \frac{\max(0,\, \min(c_{\max,f}, j_{\max,f}) - \max(c_{\min,f}, j_{\min,f}))}{\max(c_{\max,f}, j_{\max,f}) - \min(c_{\min,f}, j_{\min,f})}$$

**F3 — Best single-feature separability.** Fraction of cluster-$c$ samples inside the feature-wise overlap region, minimised over features. A low value implies the existence of at least one feature that fully separates the cluster; a high value means every feature partially confounds cluster $c$ with class $j$.

$$f_3(c, j) = \min_f \frac{|\{x \in c : lo_f \le x_f \le hi_f\}|}{|c|}$$

**F4 — Joint-feature overlap fraction.** Fraction of cluster-$c$ samples that fall inside the overlap region on all features simultaneously. Unlike F3, this captures joint multi-dimensional overlap and is insensitive to marginal separability.

$$f_4(c, j) = \frac{|\{x \in c : \forall f,\; lo_f \le x_f \le hi_f\}|}{|c|}$$

### Family N — Neighbourhood-Based Separability

These measures characterise local class structure in the k-NN graph (Gower distance, $k = 5$), capturing class intermingling at the sample level. All four share the same graph and are therefore computed simultaneously. Noise samples are excluded from cluster membership but may appear as neighbours.

**N1 — MST boundary fraction.** Fraction of cluster-$c$ samples adjacent to a class-$j$ sample in an approximate minimum spanning tree over $c \cup j$.

$$n_1(c, j) = \frac{|\{x \in c : \exists\, (x, y) \in \text{MST},\; y \in j\}|}{|c|}$$

**N2 — Intra/inter nearest-neighbour distance ratio.** For each $x \in c$, ratio of its nearest same-cluster distance to the sum of intra- and inter-cluster distances, averaged over the cluster. A value of 0.5 indicates that the local neighbourhood provides no discriminative signal; values above 0.5 indicate that adversarial samples are locally closer than same-cluster samples.

$$n_2(x) = \frac{d_{\text{intra}}}{d_{\text{intra}} + d_{\text{inter}}}$$

**N3 — 1-NN error rate.** For each $x \in c$, probability that its nearest neighbour in $c \cup j$ belongs to class $j$. Equivalent to the empirical error of a 1-NN classifier restricted to the binary problem $c$ vs. $j$.

$$n_3(c, j) = \frac{|\{x \in c : \mathrm{NN}_{c \cup j}(x) \in j\}|}{|c|}$$

**N4 — k-NN majority-vote error rate.** Fraction of cluster-$c$ samples for which the majority of their $k$ nearest neighbours in $c \cup j$ belong to class $j$. More robust to local outliers than N3.

$$n_4(c, j) = \frac{|\{x \in c : \text{votes}_j(x) > \text{votes}_c(x)\}|}{|c|}$$

### Family ND — Network Density

**Cross-class k-NN density.** Fraction of k-NN edges from cluster $c$ that point to class-$j$ samples. Whereas the N family reasons per sample, this measure gives a global summary of how strongly cluster $c$ is coupled to class $j$ in the neighbourhood graph.

$$\text{density}(c, j) = \frac{\sum_{x \in c} |\{nb \in \mathrm{NN}(x) : nb \in j\}|}{|c| \times k}$$

### Family T — Intrinsic Dimensionality

These three measures assess the dimensionality regime of the cluster relative to its size and the feature space, providing a proxy for the curse of dimensionality within the cluster.

**T2 — Feature-to-sample ratio.** Ratio of total features (numerical and categorical) to cluster size. Values significantly greater than one indicate an underdetermined regime where reliable generalisation within the cluster becomes questionable.

$$T_2 = \frac{d_{\text{num}} + d_{\text{cat}}}{n_c}$$

**T3 — PCA intrinsic dimensionality ratio.** Fraction of numerical features needed to explain 95% of intra-cluster variance. Low values indicate high feature redundancy; values close to one indicate that features carry nearly independent information.

$$T_3 = \frac{n_{\text{PCA}_{95\%}}}{d_{\text{num}}}$$

**T4 — PCA components-to-sample ratio.** Ratio of the number of PCA components required for 95% variance to the cluster size. High values indicate that the effective dimensionality of the cluster is comparable to its population.

$$T_4 = \frac{n_{\text{PCA}_{95\%}}}{n_c}$$

### Family G — Cluster Geometry

These six measures characterise geometric properties of the cluster in the metric space configured via `complexity.distance` (numerical features only). Silhouette-based quantities are approximated via stratified subsampling (at most 10,000 points) to manage the $O(n^2 d)$ cost of exact pairwise distance computation.

**Maximum dispersion.** Maximum distance from any cluster sample to the centroid $\mu_c$. Indicates whether the cluster is compact or diffuse.

$$\text{maxDisp}(c) = \max_{x \in c} \|x - \mu_c\|$$

**95th-percentile dispersion.** Outlier-robust counterpart to maximum dispersion — the 95th percentile of sample-to-centroid distances.

**Distance to nearest foreign cluster.** Minimum centroid-to-centroid Euclidean distance from $c$ to any cluster of a different class. A small value indicates high spatial collision risk.

$$d_{\text{foreign}}(c) = \min_{c'\, :\, \text{class}(c') \ne \text{class}(c)} \|\mu_c - \mu_{c'}\|_2$$

**5th-percentile silhouette score.** The 5th percentile of the silhouette distribution within the cluster, computed on non-noise samples. Focusing on the 5th percentile rather than the mean highlights the most boundary-exposed samples.

$$s(x) = \frac{b(x) - a(x)}{\max(a(x), b(x))}, \quad a(x) = \text{mean intra-cluster dist}, \quad b(x) = \text{mean dist to nearest foreign cluster}$$

**Fraction at risk.** Fraction of cluster samples with silhouette score $< 0$, i.e., samples closer to a foreign cluster than to their own centroid. Directly quantifies the proportion of boundary-ambiguous samples.

$$\text{fracAtRisk}(c) = \frac{|\{x \in c : s(x) < 0\}|}{|c|}$$

**Minimum sibling centroid distance.** Minimum centroid distance from $c$ to any other cluster of the same class. Small values indicate that the class is internally fragmented into closely-spaced sub-regions, which may reflect genuine multimodality or simply the effect of HDBSCAN over-segmentation.

$$d_{\text{sibling}}(c) = \min_{c' \ne c\, :\, \text{class}(c') = \text{class}(c)} \|\mu_c - \mu_{c'}\|_2$$

---

## Step 3 — Classifier Training

A downstream classifier is trained on the preprocessed dataset using a standard supervised learning procedure. The pipeline supports both classical ML estimators (sklearn, XGBoost) and deep neural networks (PyTorch + Ignite, with early stopping on validation loss and best-model checkpointing); the choice is controlled by the `classifier` config group. Training is intentionally performed **after** the complexity analysis is complete: the complexity measures are a pre-training diagnostic and carry no information from the classifier.

**Failure rate per cluster.** After evaluation on the held-out test set, a failure rate is computed for each cluster as the fraction of test samples in that cluster for which the classifier produces an incorrect prediction:

$$\text{failure\_rate}(c) = \frac{|\{x \in c_{\text{test}} : \hat{y}(x) \ne y(x)\}|}{|c_{\text{test}}|}$$

This metric directly links the cluster partition established in Step 1 to the classifier's observed behaviour, enabling a principled comparison between pre-training complexity and post-training performance.

---

## Step 4 — Failure–Complexity Correlation

The final step asks whether the complexity vector computed in Step 2 is predictive of the failure rate computed in Step 3. This is operationalised as a binary classification problem: each cluster is labelled *failed* if its failure rate exceeds a configurable threshold $\tau$, and *correct* otherwise. A **Random Forest** is then trained to predict this binary label from the cluster's complexity feature vector.

**Why Random Forest.** The number of clusters is typically in the tens to low hundreds — far too few for a deep model. A Random Forest is robust to this regime, naturally handles mixed-importance features, and provides interpretable feature importances that identify which complexity dimensions are most predictive of failure for a given dataset.

**Evaluation with nested cross-validation.** To obtain unbiased estimates given the small sample size, the evaluation uses a **nested cross-validation** scheme. An outer loop ($k = 5$ stratified folds) generates a complete set of out-of-fold predictions — one prediction per cluster, with no data leakage. Within each outer training fold, an inner loop ($k = 5$ stratified folds) selects the best hyperparameter configuration (over `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`) by maximising F1 score. Class imbalance between failed and correct clusters is handled via balanced class weights.

**Interpretation.** The two canonical patterns are:

- **Failed cluster, high complexity**: the classifier's failure is geometrically explained. The cluster occupies an intrinsically ambiguous region of the data space, and difficulty was predictable before any model was trained.
- **Correct cluster, low complexity**: the classifier succeeds because the region is geometrically clean. The complexity analysis correctly anticipated the absence of difficulty.

Deviations from these patterns are equally informative: a geometrically simple cluster that is nonetheless misclassified points to a model-specific weakness; a geometrically complex cluster that is correctly handled suggests the classifier is exploiting a structure that the current measure battery does not capture. Feature importances aggregated across folds further reveal which complexity dimensions drive the correlation for a specific dataset.
