# Synthetic Test Dataset — Specification

**File**: `synthetic_test.csv`  
**Shape**: ~69 500 rows × 24 columns  
**Features**: `num_1`..`num_20` (numerical), `cat_1` (A/B/C), `cat_2` (X/Y/Z/W), `label`, `true_subgroup`  
**Seed**: 42

---

## Design Rationale

The dataset is designed to demonstrate the core pipeline hypothesis end-to-end:

> *Pre-training geometric complexity predicts post-training classification failure rate, at cluster granularity.*

The previous design created intra-class sub-clusters by shifting `num_16..num_20` — an orthogonal feature subspace not used for inter-class discrimination. This made all sub-clusters of a class geometrically identical relative to adversarial classes, so complexity measures showed no intra-class variation and the failure RF had nothing to learn.

**New design**: each attack class has two or three sub-groups positioned at different distances from the nearest adversarial class, along the same feature dimensions that define the inter-class boundary.

```
canonical sub-group:  center = class_center                      (α = 0.0)
medium sub-group:     center = 0.5 × class_center + 0.5 × adv   (α = 0.5)
evasive sub-group:    center = 0.2 × class_center + 0.8 × adv   (α = 0.8)
```

After per-class clustering, HDBSCAN should find the three density peaks (they are separated by ≥ 5σ within each class). The complexity measures (N3/N4 k-NN error rate, F1 Fisher ratio, G silhouette) should rank them correctly. The classifier should make more errors on evasive sub-clusters, producing a high Spearman ρ in Step 4.

**Ground truth**: `true_subgroup` ("canonical" / "medium" / "evasive") is present in the CSV for post-hoc validation but is not listed in `num_cols` / `cat_cols` and is therefore ignored by the pipeline.

---

## Feature Layout

All 20 numerical features can participate in both inter-class and intra-class geometry. There is no reserved orthogonal subspace.

| Feature block | Primarily used by |
|---|---|
| `num_1`, `num_2`, `num_3` | class_2 (attack A) canonical discriminators |
| `num_1`, `num_2`, `num_4` | class_3 (attack B, variant) — shares `num_1`/`num_2` with class_2 |
| `num_5`, `num_6`, `num_7` | class_4 (attack C) |
| `num_8`, `num_9`, `num_10` | class_5 (overlap A) |
| `num_8`, `num_9`, `num_11` | class_6 (overlap B) — shares `num_8`/`num_9` with class_5 |
| `num_12`, `num_13`, `num_14` | class_7 (attack D) |
| `num_15` | class_8 (categorical) intra-class sub-cluster drift |
| `num_1`, `num_2`, `num_15` | class_9 (stealth) — partially overlaps class_2/3 space |
| `num_16`..`num_20` | not discriminating for any class; remain at baseline |

Shared feature blocks are intentional: class_2 and class_3 both use `num_1`/`num_2`, creating natural intra-attack confusion; class_5 and class_6 share `num_8`/`num_9`, forming the overlap pair.

---

## Class Summary

| Class | N | Sub-groups | Adversary | Pipeline purpose |
|---|---|---|---|---|
| class_1 | 22 000 | 2: 80% canonical, 20% evasive | class_2 | Benign baseline; near-attack sub-group creates confusion zone on the benign side |
| class_2 | 9 000 | 3: 50% / 30% / 20% (α = 0, 0.5, 0.8) | class_1 | Attack A — full canonical→medium→evasive gradient; reference case for the hypothesis |
| class_3 | 7 000 | 3: 50% / 30% / 20% | class_1 | Attack B, variant of A — shares `num_1`/`num_2` with class_2, uses `num_4` not `num_3` |
| class_4 | 6 000 | 3: 50% / 30% / 20% | class_1 | Attack C — well-separated block; evasive sub-group tests boundary detectability |
| class_5 | 5 000 | 3: 50% / 30% / 20% | class_6 | Overlap pair A — evasive sub-group falls deep into class_6 territory |
| class_6 | 5 000 | 3: 50% / 30% / 20% | class_5 | Overlap pair B — same canonical center as class_5 (different `num_11`), wider σ |
| class_7 | 5 000 | 3: 50% / 30% / 20% | class_1 | Attack D — distinct feature block from class_4; evasive toward benign |
| class_8 | 5 000 | 2: 60% canonical, 40% medium | class_1 | Categorical separator — numerically at benign baseline; `cat_2=W` is the only signal |
| class_9 | 5 000 | 3: 20% canonical, 40% medium, 40% evasive | class_1 | Stealth — majority evasive; tests behaviour when most of a class is near-benign |
| class_10 | 500 | 1 canonical | — | Rare class — filtered out by `rare_category_filter` (500 < `min_cat_count`=3000) |
| NaN/Inf rows | ~500 | — | — | Injected into `num_7`/`num_8` (Inf) and random num cols (NaN); removed by `drop_nans` |

---

## Per-Class Detail

Baseline: all `num_i` ~ N(5, 1). Only discriminating feature overrides are listed; unspecified features stay at baseline.

σ refers to the within-sub-group standard deviation of the discriminating features.

---

### class_1 — Benign

**N**: 22 000

| Sub-group | N | num_1 | num_2 | num_3 | σ | cat_1 | cat_2 |
|---|---|---|---|---|---|---|---|
| canonical | 17 600 | N(5, 1) | N(5, 1) | N(5, 1) | 1.0 | A 60%, B 30%, C 10% | X 40%, Y 35%, Z 15%, W 10% |
| evasive | 4 400 | N(11, 1.2) | N(10.1, 1.2) | N(9.5, 1.2) | 1.2 | same | same |

The evasive sub-group is displaced α=0.3 toward class_2's canonical center. These benign samples are closest to attack_A territory and are the primary source of false positives.

---

### class_2 — Attack A

**N**: 9 000 | **Key features**: num_1, num_2, num_3 | **Adversary**: class_1 (benign)

| Sub-group | N | num_1 | num_2 | num_3 | σ |
|---|---|---|---|---|---|
| canonical | 4 500 | N(25, 1.5) | N(22, 1.5) | N(20, 1.5) | 1.5 |
| medium | 2 700 | N(15, 1.5) | N(13.5, 1.5) | N(12.5, 1.5) | 1.5 |
| evasive | 1 800 | N(9, 1.5) | N(8.4, 1.5) | N(8, 1.5) | 1.5 |

3D separation: canonical↔medium ≈ 14 units ≈ 9σ; medium↔evasive ≈ 9 units ≈ 6σ.  
After log1p: ≈ 10σ and 6σ respectively — HDBSCAN finds all three peaks cleanly.

| cat_1 | cat_2 |
|---|---|
| A 80%, B 15%, C 5% | X 5%, Y 10%, Z 10%, W 75% |

---

### class_3 — Attack B (variant)

**N**: 7 000 | **Key features**: num_1, num_2, num_4 | **Adversary**: class_1

Shares `num_1`/`num_2` with class_2 (creating deliberate variant confusion). Uses `num_4` as third discriminator (class_2 uses `num_3`).

| Sub-group | N | num_1 | num_2 | num_4 | σ |
|---|---|---|---|---|---|
| canonical | 3 500 | N(24, 1.5) | N(20, 1.5) | N(22, 1.5) | 1.5 |
| medium | 2 100 | N(14.5, 1.5) | N(12.5, 1.5) | N(13.5, 1.5) | 1.5 |
| evasive | 1 400 | N(9.8, 1.5) | N(9, 1.5) | N(9.4, 1.5) | 1.5 |

| cat_1 | cat_2 |
|---|---|
| A 75%, B 20%, C 5% | X 5%, Y 10%, Z 35%, W 50% |

---

### class_4 — Attack C

**N**: 6 000 | **Key features**: num_5, num_6, num_7 | **Adversary**: class_1

Distinct feature block from class_2/3; well-separated at canonical.

| Sub-group | N | num_5 | num_6 | num_7 | σ |
|---|---|---|---|---|---|
| canonical | 3 000 | N(30, 1.5) | N(28, 1.5) | N(25, 1.5) | 1.5 |
| medium | 1 800 | N(17.5, 1.5) | N(16.5, 1.5) | N(15, 1.5) | 1.5 |
| evasive | 1 200 | N(9, 1.5) | N(8.6, 1.5) | N(9, 1.5) | 1.5 |

| cat_1 | cat_2 |
|---|---|
| A 50%, B 40%, C 10% | X 50%, Y 35%, Z 10%, W 5% |

---

### class_5 — Overlap pair A

**N**: 5 000 | **Key features**: num_8, num_9, num_10 | **Adversary**: class_6

Evasive sub-group is displaced 80% toward class_6's canonical center — deep inter-class overlap.

| Sub-group | N | num_8 | num_9 | num_10 | num_11 | σ |
|---|---|---|---|---|---|---|
| canonical | 2 500 | N(20, 1.5) | N(18, 1.5) | N(22, 1.5) | baseline | 1.5 |
| medium | 1 500 | N(21, 1.5) | N(16, 1.5) | N(19, 1.5) | N(10, 1.5) | 1.5 |
| evasive | 1 000 | N(21.6, 1.5) | N(14.8, 1.5) | N(16.8, 1.5) | N(16, 1.5) | 1.5 |

> Evasive sub-group is < 1σ from class_6's canonical center.

| cat_1 | cat_2 |
|---|---|
| A 40%, B 50%, C 10% | X 30%, Y 50%, Z 10%, W 10% |

---

### class_6 — Overlap pair B

**N**: 5 000 | **Key features**: num_8, num_9, num_11 | **Adversary**: class_5

Same centre on `num_8`/`num_9` as class_5; wider σ = 2.0 blurs the decision boundary.

| Sub-group | N | num_8 | num_9 | num_11 | num_10 | σ |
|---|---|---|---|---|---|---|
| canonical | 2 500 | N(22, 2.0) | N(14, 2.0) | N(20, 2.0) | baseline | 2.0 |
| medium | 1 500 | N(21, 2.0) | N(16, 2.0) | N(13, 2.0) | N(13, 2.0) | 2.0 |
| evasive | 1 000 | N(20.4, 2.0) | N(17.2, 2.0) | N(8.6, 2.0) | N(20.8, 2.0) | 2.0 |

| cat_1 | cat_2 |
|---|---|
| A 35%, B 50%, C 15% | X 30%, Y 50%, Z 10%, W 10% |

---

### class_7 — Attack D

**N**: 5 000 | **Key features**: num_12, num_13, num_14 | **Adversary**: class_1

Second well-separated attack; canonical block is in a different region from class_4.

| Sub-group | N | num_12 | num_13 | num_14 | σ |
|---|---|---|---|---|---|
| canonical | 2 500 | N(28, 1.5) | N(25, 1.5) | N(22, 1.5) | 1.5 |
| medium | 1 500 | N(16.5, 1.5) | N(15, 1.5) | N(13.5, 1.5) | 1.5 |
| evasive | 1 000 | N(9.6, 1.5) | N(9, 1.5) | N(8.6, 1.5) | 1.5 |

| cat_1 | cat_2 |
|---|---|
| A 55%, B 35%, C 10% | X 35%, Y 30%, Z 20%, W 15% |

---

### class_8 — Categorical separator

**N**: 5 000 | **Key feature**: cat_2=W exclusively | **Adversary**: class_1 (numerical only)

Numerically at the benign baseline; `cat_2=W` is the only signal. No evasive sub-group: moving numerically toward class_1 is impossible (the class IS numerically at class_1). Two sub-groups created via `num_15` drift so HDBSCAN finds distinct density peaks within the class.

| Sub-group | N | num_15 | σ | cat_2 |
|---|---|---|---|---|
| canonical | 3 000 | N(5, 1.0) | 1.0 | **W 100%** |
| medium | 2 000 | N(10, 1.5) | 1.5 | **W 100%** |

All other num at baseline. | cat_1 | A 70%, B 20%, C 10% |

---

### class_9 — Stealth

**N**: 5 000 | **Key features**: num_1, num_2, num_15 | **Adversary**: class_1

Inverted split: 80% of the class is near-benign (medium + evasive). The majority of samples will be hard to classify correctly, making this the highest-difficulty class overall.

| Sub-group | N | num_1 | num_2 | num_15 | σ |
|---|---|---|---|---|---|
| canonical | 1 000 | N(18, 1.5) | N(16, 1.5) | N(15, 1.5) | 1.5 |
| medium | 2 000 | N(11.5, 1.2) | N(10.5, 1.2) | N(10, 1.2) | 1.2 |
| evasive | 2 000 | N(8.6, 1.0) | N(8.2, 1.0) | N(8, 1.0) | 1.0 |

> Uses `num_1`/`num_2` (overlaps class_2/3 region); partial confusion is expected.

| cat_1 | cat_2 |
|---|---|
| A 60%, B 30%, C 10% | X 70%, Y 10%, Z 10%, W 10% |

---

### class_10 — Rare (filtered)

**N**: 500

Pure baseline on all features. **Automatically removed** by `rare_category_filter` because 500 < `min_cat_count: 3000`.

---

### Edge Cases — NaN/Inf

~500 modified rows (not a distinct class):
- **300 Inf**: on `num_7` or `num_8`
- **200 NaN**: on random columns among `num_1`..`num_20`

Removed by `drop_nans` in the first preprocessing step.

---

## Expected Pipeline Behaviour

After a full run on this dataset (prepare → classify → failure-classify):

| Cluster type | Expected complexity | Expected failure rate |
|---|---|---|
| Canonical sub-clusters | Low N3/N4, high F1 | ~0 |
| Medium sub-clusters | Moderate | Intermediate |
| Evasive sub-clusters | High N3/N4, low F1 | Significantly higher |
| class_8 (any) | Moderate numerical, high categorical | Depends on classifier handling of cat_2 |
| class_9 evasive | Very high (near-benign) | Near classifier baseline |

**Headline metric**: `classifier_results.json → spearman` should be ≥ 0.6, with N-family and F-family measures among the top feature importances.

The `true_subgroup` column can be used to verify that the clustering step correctly recovered the intended sub-group structure (cross-tabulate `true_subgroup` with the assigned cluster ids).

---

## Usage

```bash
# Generate the CSV (from project root, with venv active)
python generate_synthetic.py

# Full pipeline
make prepare  DATA=synthetic_test NAME=test_run
make classify DATA=synthetic_test NAME=test_run CLASSIFIER=random_forest
```
