# Synthetic Test Dataset — Class Specifications

**File**: `synthetic_test.csv`
**Shape**: 102,500 rows × 23 columns
**Features**: `num_1`..`num_20` (numerical), `cat_1` (A/B/C), `cat_2` (X/Y/Z/W), `label`
**Seed**: 42

---

## Feature Layout

| Range | Role |
|---|---|
| `num_1`..`num_15` | Per-class discriminating features — each class shifts a distinct subset |
| `num_16`..`num_20` | Intra-class sub-cluster features — reserved; not used for inter-class discrimination |

**Sub-cluster mechanism**: in every class (except class_11), the last 40% of rows have `num_16`..`num_20` shifted from N(5, 1) to N(20, 1). This creates two density peaks within each class, enabling per-class HDBSCAN to find ≥ 2 sub-clusters.

---

## Class Summary

| Class | N | Discriminating features | Pipeline purpose |
|---|---|---|---|
| class_1 | 35,000 | none (pure baseline) | reference distribution |
| class_2 | 10,000 | num_1,2,5,6,11,12,13 + cat_2=W | hard pair with class_3 — identical numericals, differ only in cat_2 |
| class_3 | 10,000 | num_1,2,5,6,11,12,13 + cat_2=Z | hard pair with class_2 |
| class_4 | 8,000 | num_1,2,7,14,15 high; num_5,6 ≈ 0 | well-separated block A |
| class_5 | 7,000 | num_3,4,8,9,10 high | well-separated block B |
| class_6 | 6,000 | num_5,6,9 high; num_1,7 ≈ 0 | well-separated block C |
| class_7 | 8,000 | num_3,4,11,12 moderately shifted | overlap pair A with class_8 |
| class_8 | 6,000 | same center as class_7, wider σ | overlap pair B — intentional boundary blur |
| class_9 | 5,000 | num_5,6 extreme; num_1,2 ≈ 0 | well-separated block D |
| class_10 | 7,000 | num_1,2,9 slight shift from baseline | partial overlap with class_1 → FP/FN stress test |
| class_11 | 500 | none (pure baseline) | filtered out by `rare_category_filter` (< min_count=3000) |
| NaN/Inf rows | ~500 | — | injected into `num_7`/`num_8` (Inf) and random num cols (NaN) → removed by `drop_nans` |


---

## Per-Class Detail

All numerical features start from N(5, 1). Only the shifted features are listed; everything else stays at baseline.
Sub-cluster columns num_16..num_20 are always shifted (see Feature Layout above) and are omitted from the tables below.

---

### class_1 — Baseline
**N**: 35,000

Pure baseline — no discriminating shift on any feature. Used as reference distribution.

| Feature | Distribution |
|---|---|
| `num_1`..`num_15` | N(5, 1) |
| `cat_1` | A 60%, B 30%, C 10% |
| `cat_2` | X 40%, Y 30%, Z 15%, W 15% |

---

### class_2 — Hard-pair A
**N**: 10,000 | **Only structural difference from class_3**: `cat_2 = W`

| Feature | Distribution |
|---|---|
| `num_1` | N(20, 1) |
| `num_2` | N(18, 1) |
| `num_5` | N(1, 0.3).clip(0) |
| `num_6` | N(1, 0.3).clip(0) |
| `num_11` | U(1, 3) |
| `num_12` | U(1, 3) |
| `num_13` | N(18, 0.8) |
| `cat_1` | A 80%, B 15%, C 5% |
| `cat_2` | **W 100%** |

---

### class_3 — Hard-pair B
**N**: 10,000 | **Only structural difference from class_2**: `cat_2 = Z`

Identical numerical distributions to class_2.

| Feature | Distribution |
|---|---|
| `num_1`..`num_13` | identical to class_2 |
| `cat_1` | A 80%, B 15%, C 5% |
| `cat_2` | **Z 100%** |

---

### class_4 — Block A
**N**: 8,000 | **Key features**: num_1,2,7,14,15 high; num_5,6 ≈ 0

| Feature | Distribution |
|---|---|
| `num_1` | N(30, 1.5) |
| `num_2` | N(28, 1.5) |
| `num_5` | U(0, 0.5) |
| `num_6` | U(0, 0.5) |
| `num_7` | N(25, 1.5) |
| `num_14` | N(22, 1) |
| `num_15` | N(20, 1) |
| `cat_1` | A 50%, B 40%, C 10% |
| `cat_2` | X 50%, Y 40%, Z 5%, W 5% |

---

### class_5 — Block B
**N**: 7,000 | **Key features**: num_3,4,8,9 high; num_10 tightly concentrated at 35

| Feature | Distribution |
|---|---|
| `num_3` | N(28, 1) |
| `num_4` | N(25, 1) |
| `num_8` | N(22, 1) |
| `num_9` | N(20, 1) |
| `num_10` | N(35, 0.3) |
| `cat_1` | A 40%, B 50%, C 10% |
| `cat_2` | X 30%, Y 50%, Z 10%, W 10% |

---

### class_6 — Block C
**N**: 6,000 | **Key features**: num_5,6,9 high; num_1,7 ≈ 0

| Feature | Distribution |
|---|---|
| `num_1` | U(0, 0.5) |
| `num_5` | N(35, 2) |
| `num_6` | N(30, 2) |
| `num_7` | N(0.3, 0.1).clip(0) |
| `num_9` | N(28, 2) |
| `cat_1` | A 70%, B 20%, C 10% |
| `cat_2` | X 60%, Y 10%, Z 10%, W 20% |

---

### class_7 — Overlap-pair A
**N**: 8,000 | **Key features**: num_3,4,11,12 moderately shifted

| Feature | Distribution |
|---|---|
| `num_3` | N(14, 1.5) |
| `num_4` | N(13, 1.5) |
| `num_11` | N(15, 1.5) |
| `num_12` | N(13, 1) |
| `cat_1` | A 55%, B 35%, C 10% |
| `cat_2` | X 35%, Y 25%, Z 20%, W 20% |

---

### class_8 — Overlap-pair B
**N**: 6,000 | **Same center as class_7, wider σ → intentional overlap**

| Feature | Distribution |
|---|---|
| `num_3` | N(14, 4).clip(0) |
| `num_4` | N(13, 4).clip(0) |
| `num_11` | N(15, 1.5) |
| `num_12` | N(13, 2.5) |
| `cat_1` | A 55%, B 35%, C 10% |
| `cat_2` | X 30%, Y 30%, Z 20%, W 20% |

> class_7 and class_8 share the same mean on num_3,4,11,12. class_8's wider variance blurs the decision boundary.

---

### class_9 — Block D
**N**: 5,000 | **Key features**: num_5,6 extreme; num_1,2 ≈ 0

| Feature | Distribution |
|---|---|
| `num_1` | U(0, 0.3) |
| `num_2` | N(0.3, 0.1).clip(0) |
| `num_5` | N(50, 4) |
| `num_6` | N(55, 5) |
| `cat_1` | A 70%, B 20%, C 10% |
| `cat_2` | X 50%, Y 10%, Z 10%, W 30% |

---

### class_10 — Partial overlap with class_1
**N**: 7,000 | **Key features**: num_1,2,9 slightly above baseline → FP/FN stress test

| Feature | Distribution |
|---|---|
| `num_1` | N(7, 1.2) |
| `num_2` | N(7.5, 1.2) |
| `num_9` | N(7, 1) |
| `cat_1` | A 60%, B 30%, C 10% |
| `cat_2` | X 70%, Y 10%, Z 5%, W 15% |

---

### class_11 — Rare (filtered)
**N**: 500

Pure baseline on all features. No sub-cluster shift applied. **Automatically removed** by `rare_category_filter` during `prepare_data.py` because 500 < `min_cat_count: 3000`.

---

### Edge Cases — NaN/Inf (filtered)
**N**: ~500 modified rows (not a distinct class)

Injected into existing rows before saving:
- **300 Inf**: on `num_7` or `num_8` (randomly assigned)
- **200 NaN**: on random columns among `num_1`..`num_20`

Removed by `drop_nans` in the first preprocessing step.

---

## Usage

```bash
# Generate the CSV (from project root, with venv active)
python resources/raw_data/synthetic/generate_synthetic.py

# Preprocessing pipeline
make prepare DATA=synthetic_test NAME=test_run
```
```

To use only `num_cols` (ignoring `cat_1`, `cat_2`), comment out `cat_cols` in `configs/data/synthetic_test.yaml` and switch to `model=numerical_classifier`.
