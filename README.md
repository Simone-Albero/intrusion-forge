# Intrusion Forge

Tabular classification framework for network traffic — supports both classical ML (sklearn, XGBoost) and deep learning (PyTorch + Ignite) classifiers on cybersecurity datasets (UNSW-NB15, BoT-IoT, CIC-IDS-2018, ToN-IoT) and a handful of generic tabular benchmarks (Bank Marketing, Covertype, Letter Recognition, Statlog Landsat Satellite, Thyroid Disease).

The pipeline covers data preparation, classifier training, per-cluster complexity analysis, failure prediction, and plot rendering — all driven by a declarative Hydra configuration system.

---

## Project Structure

```
intrusion-forge/
├── pipelines/                       # Pipeline entry points (Hydra-driven)
│   ├── prepare_data.py              # Step 1 — preprocess raw CSV → parquet splits + per-class clustering
│   ├── classify.py                  # Step 2 — train & evaluate one ML or DL classifier
│   ├── compute_complexity.py        # Step 3a — per-cluster + per-class complexity (shared)
│   ├── fit_failure_classifier.py    # Step 3b — Random Forest predicting cluster failure
│   └── render_plots.py              # Step 4 — render figures from analysis artifacts
├── generate_synthetic.py            # Generate the synthetic test dataset
├── dashboard.py                     # Streamlit dashboard for browsing experiment outputs
├── Makefile                         # Experiment runner (prepare / classify / complexity / …)
│
├── configs/                         # Hydra configuration hierarchy
│   ├── config.yaml                  # Root config + global parameters
│   ├── data/                        # Dataset definitions (columns, splits, filtering)
│   ├── classifier/                  # ML and DL classifier configs
│   ├── loops/                       # DL training/validation loop settings
│   ├── loss/                        # DL loss functions (cross-entropy, focal)
│   ├── optimizer/                   # DL optimizer configs
│   ├── scheduler/                   # DL LR scheduler configs
│   └── path/                        # Output path templates
│
├── docs/                            # Reference documentation
│   ├── approach.md                  # Methodological description of the pipeline
│   ├── complexity_measures.md       # Per-cluster complexity measures (F, N, ND, T, G families)
│   ├── synthetic_dataset.md         # Synthetic dataset class specifications
│   └── pipeline.png                 # Pipeline overview diagram
│
├── src/                             # Pure library code (no cfg, no logger, no I/O)
│   ├── core/                        # Config loading, factory, logging, I/O, paths, utilities
│   ├── domain/                      # Domain logic
│   │   ├── analysis/                # Complexity measures, separability, metadata
│   │   ├── clustering/              # HDBSCAN clustering with grid search
│   │   ├── data/                    # Preprocessing (cleaning, scaling, encoding, sampling)
│   │   ├── plot/                    # Matplotlib helpers (charts, metrics, style)
│   │   ├── training/                # ML and DL training loops
│   │   └── projection.py            # t-SNE / UMAP projection
│   ├── engine/                      # Classifier engines
│   │   ├── dl/                      # PyTorch models, losses, datasets, Ignite engine
│   │   └── ml/                      # sklearn / XGBoost models and preprocessing
│   └── registries.py                # Factory registries (auto-imported)
│
└── resources/
    ├── raw_data/                    # Input CSV files (one subdir per dataset family)
    └── experiments/                 # Experiment outputs (per name/dataset/seed)
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The pinned dependencies are in [requirements.txt](requirements.txt); the unpinned source list is in [requirements.in](requirements.in). All commands below assume the venv is active.

---

## Input Data

Place raw CSV files under `resources/raw_data/`. Each dataset config in [configs/data/](configs/data/) declares its label column, numerical/categorical feature columns, split fractions, and filtering rules. The raw CSV path is resolved automatically via:

```
resources/raw_data/${data.dir}/${data.file_name}.csv
```

### Synthetic Dataset

A synthetic dataset is included for local testing without real data:

```bash
make generate              # default: ~102,500 rows
make generate ROWS=50000
```

This writes `resources/raw_data/synthetic/synthetic_test.csv` and can be used immediately with `data=synthetic_test`. The dataset covers 11 classes with engineered separation challenges (hard pairs, overlapping distributions, rare-class filtering, NaN/Inf injection). See [docs/synthetic_dataset.md](docs/synthetic_dataset.md) for the full class specification.

---

## Configuration

The project uses [Hydra](https://hydra.cc/) (Compose API) for configuration management. The root config is [configs/config.yaml](configs/config.yaml) with these defaults:

```yaml
defaults:
  - data: cic_2018_v2
  - classifier: tabular
  - loss: focal             # consumed only by DL classifiers
  - optimizer: adamw        # consumed only by DL classifiers
  - scheduler: one_cycle    # consumed only by DL classifiers
  - loops: default          # consumed only by DL classifiers
  - _self_
  - path: default
```

### Config Groups

| Group | Options | Description |
|---|---|---|
| `data` | `nb15_v2`, `bot_iot_v2`, `cic_2018_v2`, `cic_2018_f`, `ton_iot_v2`, `bank_marketing`, `covertype`, `letter_recognition`, `statlog_landsat_satellite`, `thyroid_disease`, `synthetic_test` | Dataset definition: columns, split ratios, filtering |
| `classifier` | DL: `tabular`, `numerical`, `categorical` &nbsp;·&nbsp; ML: `decision_tree`, `random_forest`, `hist_gradient_boosting`, `xgboost`, `knn`, `lda`, `logistic_regression`, `naive_bayes`, `svm_rbf` | Classifier kind (`ml` / `dl`), name, hyperparameters, optional grid-search grid |
| `clustering` | `hdbscan`, `kmeans`, `spectral`, `birch`, `kprototypes`, `ensemble` | Per-class clustering strategy and hyperparameter grids (`kprototypes` includes categorical features; GMM was evaluated and excluded for pathological fragmentation) |
| `loss` | `cross_entropy`, `focal` | DL loss functions |
| `optimizer` | `adamw` | DL optimizer settings |
| `scheduler` | `one_cycle` | DL learning-rate scheduler |
| `loops` | `default` | DL training loop (epochs, batch size, early stopping) |
| `path` | `default` | Output directory template |

### Key Global Parameters

| Parameter | Default | Description |
|---|---|---|
| `seed` | `42` | Random seed for reproducibility (splits, clustering, CV, weight init) |
| `name` | `exp` | Experiment name (component of the output path) |
| `device` | `cpu` | Device for DL training (`cpu`, `cuda`) |
| `stage` | `all` | DL only — which stages to run: `training`, `testing`, or `all` |
| `n_samples` | `null` | Optional training set subsampling cap |
| `balance` | `undersample` | Training-set class balancing at training time (`undersample` / `none`); persisted splits keep the original distribution |
| `distance` | `cosine` | Single metric (`euclidean` / `cosine`) interpolated by both `clustering.distance` and `complexity.distance` — coherence is structural |
| `kfold` | `true` | Classifier evaluation protocol: `true` = k-fold out-of-fold over train+test (one model per fold, saved under `models/fold_*`; metrics + per-cluster failure rates both come from the OOF predictions, `eval_mode: oof_kfold`); `false` = single held-out test split. Override to `false` on the largest datasets to skip the K× training cost |
| `kfold_splits` | `5` | Number of OOF folds when `kfold=true` (capped to the rarest class count) |
| `grid_search.enabled` | `false` | Enable sklearn `GridSearchCV` over `classifier.grid` (ML only) |
| `prepare.force` | `false` | Re-run preprocessing + clustering even if shared outputs exist |
| `complexity.k` | `30` | k for the shared k-NN graph |
| `complexity.top_k_clusters` | `10` | Top-K nearest adversarial clusters per cluster |
| `complexity.force` | `false` | Re-run complexity computation even if shared output exists |
| `clustering.min_cluster_floor` | `50` | Clusters below this size are absorbed into the class pseudo-cluster (all algorithms) |
| `clustering.target_cluster_size` | `25000` | Absolute cap on the split target: clusters above the effective target are split post-hoc with MiniBatchKMeans on the full cluster points |
| `clustering.target_size_frac` | `0.05` | Makes the split target relative to dataset size: effective target = clamp(2·`min_cluster_floor`, `total_n`·frac, `target_cluster_size`). Small datasets get a lower ceiling (more, finer clusters); large datasets stay at the absolute cap |
| `failure_classifier.labeling` | `binomial` | Failure label: `binomial` (error rate significantly above the classifier's global rate, level `alpha`) or `threshold` (`failure_rate > threshold`) |
| `failure_classifier.min_test_support` | `5` | Clusters with fewer test samples are excluded from the failure dataset |

The full set of nested parameters (clustering grid, failure-classifier nested CV grid, plot caps) lives in [configs/config.yaml](configs/config.yaml).

Any parameter can be overridden from the command line:

```bash
make classify DATA=bot_iot_v2 NAME=my_experiment SEED=123 CLASSIFIER=random_forest
# or via Hydra directly:
PYTHONPATH=. python pipelines/classify.py data=bot_iot_v2 name=my_experiment seed=123 classifier=random_forest
```

---

## Pipeline

The pipeline has four steps, driven by [pipelines/](pipelines/) entry points or the [Makefile](Makefile). Outputs are organised under:

```
resources/experiments/${name}/${data.file_name}_${seed}/
├── processed_data/                  # train.parquet, val.parquet, test.parquet (shared)
├── shared/                          # df_meta, clusters_meta, complexity & class_complexity metrics
└── ${classifier.name}/              # per-classifier subtree
    ├── configs/                     # resolved Hydra config snapshot
    ├── models/                      # checkpoints / serialized estimators
    ├── outputs/                     # JSON outputs (training, testing, analysis)
    ├── pickle/                      # binary side artifacts (confusion matrices, …)
    └── figures/                     # rendered PNGs
```

### Step 1 — Data Preparation

```bash
make prepare DATA=cic_2018_v2 NAME=my_exp
```

Loads the raw CSV, applies preprocessing (NaN removal, rare category filtering, log-scaling, robust scaling, optional top-N hash encoding), runs a stratified train/val/test split, encodes labels, and runs per-class clustering (HDBSCAN by default, selectable via `clustering=`) to assign every sample a globally unique cluster id. Noise points and clusters smaller than `clustering.min_cluster_floor` are merged into a per-class pseudo-cluster. The persisted splits keep the original class distribution — balancing happens at training time (Step 2, `balance`).

**Shared outputs (dataset-level):**

```
processed_data/                      # train.parquet, val.parquet, test.parquet
shared/
├── df_info.json                     # basic DataFrame stats
├── df_meta.json                     # label mapping, class weights, split sizes
└── clusters_meta.json               # cluster ids per split, centroids, noise ids
```

This step is idempotent: existing outputs are reused unless `prepare.force=true` (or `make ... FORCE=1`).

### Step 2 — Classifier Training

```bash
make classify DATA=cic_2018_v2 NAME=my_exp CLASSIFIER=random_forest
make ml-all   DATA=cic_2018_v2 NAME=my_exp     # every ML classifier in turn
make dl-all   DATA=cic_2018_v2 NAME=my_exp     # every DL classifier in turn
```

Trains and evaluates a single classifier (ML or DL, selected via the `classifier` config group), then writes per-class metrics and per-sample predictions used downstream by the failure classifier. The training split is balanced via random undersampling at load time (`balance=undersample`, the default); pass `balance=none` to train on the original distribution. By default (`kfold=true`) evaluation is **k-fold out-of-fold over train+test**: one model is trained per fold and saved under `models/fold_*`, and the reported metrics and per-cluster failure rates both come from the leakage-free OOF predictions (`eval_mode: oof_kfold`). Pass `kfold=false` for single held-out test-split evaluation (e.g. to skip the K× cost on the largest datasets). With `extend.generate=true` (or `make ... EXTEND=1`) the classifier is instead trained on the complexity-extended splits (`*_extended.parquet`, produced by Step 3a) and explained with SHAP — every artifact of this variant is written next to the base one with an `_extended` leaf-name suffix (`summary_extended.json`, `model_extended.joblib`, …). Its F1 is a transductive upper bound, reported as "F1 extended (transductive)"; see [docs/approach.md](docs/approach.md).

**Per-classifier outputs:**

```
${classifier.name}/
├── models/                          # checkpoint (DL) or serialized estimator (ML)
├── outputs/
│   ├── training/                    # training-set metrics (and predictions where applicable)
│   ├── testing/                     # test-set metrics
│   └── analysis/predictions/        # per-sample predictions + per-cluster failure rates
└── configs/config_composed.json     # resolved Hydra config
```

For DL classifiers, the `stage` parameter controls execution: `training` (train only), `testing` (test only, requires existing checkpoint), or `all` (train + test). For ML classifiers, `grid_search.enabled=true` switches to `GridSearchCV` over `classifier.grid`.

### Step 3a — Complexity Computation

```bash
make complexity DATA=cic_2018_v2 NAME=my_exp
```

Computes complexity measures at **two parallel partition levels** under a single neutral schema:

- **Cluster-level** (`complexity.json`): each cluster aggregated against its top-K nearest adversarial clusters.
- **Class-level** (`class_complexity.json`): each class aggregated against its top-K nearest adversarial classes.

Both levels share the same Gower-style mixed-distance k-NN backbone (governed by the top-level `distance` key, the same metric used by the clustering) and the same five families: F (feature), N (neighbourhood), ND (network density), T (dimensionality), G (geometry). The two outputs are independent and can be compared row-wise. See [docs/complexity_measures.md](docs/complexity_measures.md) for definitions.

**Shared outputs:**

```
shared/
├── complexity.json                  # per-cluster complexity vector
└── class_complexity.json            # per-class complexity vector (same schema)
```

This step is dataset-level (classifier-independent) and idempotent: existing outputs are reused unless `complexity.force=true`.

### Step 3b — Failure Classification

```bash
make failure-classify DATA=cic_2018_v2 NAME=my_exp CLASSIFIER=random_forest
```

Builds a per-cluster summary by joining (a) the cluster-level complexity vector, (b) the class-level complexity vector of the cluster's class (`cluster_*` / `class_*` feature prefixes), and (c) the classifier's per-cluster failure rate (Step 2). A Random Forest is then trained — with nested stratified cross-validation (5 outer × 5 inner folds) — to predict whether a cluster's failure rate exceeds `failure_classifier.threshold`. Reports out-of-fold metrics, ROC curves, and feature importances.

**Per-classifier outputs:**

```
${classifier.name}/outputs/analysis/
├── cluster_summary.json             # complexity + failure rate per cluster
└── failure_classifier_results.json  # nested-CV scores, feature importances, ROC data
```

### Step 4 — Plot Rendering

```bash
make render DATA=cic_2018_v2 NAME=my_exp CLASSIFIER=random_forest
```

Renders figures from the JSON / pickle artifacts produced by the previous steps (confusion matrices, per-class F1, complexity distributions, ROC curves, failure–complexity scatters). All figures land under `${classifier.name}/figures/`.

### Full Pipeline Shortcuts

```bash
make run DATA=cic_2018_v2 NAME=my_exp CLASSIFIER=tabular       # prepare → classify → failure-classify → render
make all NAME=my_exp                                            # every dataset, default classifier per dataset
```

The `all` target iterates over the `DATASET_CLASSIFIERS` list in the [Makefile](Makefile); override `SEED`, `DISTANCE` (`euclidean` / `cosine`), or add `FORCE=1` to recompute cached shared stages.

---

## Dashboard

```bash
make dashboard
```

Launches a Streamlit dashboard ([dashboard.py](dashboard.py)) for browsing experiment outputs across datasets, classifiers, and seeds. The Overview heatmap includes the optional "F1 extended (transductive)" metric, and the drill-down shows the explain panel (extended metrics + SHAP beeswarm figures) for runs produced with `EXTEND=1`.

---

## Library Packages (`src/`)

`src/` is a pure library — no `cfg`, no `logger`, no I/O. All side effects live in [pipelines/](pipelines/).

### `src/core`

- **config.py** — `load_config`, `save_config`, `to_container` (Hydra Compose API)
- **factory.py** — Generic `Factory[T]` with `@register` decorator; `discover_and_import_modules` for auto-registration
- **log.py** — `setup_logger`, `LogBundle`, `LogDispatcher`, and subscribers (`JSONSubscriber`, `PickleSubscriber`, `FilesystemFigureSubscriber`)
- **io.py** — Format-agnostic DataFrame I/O (`load_df`, `save_df`, `load_listed_dfs`) for CSV / Parquet / Pickle
- **paths.py** — `OutputPaths` dataclass holding the resolved output layout
- **utils.py** — JSON / pickle / joblib helpers, `timed` decorator, `flush_timing`, `skip_if_exists`

### `src/domain`

- **data/preprocessing.py** — Cleaning (`drop_nans`, `rare_category_filter`, `query_filter`), splitting / sampling (`ml_split`, `random_undersample_df`, `subsample_df`), transformers (`LogTransformer`, `TopNHashEncoder`), `build_preprocessor`, `encode_labels`
- **clustering/** — HDBSCAN clustering with grid search (`fit_hdbscan`, `grid_search`)
- **analysis/metadata.py** — `compute_df_metadata`, `compute_clusters_metadata`, `get_df_info`
- **analysis/separability.py** — Pairwise cluster separability scores
- **analysis/complexity/** — F / N / ND / T / G complexity families (`feature`, `neighborhood`, `network`, `dimensionality`, `clusters`)
- **projection.py** — t-SNE projection, stratified subsampling
- **plot/** — `Plot` dataclass, charts (`bar_plot`, `line_plot`, `scatter_plot`, `heatmap_plot`, `ridgeline_plot`, `strip_plot`, `violin_plot`), metric plots (`confusion_matrix_plot`, `roc_plot`), shared palette and `style.py`
- **training/ml.py** — `fit_classifier`, `grid_search_classifier`, `predict_with_proba`, `save_model`, `load_model` for ML pipelines
- **training/dl.py** — DL training loop built on PyTorch Ignite (`fit_classifier`, `predict_with_proba`, `save_model`, `load_model`)

### `src/engine`

- **dl/model/** — `NumericalClassifier`, `CategoricalClassifier`, `TabularClassifier` (registered via `DLClassifierFactory`)
- **dl/module/** — Encoder, decoder, MLP, embedding building blocks; checkpoint utilities
- **dl/loss/** — `CrossEntropyLoss`, `FocalLoss` (registered via `LossFactory`)
- **dl/data/** — `TabularDataset` (numerical + categorical), custom collate
- **dl/ignite_builder.py** — `EngineBuilder` fluent builder for Ignite engines (metrics, early stopping, checkpointing)
- **dl/ignite_metrics.py** — Per-class F1 / Precision / Recall wrappers for Ignite
- **ml/model/** — sklearn / XGBoost classifiers (`ensemble`, `linear`, `naive_bayes`, `neighbors`, `svm`, `tree`, `xgboost`), all registered via `MLClassifierFactory`
- **ml/preprocessing.py** — ML-specific column preprocessing helpers

### `src/registries.py`

Central re-export of `DLClassifierFactory`, `MLClassifierFactory`, and `LossFactory`. Import factories from here so the owning packages stay free to move.
