# Intrusion Forge

Tabular deep learning framework for network traffic classification. Supports supervised training, inference diagnostics, and post-hoc analysis on cybersecurity datasets (UNSW-NB15, BoT-IoT, CIC-IDS-2018, ToN-IoT).

The pipeline covers data preparation, model training, inference, and result analysis — all driven by a declarative Hydra configuration system.

---

## Project Structure

```
intrusion-forge/
├── prepare_data.py          # Step 1: preprocess raw CSV → parquet splits
├── classify.py              # Step 2: train and evaluate supervised classifier
├── analyze_data.py          # Step 3: post-hoc analysis of model predictions
├── generate_synthetic.py    # Generate synthetic test dataset
├── Makefile                 # Experiment runner (prepare / classify / analyze / generate / all)
│
├── configs/                 # Hydra configuration hierarchy
│   ├── config.yaml          # Root config with defaults
│   ├── data/                # Dataset definitions (columns, splits, filtering)
│   ├── experiment/          # Experiment presets (e.g. supervised)
│   ├── loops/               # Training/validation/test loop settings
│   ├── model/               # Model architectures
│   ├── loss/                # Loss functions (cross-entropy, focal)
│   ├── optimizer/           # Optimizer configs
│   ├── scheduler/           # LR scheduler configs
│   └── path/                # Output path templates
│
├── docs/                    # Reference documentation
│   ├── cluster_features.md  # Feature selection rationale for clustering
│   └── synthetic_dataset.md # Synthetic dataset class specifications
│
├── src/                     # Library code
│   ├── common/              # Config loading, logging, factory, utilities
│   ├── data/                # I/O, preprocessing, analysis functions
│   ├── ml/                  # Clustering, dimensionality reduction
│   ├── plot/                # Plotting helpers (confusion matrix, bar charts)
│   ├── torch/               # PyTorch models, losses, dataset, engine steps
│   └── ignite/              # Ignite engine builder, custom metrics
│
└── resources/
    ├── raw_data/dataset_v2/     # Input CSV files (one per dataset)
    ├── raw_data/synthetic/      # Generated synthetic CSV
    └── experiments/             # Experiment outputs (per name/dataset/seed/run)
```

---

## Setup

### Requirements

Python 3.12+.

### Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The pinned dependencies are in [requirements.txt](requirements.txt). The unpinned source list is in [requirements.in](requirements.in).

---

## Input Data

Place raw CSV files under `resources/raw_data/`. Each dataset config in `configs/data/` declares its label column, numerical/categorical feature columns, split fractions, and filtering rules. The raw CSV path is resolved automatically via:

```
resources/raw_data/${data.dir}/${data.file_name}.csv
```

### Synthetic Dataset

A synthetic dataset is included for local testing without real data:

```bash
make generate          # default: ~102,500 rows
make generate ROWS=50000
```

This writes `resources/raw_data/synthetic/synthetic_test.csv` and can be used immediately with `data=synthetic_test`. The dataset covers 11 classes with engineered separation challenges (hard pairs, overlapping distributions, rare-class filtering, NaN/Inf injection). See [docs/synthetic_dataset.md](docs/synthetic_dataset.md) for the full class specification.

---

## Configuration

The project uses [Hydra](https://hydra.cc/) (Compose API) for configuration management. The root config is [configs/config.yaml](configs/config.yaml) with these defaults:

```yaml
defaults:
  - data: cic_2018_v2
  - loops: default
  - model: tabular_classifier
  - loss: cross_entropy
  - optimizer: adamw
  - scheduler: one_cycle
  - experiment: supervised
  - path: default
```

### Config Groups

| Group | Options | Description |
|-------|---------|-------------|
| `data` | `nb15_v2`, `bot_iot_v2`, `cic_2018_v2`, `cic_2018_f`, `ton_iot_v2` | Dataset definition: columns, split ratios, filtering |
| `model` | `numerical_classifier`, `categorical_classifier`, `tabular_classifier` | Model architecture and hyperparameters |
| `loss` | `cross_entropy`, `focal` | Loss functions|
| `optimizer` | `adamw` | Optimizer settings |
| `scheduler` | `one_cycle` | Learning rate scheduler |
| `loops` | `default` | Epochs (30), batch size (512), early stopping (patience 5) |
| `experiment` | `supervised` | Experiment preset — overrides model, loss, and analysis settings |
| `path` | `default` | Output directory template |

### Key Global Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed` | 42 | Random seed for reproducibility |
| `name` | `exp` | Experiment name (part of output path) |
| `device` | `cpu` | Device for training (`cpu`, `cuda`) |
| `stage` | `all` | Which stages to run: `training`, `testing`, or `all` |
| `run_id` | 0 | Run index within an experiment (set by `experiment` config) |
| `n_samples` | `null` | Optional training set subsampling size |
| `failure_threshold` | 0.1 | Cluster failure rate threshold for analysis (set by `experiment` config) |

Any parameter can be overridden from the command line:

```bash
python prepare_data.py data=bot_iot_v2 name=my_experiment seed=123 device=cuda
```

---

## Pipeline

The pipeline consists of four sequential steps. Each script reads the Hydra config and produces outputs under:

```
resources/experiments/${name}/${data.file_name}_${seed}/${run_id}/
```

### Step 1 — Data Preparation

```bash
python prepare_data.py data=cic_2018_v2 experiment=supervised name=my_exp
```

Loads the raw CSV, applies preprocessing (NaN removal, rare category filtering, log-scaling, optional hash encoding), performs a stratified train/val/test split (80/10/10), balances classes via random undersampling, and encodes labels.

**Outputs:**

```
{run_id}/
├── processed_data/
│   ├── train.parquet
│   ├── val.parquet
│   └── test.parquet
├── logs/data/
│   ├── df_info.json          # Basic DataFrame stats
│   └── df_meta.json          # Label mapping, class weights, split sizes
└── configs/
    └── config_composed.json  # Resolved Hydra config for reproducibility
```

### Step 2 — Training & Testing

```bash
make classify DATA=cic_2018_v2 NAME=my_exp
# or: python classify.py experiment=supervised data=cic_2018_v2 name=my_exp
```

Trains a classifier using PyTorch Ignite (with early stopping and best-model checkpointing), then runs evaluation on the test set. Metrics: accuracy, precision, recall, F1 (macro, weighted, per-class).

The `stage` parameter controls execution: `training` (train only), `testing` (test only, requires existing checkpoint), or `all` (train + test).

**Outputs:**

```
{run_id}/
├── models/
│   └── model_{epoch}_loss={loss}.pt     # Best model checkpoint
├── logs/testing/
│   └── summary.json                     # All scalar metrics
└── logs/analysis/
    └── predictions/
        └── test.json                    # Per-class failure analysis
tb/
├── training/    # TensorBoard events (loss, grad norm, epoch duration)
├── validation/  # TensorBoard events (loss)
└── testing/     # TensorBoard events (confusion matrix, per-class F1, projections)
```

### Step 3 — Analysis

```bash
make analyze DATA=cic_2018_v2 NAME=my_exp
# or: python analyze_data.py experiment=supervised data=cic_2018_v2 name=my_exp
```

Performs post-hoc analysis on the model outputs. The specific analyses depend on the experiment configuration (see [Experiments](#experiments)).

**Outputs:**

```
{run_id}/logs/
├── data/              # Data-level analysis results
└── analysis/          # Prediction-level analysis results
```

---

## Running All Datasets

Use the `all` Makefile target to run the full pipeline on every dataset:

```bash
make all NAME=my_experiment
```

This runs `prepare → classify → analyze` for each dataset in `DATASETS` (`nb15_v2`, `bot_iot_v2`, `cic_2018_v2`, `ton_iot_v2`). Override `SEED` and `EXPERIMENT` as needed:

```bash
make all NAME=my_experiment SEED=123 EXPERIMENT=supervised
```

Individual steps can also be run with `make prepare`, `make classify`, `make analyze`.

---

## Experiments

The framework supports pluggable experiment configurations. Each experiment preset (in `configs/experiment/`) can override model, loss, and pipeline-specific settings.

### Cluster Separability

The current experiment (`supervised`) extends the base pipeline with HDBSCAN clustering and cluster-level analysis:

- **Data preparation** optionally runs HDBSCAN clustering on processed features, assigning each sample to a cluster and saving cluster metadata (distributions, centroids).
- **Inference** computes per-cluster failure rates alongside per-class metrics.
- **Analysis** (`analyze_data.py`) computes pairwise cluster separability (inter/intra-cluster distance ratios), builds a per-cluster summary combining separability scores with inference failure rates, and runs a Random Forest grid search to identify which cluster properties correlate with model errors.

Additional outputs produced by this experiment:

```
{run_id}/logs/
├── data/
│   └── clusters_meta.json             # Cluster distributions and centroids
└── analysis/
    ├── cluster_summary.json           # Aggregated per-cluster summary
    ├── classifier_results.json        # RF grid search results, feature importances
    └── separability/
        ├── cluster_train.json         # Pairwise separability (train)
        └── cluster_test.json          # Pairwise separability (test)
```

---

## Output Directory Layout

All outputs are organized under `resources/experiments/` following this structure:

```
resources/experiments/{name}/{data.file_name}_{seed}/
├── {run_id}/
│   ├── processed_data/       # train.parquet, val.parquet, test.parquet
│   ├── models/               # Best model checkpoint (.pt)
│   ├── logs/
│   │   ├── data/             # df_info, df_meta, experiment-specific logs
│   │   ├── testing/          # summary.json (test metrics)
│   │   └── analysis/         # predictions/, cluster_summary, separability, RF results
│   └── configs/              # Resolved config snapshot
└── tb/                       # TensorBoard logs (shared across runs)
    ├── training/
    ├── validation/
    ├── testing/
    └── analysis/
```

To view TensorBoard logs:

```bash
make tensorboard NAME=my_experiment DATA=cic_2018_v2
```

---

## Library Packages (`src/`)

### `src/common`

- **config.py** — Load Hydra config via Compose API (`load_config`)
- **factory.py** — Generic Factory with `@register` decorator (used by models and losses)
- **log.py** — Logger setup (`setup_logger`), `LogBundle`, `LogDispatcher`, and subscribers (`TensorBoardSubscriber`, `JSONSubscriber`, `PickleSubscriber`)
- **utils.py** — JSON/pickle save/load helpers, `timed` decorator, `flush_timing`

### `src/data`

- **io.py** — Format-agnostic DataFrame I/O (CSV, Parquet, Pickle)
- **preprocessing.py** — Cleaning (NaN removal, rare category filtering), sampling (stratified split, undersampling), transformers (`LogTransformer`, `TopNHashEncoder`), sklearn `ColumnTransformer` builder
- **analyze.py** — DataFrame metadata, cluster statistics, pairwise separability, summary aggregation

### `src/ml`

- **clustering.py** — HDBSCAN clustering with grid search and silhouette scoring
- **projection.py** — t-SNE projection, stratified subsampling

### `src/plot`

- **base.py** — `Plot` dataclass wrapping a rendered PNG buffer
- **style.py** — Shared color palettes and Matplotlib style helpers
- **array.py** — Confusion matrix, scatter/strip/violin/bar plots from arrays
- **dict.py** — Bar charts and table plots from dictionaries

### `src/torch`

- **builders.py** — Factory functions for dataset, dataloader, model, loss, optimizer, scheduler
- **engine.py** — Training, evaluation, and test step functions
- **infer.py** — DataFrame-to-tensor conversion, model inference, prediction extraction
- **data/** — `TabularDataset` (numerical + categorical features), custom collate
- **model/** — `NumericalClassifier`, `CategoricalClassifier`, `TabularClassifier` (encoder + head architecture, registered via `ModelFactory`)
- **loss/** — `CrossEntropyLoss`, `FocalLoss` with label smoothing and class weighting (registered via `LossFactory`)
- **module/** — Encoder, decoder, MLP, embedding, and checkpoint utilities

### `src/ignite`

- **builders.py** — `EngineBuilder`: fluent builder for Ignite engines (metrics, early stopping, checkpointing, TensorBoard)
- **metrics.py** — Per-class F1, Precision, Recall wrappers for Ignite
