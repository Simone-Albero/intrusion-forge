# Intrusion Forge

Tabular deep learning framework for network traffic classification. Supports supervised training, inference diagnostics, and post-hoc analysis on cybersecurity datasets (UNSW-NB15, BoT-IoT, CIC-IDS-2018, ToN-IoT).

The pipeline covers data preparation, model training, inference, and result analysis — all driven by a declarative Hydra configuration system.

---

## Project Structure

```
intrusion-forge/
├── prepare_data.py          # Step 1: preprocess raw CSV → parquet splits
├── sup_classify.py          # Step 2: train and test a supervised classifier
├── run_inference.py         # Step 3: full inference with visualizations and error analysis
├── analyze_data.py          # Step 4: post-hoc analysis of model predictions
├── run_exp.sh               # Batch runner for multiple datasets
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
├── src/                     # Library code
│   ├── common/              # Config loading, logging, factory, utilities
│   ├── data/                # I/O, preprocessing, analysis functions
│   ├── ml/                  # Clustering, dimensionality reduction
│   ├── plot/                # Plotting helpers (confusion matrix, bar charts)
│   ├── torch/               # PyTorch models, losses, dataset, engine steps
│   └── ignite/              # Ignite engine builder, custom metrics
│
└── resources/
    ├── raw_data/dataset_v2/ # Input CSV files (one per dataset)
    └── experiments/         # Experiment outputs (per name/dataset/seed/run)
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

---

## Configuration

The project uses [Hydra](https://hydra.cc/) (Compose API) for configuration management. The root config is [configs/config.yaml](configs/config.yaml) with these defaults:

```yaml
defaults:
  - data: cic_2018_v2
  - loops: default
  - model: tabular_classifier
  - loss: classification
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
| `run_id` | 0 | Run index within an experiment |

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
python sup_classify.py data=cic_2018_v2 experiment=supervised name=my_exp
```

Trains a classifier using PyTorch Ignite (with early stopping and best-model checkpointing), then runs evaluation on the test set. Metrics: accuracy, precision, recall, F1 (macro, weighted, per-class).

The `stage` parameter controls execution: `training` (train only), `testing` (test only, requires existing checkpoint), or `all` (train + test).

**Outputs:**

```
{run_id}/
├── models/
│   └── model_{epoch}_loss={loss}.pt     # Best model checkpoint
└── logs/test/
    └── summary.json                     # All scalar metrics
tb/
├── training/    # TensorBoard events (loss, grad norm)
├── validation/  # TensorBoard events (loss, metrics)
└── testing/     # TensorBoard events (confusion matrix, per-class F1)
```

### Step 3 — Inference & Diagnostics

```bash
python run_inference.py data=cic_2018_v2 experiment=supervised name=my_exp
```

Runs the best model on all splits (train, val, test). For each split: extracts predictions, confidences, and latent embeddings; generates confusion matrices (raw and normalized); computes per-class failure rates; produces t-SNE projections of raw features and learned representations.

**Outputs:**

```
{run_id}/
├── pickle/analysis/
│   └── confusion_matrices/
│       ├── train.pkl, val.pkl, test.pkl
├── logs/analysis/
│   └── predictions/
│       ├── train.json, val.json, test.json   # Per-class failure analysis
tb/
└── analysis/
    ├── train/, val/, test/                    # TensorBoard: t-SNE, confusion matrices
```

### Step 4 — Analysis

```bash
python analyze_data.py data=cic_2018_v2 experiment=supervised name=my_exp
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

The [run_exp.sh](run_exp.sh) script iterates over all four datasets and runs the full pipeline:

```bash
chmod +x run_exp.sh
./run_exp.sh
```

Edit the script to uncomment the stages you want to run and set the experiment `NAME`:

```bash
DATASETS=("nb15_v2" "bot_iot_v2" "cic_2018_v2" "ton_iot_v2")
NAME="my_experiment"

for dataset in "${DATASETS[@]}"; do
    python3 prepare_data.py experiment=supervised data="$dataset" name="$NAME"
    python3 sup_classify.py experiment=supervised data="$dataset" name="$NAME"
    python3 run_inference.py experiment=supervised data="$dataset" name="$NAME"
    python3 analyze_data.py experiment=supervised data="$dataset" name="$NAME"
done
```

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
    ├── correlation_results.json       # RF grid search results, feature importances
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
│   │   ├── test/             # summary.json (test metrics)
│   │   └── analysis/         # predictions/, experiment-specific analysis
│   ├── pickle/               # Serialized objects (confusion matrices)
│   └── configs/              # Resolved config snapshot
└── tb/                       # TensorBoard logs (shared across runs)
    ├── training/
    ├── validation/
    ├── testing/
    └── analysis/
```

To view TensorBoard logs:

```bash
tensorboard --logdir resources/experiments/{name}/{data.file_name}_{seed}/tb
```

---

## Library Packages (`src/`)

### `src/common`

- **config.py** — Load Hydra config via Compose API (`load_config`)
- **factory.py** — Generic Factory with `@register` decorator (used by models and losses)
- **logging.py** — Logger setup (`setup_logger`)
- **utils.py** — JSON/pickle save/load helpers

### `src/data`

- **io.py** — Format-agnostic DataFrame I/O (CSV, Parquet, Pickle)
- **preprocessing.py** — Cleaning (NaN removal, rare category filtering), sampling (stratified split, undersampling), transformers (`LogTransformer`, `TopNHashEncoder`), sklearn `ColumnTransformer` builder
- **analyze.py** — DataFrame metadata, cluster statistics, pairwise separability, summary aggregation

### `src/ml`

- **clustering.py** — HDBSCAN clustering with grid search and silhouette scoring
- **projection.py** — t-SNE projection, stratified subsampling

### `src/plot`

- **array.py** — Confusion matrix and sample scatter plots
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
