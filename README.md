# Intrusion Forge

A framework for finding out **where** a trained classifier fails, and **why**, without needing labelled test data for the regions in question.

It divides a dataset into regions, describes each region by how it sits relative to the classes competing with it, and learns how those descriptions relate to the mistakes a classifier actually made. The result is an error-rate estimate for any region, including regions the framework has never seen, together with a ranking of the data properties that drive the risk.

The estimate belongs to the model you point it at. It is fitted on that model's own errors, so it describes how *that* classifier copes with the shape of your data rather than how difficult the data is in the abstract. This makes it useful for identifying unreliable regions before you have the labels to prove they are unreliable, for deciding where to gather more data or review labels, and for monitoring a model after deployment.

Everything is tabular and everything is driven by configuration: a dozen classifiers (scikit-learn, XGBoost, PyTorch), four ways of dividing the data into regions, configurations for the public network-security datasets, and a synthetic dataset that exercises the whole pipeline in about nine minutes.

## Quickstart — the synthetic demo

`resources/` is not tracked by git, so a fresh clone carries no data. The synthetic generator is the only dataset that works out of the box, and the quickest way to see every stage run.

### 1. Install

Python 3.12 recommended.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run everything from the repository root — Hydra resolves paths relative to it.

### 2. Generate the dataset

```bash
make generate
```

Writes `resources/raw_data/synthetic/synthetic_test.csv`: 69,500 rows, 10 classes, 20 numerical and 2 categorical features. `ROWS=30000` produces a smaller one.

The dataset has a *known* difficulty gradient built into it ([described below](#inside-the-synthetic-dataset)), so you can tell whether the pipeline has recovered something real.

### 3. Run the pipeline

```bash
make run DATA=synthetic_test NAME=demo CLASSIFIER=random_forest CLUSTERING=kmeans
```

One command, five stages:

| Stage | What it does | Time |
|---|---|---|
| prepare | preprocess, split, divide each class into regions (1021 of them) | 20 s |
| complexity | describe every region | 25 s |
| classify | train and evaluate the Random Forest | 20 s |
| failure-regress | fit the region → error-rate estimator | 5 min |
| render | 26 figures | 5 s |

The result appears at the end of `failure-regress`:

```
Failure regressor results — Spearman: 0.9353, R²: 0.9189, MAE: 0.0343, MSE: 0.0030
```

**Spearman ρ ≈ 0.94.** Across 1019 regions the estimated and observed error rates rank almost identically, measured on regions held back from the fitting. Expect a little drift in the third decimal between runs.

`instance_baselines.json` is a check on that signal rather than the purpose of the framework: it compares the regressor's per-sample ranking against confidence-based baselines (MCP, ATC, and rank-averaged combinations of each with the regressor) using *oracle benefit recovered* — the share of a perfect oracle's accuracy gain that abstaining on the riskiest samples actually captures. The regressor recovers about 61% here, ahead of MCP and ATC alone (~59% each). Were the estimate merely tracking some general notion of difficult data rather than this model's own errors, it would not consistently beat scores derived from the model's own confidence.

### 4. Read the results

Everything is written to `resources/experiments/demo/synthetic_test_42/`:

```
processed_data/                       # train / val / test parquet
shared/                               # dataset-level, the same for every classifier
├── complexity.json                   #   descriptors per region
├── class_complexity.json             #   descriptors per class
└── metadata/                         #   label map, split sizes, region centroids
random_forest/
├── outputs/testing/summary.json      # accuracy, macro F1, per-class metrics
├── outputs/analysis/
│   ├── cluster_summary.json          #   descriptors + observed error rate, per region
│   ├── failure_regressor_results.json #  ρ, R², MAE, MSE, importances
│   ├── instance_baselines.json       #   ρ and oracle-benefit recovered, regressor vs confidence baselines
│   └── predictions/                  #   per-sample predictions
├── models/fold_0 … fold_4/           # one model per fold (~500 MB)
└── figures/                          # 26 PDFs
```

The fold models account for ~500 MB of the ~540 MB the run occupies; delete `models/` once you have the metrics.

### 5. Optional — browse in the dashboard

```bash
make dashboard
```

A Streamlit application over `resources/experiments/`: choose a dataset, classifier and seed, compare runs in a heatmap, then examine one in detail.

### Variations worth trying

```bash
# a deep-learning classifier (much slower: 12-15 min in the classify stage alone)
make run DATA=synthetic_test NAME=demo_dl CLASSIFIER=tabular CLUSTERING=kmeans

# cosine instead of euclidean
make run DATA=synthetic_test NAME=demo_cos CLASSIFIER=random_forest CLUSTERING=kmeans DISTANCE=cosine

# a different clustering algorithm
make run DATA=synthetic_test NAME=demo_birch CLASSIFIER=random_forest CLUSTERING=birch
```

`prepare` and `complexity` are cached per `(NAME, dataset, seed)`, so changing classifier reuses them. `FORCE=1` recomputes them.

How the data is divided into regions matters more than which classifier you use. `kmeans` and `birch` both find about a thousand regions here; `hdbscan` finds thirty and consigns a quarter of the points to leftover buckets, because this dataset's difficulty gradient is continuous rather than broken by real gaps. Count-based algorithms suit data of this kind, density-based ones suit data with genuine separation.

### Inside the synthetic dataset

Each class consists of a clean core together with a *difficulty ladder* running towards every class it is meant to overlap with (`_OVERLAPS` in [generate_synthetic.py](generate_synthetic.py)). Every rung sits a fixed distance from the boundary between the two classes, and both sides face one another symmetrically, so the best error rate any classifier could reach on a rung is known in advance:

| Sub-group | Distance from boundary | Best possible error | Rows |
|---|---|---|---|
| `canonical` | — | ≈ 0 % | 23,966 |
| `clear` | 1.80 σ | ≈ 4 % | 19,316 |
| `evasive` | 0.90 σ | ≈ 18 % | 16,560 |
| `mimicry` | 0.25 σ | ≈ 40 % | 9,658 |

The two categorical features fade out along the same ladder, so they cease to help exactly where the numbers begin to overlap. No class sits at the origin either: the benign class has features of its own, just as every attack does, which is what allows `DISTANCE=cosine` to see the same structure as `DISTANCE=euclidean`.

Recovering that ladder is the task. Clustering divides the overlap corridors into regions, the descriptors rank them, and the classifier fails progressively more often on the harder ones. Each row's intended difficulty is recorded in a `true_subgroup` column so that you can check afterwards; the pipeline never reads it.

Accuracy settles at about 0.85 by design. The hard rungs are genuinely hard, and that spread is what makes a per-region correlation meaningful.

## Running on your own data

The configurations in [configs/data/](configs/data/) cover UNSW-NB15, BoT-IoT, CIC-IDS-2018, ToN-IoT and five general benchmarks, but the CSV files are not in the repository: supply them under `resources/raw_data/<dir>/`, matching each configuration's `dir` and `file_name`.

To add a dataset of your own:

1. Place the CSV at `resources/raw_data/<dir>/<file_name>.csv`.
2. Copy a configuration from [configs/data/](configs/data/) and edit it: `label_col`, `num_cols`, `cat_cols`, `benign_tag`, split fractions, filtering.
3. Add it to `DATASETS` in the [Makefile](Makefile).
4. Run `make run DATA=<name> NAME=my_exp CLASSIFIER=random_forest CLUSTERING=kmeans`.

On datasets of millions of rows, out-of-fold evaluation is disabled automatically (`LARGE_DATASETS` in the Makefile), since one model per fold would otherwise take hours.

## How it works

Three steps, together with the classifier whose errors they are fitted on. The method assumes nothing about the classifier, the clustering algorithm or the number of classes; it requires only tabular features.

1. **Identify regions.** Divide each class into compact sub-regions, using the training data alone. The number per class is chosen by grid search, biased towards finer partitions so that the final estimator has enough to learn from. Points the algorithm treats as noise are gathered into a per-class leftover bucket and discarded downstream.
2. **Describe each region.** Score how separable it is from the rival regions nearest to it, and do the same at class level. These descriptors are drawn from the data alone; nothing about the classifier enters here.
3. **Learn the mapping.** A Random Forest regresses each region's observed error rate on its descriptors under nested cross-validation, reporting the correlation together with the descriptors that mattered. The choice of a tree-based estimator is deliberate: every risk estimate arrives with the properties that produced it.

The classifier is trained separately and contributes exactly one thing to step 3, namely the error rate it achieved in each region. That is what ties the mapping to the model rather than to the dataset.

Two points worth knowing:

- Regions are built from the training split alone. Held-out samples are then routed to the nearest region **without reference to their label**, exactly as a sample would be routed at inference time. This is what keeps the correlation a genuine prediction rather than a restatement of labels already known.
- The analysis works chiefly per region rather than per class, since one class usually occupies several separate areas of the space. Class-level figures are kept as a coarse reference.

### What the descriptors measure

Five families, all built on a single shared nearest-neighbour graph over mixed numerical and categorical features. Each compares a region with the ten rival regions nearest to it, reported as min / mean / max across them.

| Family | Keys | What it captures |
|---|---|---|
| **F** — feature | `f1`–`f4` | how well individual features separate the region from its rivals |
| **N** — neighbourhood | `n1`–`n4` | how many of a region's neighbours belong to another class |
| **ND** — network | `network_density`, `cls_coef`, `hub` | how many neighbour links cross into a rival region |
| **T** — dimensionality | `t2`–`t4` | how many features there are relative to samples, and how many of them matter |
| **G** — geometry | `max_dispersion`, `p95_dispersion`, `dist_to_nearest_centroid`, `p5_silhouette`, `frac_at_risk` | how widely the region is spread, and how close the nearest rival lies |

In the demo the estimator relies most on `cluster_p5_silhouette`, `cluster_f3_max` and `cluster_network_density_mean`.

## Pipeline reference

Each stage is a script under [pipelines/](pipelines/), wrapped by the [Makefile](Makefile). Every target takes `DATA`, `NAME`, `SEED`, `CLASSIFIER`, `CLUSTERING`, `DISTANCE`.

| Target | Script | Scope |
|---|---|---|
| `make prepare` | `prepare_data.py` | per dataset, cached |
| `make complexity` | `compute_complexity.py` | per dataset, cached |
| `make classify` | `classify.py` | per classifier |
| `make failure-regress` | `fit_failure_regressor.py` | per classifier |
| `make render` | `render_plots.py` | per classifier |
| `make run` | all of the above | omitted variables iterate, those passed stay fixed |
| `make comparisons` | `comparisons.py` | reduces a whole sweep to cross-run figures and tables |
| `make help` | — | every target, with defaults |

**prepare** produces the splits and the regions. It removes NaNs, filters rare categories, log-scales and robust-scales the numerical columns, hashes high-cardinality categorical ones, splits the data stratified, then clusters each class and gives every sample a region id. The saved splits retain the original class balance; balancing takes place at training time.

**classify** trains one classifier and records its per-class metrics and per-sample predictions. Evaluation is by default out-of-fold across train and test together: one model per fold, with both the metrics and the per-region error rates taken from predictions no model saw while training.

**failure-regress** assembles the table — a region's descriptors, its class's descriptors and its observed error rate — and fits the estimator over five outer and five inner folds. It also compares the estimate against confidence-based baselines (MCP, ATC, and rank-averaged combinations of the two with the regressor).

**render** turns the saved JSON and pickle artefacts into figures. `figure_format` selects `pdf`, the default, or `png`.

### Sweeps

```bash
make run NAME=x                             # every dataset × classifier × clustering algorithm
make run NAME=x DATA=covertype              # one dataset, every compatible classifier
make run NAME=x CLASSIFIER=random_forest    # every dataset, one classifier
make comparisons FIGURES_DIR=paper/figures
```

Omit `CLUSTERING` and the sweep runs every algorithm into a separate `NAME_<algo>` tree. `comparisons` then reduces such a tree to the cross-run figures (ρ per configuration, ρ against region count, family importance, per-classifier and per-dataset baseline comparisons) and the matching JSON tables.

## Configuration

The root configuration is [configs/config.yaml](configs/config.yaml), where every parameter is documented inline. Any key may be overridden on the command line:

```bash
make classify DATA=bot_iot_v2 NAME=my_exp SEED=123 CLASSIFIER=random_forest
PYTHONPATH=. python pipelines/classify.py data=bot_iot_v2 name=my_exp seed=123 classifier=random_forest
```

| Group | Options |
|---|---|
| `data` | network traffic: `nb15_v2`, `bot_iot_v2`, `cic_2018_v2`, `ton_iot_v2` · benchmarks: `bank_marketing`, `covertype`, `letter_recognition`, `statlog_landsat_satellite`, `thyroid_disease` · `synthetic_test`, the only one needing no external CSV |
| `classifier` | deep: `tabular` (adapts to the dataset's numerical/categorical feature counts) · classical: `decision_tree`, `random_forest`, `hist_gradient_boosting`, `xgboost`, `knn`, `lda`, `logistic_regression`, `naive_bayes`, `linear_svc` |
| `clustering` | `kmeans`, `hdbscan`, `birch`, `spectral` |
| `complexity` | `default` — descriptor graph parameters (`k`, cluster sample caps) |
| `failure_regressor` | `random_forest` — nested-CV folds and hyperparameter grid |
| `grid_search` | `default` — scoring, CV folds and sample cap for classifier tuning |
| `loss` / `optimizer` / `scheduler` / `loops` | deep learning only: `cross_entropy` \| `focal` / `adamw` / `one_cycle` / `default` |
| `path` | `default` |

Results are written to:

```
resources/experiments/${name}/${data.file_name}_${seed}/
├── processed_data/         # train / val / test parquet, shared
├── shared/                 # descriptors and metadata, shared
└── ${classifier.name}/
    ├── configs/            # resolved configuration snapshot
    ├── models/             # checkpoints or serialised estimators
    ├── outputs/            # training, testing and analysis JSON
    ├── pickle/             # binary side artefacts
    └── figures/            # rendered figures
```

## Repository layout

```
intrusion-forge/
├── pipelines/                    # entry points — own the config, I/O, logging and paths
│   ├── prepare_data.py           #   preprocess + divide into regions
│   ├── classify.py               #   train + evaluate one classifier (splits/training/evaluation in sibling modules)
│   ├── compute_complexity.py     #   region and class descriptors
│   ├── fit_failure_regressor.py  #   descriptors → error rate
│   ├── render_plots.py           #   figures
│   └── comparisons.py            #   cross-run aggregation
├── generate_synthetic.py         # synthetic dataset generator
├── dashboard.py                  # Streamlit experiment browser
├── Makefile                      # experiment runner
├── configs/                      # Hydra hierarchy
├── src/                          # pure library — no config, no I/O, no path building
│   ├── core/                     # config, Factory, LogDispatcher, DataFrame I/O, OutputPaths
│   ├── domain/
│   │   ├── data/                 # cleaning, splitting, scaling, encoding
│   │   ├── clustering/           # the four algorithms + grid search
│   │   ├── analysis/complexity/  # the F / N / ND / T / G families
│   │   ├── analysis/             # metadata, confidence & risk-coverage scores, the failure regressor
│   │   ├── training/             # ml.py (sklearn / XGBoost), dl.py (Ignite loop)
│   │   ├── plot/                 # Plot payload, chart primitives, analysis/comparison composers, metrics, palette
│   │   └── projection.py         # t-SNE
│   └── engine/
│       ├── dl/                   # models, modules, losses, dataset, EngineBuilder
│       └── ml/                   # sklearn / XGBoost wrappers, column preprocessing
└── resources/                    # not tracked by git: raw_data/ in, experiments/ out
```

Three conventions to know before editing:

- `src/` is input to output. Configuration loading, file I/O and path building belong to `pipelines/` and `src/core` alone.
- Every write, whether JSON, pickle or figure, goes through `LogDispatcher.publish(LogBundle)` and a subscriber, never a direct `save_to_*`.
- Classifiers and losses register themselves with a factory (`@DLClassifierFactory.register()`, `@LossFactory.register()`, or `MLClassifierFactory.register("name")(SklearnClass)`) and are discovered automatically at import.
