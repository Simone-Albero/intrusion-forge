# Intrusion Forge

A framework for finding out **where** a trained classifier fails, and **why**, without needing labelled test data for the regions in question.

It divides a dataset into regions, describes each region by how it sits relative to the classes competing with it, and learns how those descriptions relate to the mistakes a classifier actually made. The result is an error-rate estimate for any region, including regions the framework has never seen, together with a ranking of the data properties that drive the risk.

The estimate belongs to the model you point it at. It is fitted on that model's own errors, so it describes how *that* classifier copes with the shape of your data rather than how difficult the data is in the abstract. This makes it useful for identifying unreliable regions before you have the labels to prove they are unreliable, for deciding where to gather more data or review labels, and for monitoring a model after deployment.

Everything is tabular and everything is driven by configuration: ten classifiers (scikit-learn, XGBoost, PyTorch), four ways of dividing the data into regions, configurations for the public network-security datasets, and a synthetic dataset that exercises the whole pipeline in about five minutes.

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

One command, seven stages:

| Stage | What it does | Time |
|---|---|---|
| split | clean, split and preprocess the raw CSV | 2 s |
| graph | sample the training split and build its nearest-neighbour graph | 5 s |
| regions | divide each class into regions (707 of them here), place every sample in one | 4 s |
| complexity | describe every region and every class | ~15 s |
| classify | train the Random Forest on the training split, predict every split | ~3 min |
| regress | fit the region → error-rate estimators | ~105 s |
| render | 27 figures, and 3 of the classifier | 5 s |

Nearly all of `classify` goes on training the Random Forest: its hyperparameter search
cross-validates every candidate of its grid on the training split before refitting the winner.

`regress` fits four estimators of each region's error rate and logs one line for each, and `render` sets their ρ and MSE side by side in `correlation/regressor_comparison`. The first line is the Random Forest's, the primary estimate, the one the numbers below refer to:

```
Failure regressor random_forest — Spearman: 0.8594, R²: 0.8037, MAE: 0.0376, MSE: 0.0032
```

**Spearman ρ ≈ 0.86.** The estimated and observed error rates put the 707 regions in much the same order, measured on regions held back from the fitting. Expect a little drift in the third decimal between runs.

`regress/baselines.json` is a check on that signal rather than the purpose of the framework: it compares the regressor's per-sample ranking against confidence-based baselines (MCP, ATC, and rank-averaged combinations of each with the regressor) using *oracle benefit recovered* — the share of a perfect oracle's accuracy gain that abstaining on the riskiest samples actually captures. ATC's confidence threshold is chosen on the validation split, so the test samples it is scored on never pick their own cut. The regressor recovers about 22% here, a little more than MCP and ATC averaged over each region (about 20% and 21%): at the same granularity, one score per region, an estimate built from the data's geometry ranks this model's errors slightly better than the model's own confidence does. Rank-averaged with each sample's own MCP it recovers the most of these five, about 35%, as the only variant that tells apart the samples within a region. Two more baselines need no estimator at all: each region's error rate on the training samples the classifier was fitted on, and on the validation samples routed to it. They ask whether the estimate adds anything to the rate a region has already shown on other samples. MCP, ATC and the training rate need not lie on the scale of the test error rate, so each also appears calibrated onto it by a fit on the validation regions: the calibrated variant keeps the raw one's ranking of regions and samples, and differs only in how far it misses the observed rate. `baselines.json` also breaks down by region size how far the regressor, the two rates, MCP and ATC averaged over each region and the three calibrated variants miss the observed rate, and how well each ranks the regions of a similar size, which `render` draws in `baselines/mse_by_region_size`, `baselines/bias_by_region_size` and `baselines/spearman_by_region_size`.

### 4. Read the results

Everything is written to `resources/experiments/demo/synthetic_test_42/`:

```
split/                                # dataset-level, the same for every classifier
├── {train,val,test}.parquet          #   the preprocessed splits, integer `label`
├── preprocessor.joblib               #   the fitted preprocessing
└── meta.json                         #   columns, classes and their counts per split
graph/
├── graph.npz                         #   which training rows are the graph's nodes, their k-NN graph and spanning tree
└── space.json                        #   the codes of each categorical column that keep a one-hot slot, the mismatch cost
regions/
├── centroids.parquet                 #   the centre of every region
├── assignments.parquet               #   the region of every sample of every split, and its nearest one
└── report.json                       #   per-class clustering report and the candidates swept
complexity/
├── regions.parquet                   #   descriptors per region
└── classes.parquet                   #   descriptors per class
random_forest/
├── classify/
│   ├── model/model.joblib            #   the trained Random Forest
│   ├── predictions.parquet           #   predicted class and class probabilities of every sample, and which ones the model was fitted on
│   ├── metrics.json                  #   accuracy, macro F1, per-class metrics
│   └── training.json                 #   the hyperparameter search
├── regress/
│   ├── regions.parquet               #   per region: observed and estimated error rate, and the rate observed on the fit and validation samples
│   ├── results.json                  #   the primary estimator's ρ, R², MAE, MSE, importances, best parameters per fold; every estimator's scores
│   └── baselines.json                #   ρ, MSE and oracle benefit recovered, regressor vs baselines, the calibration fits, and error by region size
└── render/figures/                   #   27 PDFs, and the classifier's 3 under classification/
```

Every stage folder also holds `record.json`, which the caches check, and `timing.json`. No stage reads `classify/model/` back; delete it once you have the predictions.

### Variations worth trying

```bash
# a deep-learning classifier
make run DATA=synthetic_test NAME=demo_dl CLASSIFIER=mlp CLUSTERING=kmeans

# cosine instead of euclidean
make run DATA=synthetic_test NAME=demo_cos CLASSIFIER=random_forest CLUSTERING=kmeans DISTANCE=cosine

# a different clustering algorithm
make run DATA=synthetic_test NAME=demo_birch CLASSIFIER=random_forest CLUSTERING=birch
```

The MLP of the first variation spends about 15 s in `classify` and reaches ρ ≈ 0.88 on the same 707 regions, at an accuracy of 0.83.

`split`, `graph`, `regions` and `complexity` are cached per `(NAME, dataset, seed)`, so changing classifier reuses them. Each stage owns one folder, empties it when it recomputes, and writes `record.json` last — the settings it depends on, the ids of the stages it read and an id of its own — so a run interrupted before the end is recomputed rather than trusted. A cached stage skips when its record matches what a rerun would write; otherwise it recomputes, naming what changed. A record holds settings and stage ids, neither the code nor the data: once `split` has run, it never looks at the raw CSV again, so after replacing the CSV or changing the code, force the stage affected (`make split FORCE=1` for a new CSV). Its id is new on every write, so the next `make run` recomputes every stage downstream as well. Without the CSV, a cached `split` skips, and one that must recompute stops with an error before touching its folder. No stage runs on stale inputs: when a stage it reads last ran with other settings, or was built from another version of its own inputs, it stops and names the target to re-run, so after `make regions` under another clustering setting, `make regress` asks for `make complexity` first.

`classify` is cached too: re-running it under the same `NAME` skips it when it last ran on the same `split` under the same settings, and retrains by itself — naming the top-level key that differs — when it did not, rewriting its whole folder. Every setting of the classifier, loss, optimizer, scheduler, fit and grid-search groups counts, and the seed, except the device, the parallelism and the data loaders' workers and memory pinning, so even one the classifier ignores retrains it: changing the loss, which only deep classifiers use, retrains a Random Forest as well. So does any recompute of `split`, even under the same settings. The classifier depends on neither the space, the graph, the clustering nor the descriptors, so re-deriving results after changing those leaves `classify` cached. `regress`, `render` and `compare` always recompute. `FORCE=1` recomputes every cached stage and retrains the classifier.

How the data is divided into regions matters. `kmeans` divides this dataset into 707 regions. Density-based clustering such as `hdbscan` looks for genuine gaps between groups, and this dataset's difficulty gradient is continuous, so expect it to find far fewer: count-based algorithms (`kmeans`, `birch`) suit data of this kind, density-based ones suit data with genuine separation.

### Inside the synthetic dataset

Each class consists of a clean core together with a *difficulty ladder* running towards every class it is meant to overlap with (`_OVERLAPS` in [generate_synthetic.py](generate_synthetic.py)). Every rung sits a fixed distance from the boundary between the two classes, and both sides face one another symmetrically, so the best error rate any classifier could reach on a rung is known in advance:

| Sub-group | Distance from boundary | Best possible error | Rows |
|---|---|---|---|
| `canonical` | — | ≈ 0 % | 23,966 |
| `clear` | 1.80 σ | ≈ 4 % | 19,316 |
| `evasive` | 0.90 σ | ≈ 18 % | 16,560 |
| `mimicry` | 0.25 σ | ≈ 40 % | 9,658 |

The two categorical features fade out along the same ladder, so they cease to help exactly where the numbers begin to overlap. No class sits at the origin either: the benign class has features of its own, just as every attack does, which is what allows `DISTANCE=cosine` to see the same structure as `DISTANCE=euclidean`.

Recovering that ladder is the task. Clustering divides the overlap corridors into regions, the descriptors rank them, and the classifier fails progressively more often on the harder ones. Each row's intended difficulty is recorded in the CSV's `true_subgroup` column so that you can check afterwards; the pipeline never reads it, and `split` keeps only the columns the configuration lists.

Accuracy settles at about 0.85 by design. The hard rungs are genuinely hard, and that spread is what makes a per-region correlation meaningful.

## Running on your own data

The configurations in [configs/data/](configs/data/) cover UNSW-NB15, BoT-IoT, CIC-IDS-2018, ToN-IoT and five general benchmarks, but the CSV files are not in the repository: supply them under `resources/raw_data/<dir>/`, matching each configuration's `dir` and `file_name`.

To add a dataset of your own:

1. Place the CSV at `resources/raw_data/<dir>/<file_name>.csv`.
2. Copy a configuration from [configs/data/](configs/data/) and edit it: `label_col`, `num_cols`, `cat_cols`, split fractions, filtering (`filter_query`, `min_class_count`).
3. Add it to `DATASETS` in the [Makefile](Makefile).
4. Run `make run DATA=<name> NAME=my_exp CLASSIFIER=random_forest CLUSTERING=kmeans`.

Every configuration shipped here splits the data 40 % training, 5 % validation and 55 % test (`train_frac`, `val_frac`, `test_frac`). The test split is kept large because it is the only one the estimator's target, every region's error rate, is measured on. The validation split serves deep classifiers, which stop early and keep their best epoch by its loss, the ATC baseline, whose confidence threshold it chooses, the validation-rate baseline, and the calibrated baselines, whose maps it fits.

One known limit: clustering holds each class's points in memory, and the one-hot space widens them. For the largest classes of the network-traffic datasets, millions of rows by up to 165 float32 columns, that matrix reaches several GB; BoT-IoT and CIC-IDS-2018 have not been run at full size.

## How it works

Three steps, together with the classifier whose errors they are fitted on. The method assumes nothing about the classifier, the clustering algorithm or the number of classes; it requires only tabular features.

1. **Identify regions.** Divide each class into compact sub-regions, using the training data alone. The number per class is the candidate of a grid that would misread its samples least: each sample's local hardness, the share of its nearest neighbours in a uniform sample of the training split that belong to another class, stands for the error a region's rate should report about it. A finer partition hides less of that hardness's spread inside each region's average, but leaves each region fewer of the test samples the classifier's evaluation will measure it on, so its rate carries more sampling noise; the candidate with the smallest sum of the two, weighted by region size, is kept. A region below a minimum size, and any point the algorithm leaves as noise, merges into the nearest surviving region of the same class, so no training sample is left out of the descriptors.
2. **Describe each region.** Score how separable it is from the rival regions nearest to it, and do the same at class level. These descriptors are drawn from the data alone; nothing about the classifier enters here.
3. **Learn the mapping.** A Random Forest regresses each region's observed error rate on its descriptors under nested cross-validation, reporting the correlation together with the descriptors that mattered. The choice of a tree-based estimator is deliberate: every risk estimate arrives with the properties that produced it. Ridge regression, XGBoost and a multilayer perceptron are fitted beside it on the same folds, so its correlation can be read against theirs.

The classifier is trained separately and contributes exactly one thing to step 3, namely the error rate it achieved in each region. That is what ties the mapping to the model rather than to the dataset.

Four points worth knowing:

- Every distance — clustering, routing, hardness, the neighbour graph and every descriptor — is measured in one euclidean space: the numerical features as preprocessed, and each categorical feature as a one-hot block over its 16 most frequent training codes plus one slot shared by all the others, weighted so that a mismatch costs as much as one interquartile step on a numerical feature (`space.top_k`, `space.cat_cost`). Under `DISTANCE=cosine` every row is scaled to unit length first. The classifier keeps its own encoding of the categorical features: it is the object under study, not part of the geometry.
- Regions are built from the training split alone, each inside one class. Every test sample is then routed to the region whose centre lies nearest **without reference to its label**, exactly as a sample would be routed at inference time, and a region's error rate counts the classifier's mistakes on the test samples routed to it, whatever their class. Assigned by label instead, a region would hold only its own class and count only that class's mistakes. Routing is what keeps the correlation a genuine prediction rather than a restatement of labels already known.
- The error rates the estimator learns are counted on the test split alone. The training samples placed the region centres with their labels known, so even routed without the label they fall back into their own class's region more often than new data would, and counting them would flatter the regions; only the training-rate baseline and its calibrated variant count the ones the classifier was fitted on, as a comparison. The classifier, the training graph, the regions and the descriptors all come from the training split; no test sample shapes any of them.
- The analysis works chiefly per region rather than per class, since one class usually occupies several separate areas of the space. Class-level figures are kept as a coarse reference.

### What the descriptors measure

Five families, all measured in that one space on each region's own training samples. The neighbourhood and network families read the training graph: its nodes are a uniform sample of the training split, the whole of it up to `graph.max_samples` (50,000) rows, joined by their `graph.k` (30) nearest-neighbour graph and a spanning tree over it; each region's samples look up their nearest nodes there. Above that size a region is measured on at most `complexity.max_sample_per_region` of its samples (200) and a class on `complexity.max_sample_per_class` (2,000), while T2 and T3 still divide by the samples the population really holds. N1 and `hub` count the region's own nodes in the graph, so they are empty for a region with no node there; a single node gives a defined, if noisy, value. A rival is a region of another class. The feature and neighbourhood families and `network_density` compare a region with the ten rival regions whose centres lie nearest to it, reported as min / mean / max across them; every other key is a single value per region.

| Family | Keys | What it captures |
|---|---|---|
| **F** — feature | `f1`–`f4` | how well individual features separate the region from its rivals |
| **N** — neighbourhood | `n1`–`n4` | how many of a region's neighbours belong to another class |
| **ND** — network | `network_density`, `cls_coef`, `hub` | how many neighbour links cross into a rival region |
| **T** — dimensionality | `t2`–`t4` | how many features there are relative to samples, and how many of them matter |
| **G** — geometry | `max_dispersion`, `p95_dispersion`, `dist_to_nearest_rival`, `p5_silhouette`, `frac_at_risk` | how widely the region is spread, and how close the nearest rival lies |

The estimators see every key twice, as `region_<key>` and as `class_<key>` for the region's class. In the demo the Random Forest relies most on `region_f1_max`, `region_f3_max` and `region_network_density_mean`.

## Pipeline reference

Each stage is a script under [stages/](stages/), wrapped by the [Makefile](Makefile). Every stage target takes `DATA`, `NAME`, `SEED`, `CLASSIFIER`, `CLUSTERING`, `DISTANCE`, and `ARGS` for any other override ([Configuration](#configuration)).

| Target | Script | Scope |
|---|---|---|
| `make split` | `split.py` | per dataset, cached |
| `make graph` | `graph.py` | per dataset, cached |
| `make regions` | `regions.py` | per dataset, cached |
| `make complexity` | `complexity.py` | per dataset, cached |
| `make classify` | `classify.py` | per classifier, cached |
| `make regress` | `regress.py` | per classifier |
| `make render` | `render.py` | per classifier |
| `make run` | all of the above | omitted variables iterate, those passed stay fixed |
| `make compare` | `compare.py` | reduces a whole sweep to cross-run figures and tables |
| `make help` | — | every target, with defaults |

**split** produces the splits. It reads only the columns the configuration lists (every column when `data.filter_query` is set, since a filter may name any), removes NaNs, applies the filter, drops the classes with fewer rows than `data.min_class_count` and splits the data stratified. It then fits the preprocessing on the training split: a signed log, `sign(x)·log1p(|x|)`, and a robust scaling of the numerical columns, a top-N and hashed encoding of the categorical ones, and integer class ids in `label`. It saves the three splits holding only the listed columns and `label`, with the original class balance (balancing takes place at training time); the fitted `preprocessor.joblib`; and `meta.json`, which records the columns, every class with its id, name and count per split, and the raw file's row and column counts and its rows per class. A class absent from the validation or test split is kept, with a count of zero there.

**graph** fits the space on the training split — the `space.top_k` most frequent codes of each categorical column — and saves it in `space.json`, which `regions` and `complexity` read. It then draws the nodes of the training graph uniformly from the training split, or takes the whole split when it holds no more than `graph.max_samples` rows, and saves in `graph.npz` which rows they are, each one's `graph.k` nearest neighbours among them and an approximate minimum spanning tree over them. A graph of fewer than `graph.k + 1` nodes lowers `k` to fit.

**regions** clusters each class of the training split, scoring each candidate by the expected squared error of reading its samples' hardness, measured against the graph's nodes, off their region's rate (`loss` in `regions/report.json`, the sum of `loss_within` and `loss_noise`), keeps the lowest, and merges any cluster below `clustering.min_cluster_floor` into the nearest survivor. Each class may hold at most its equal share of `clustering.max_regions`; beyond it the stage stops with an error. How finely each class is divided depends on how many test samples each region's error rate will be counted over, which `regions` estimates from the ratio of test to training rows (`eval_rows_per_train_row` in `regions/report.json`). Every sample of every split then gets one `region` in `assignments.parquet`: a training sample the region its class's clustering drew it into, a validation or test sample the region whose centre lies nearest, found without the label. `nearest_region` holds that label-free region for every sample, the training ones included.

**complexity** describes every region by its own training samples, and every class likewise, against the training graph ([What the descriptors measure](#what-the-descriptors-measure)), into `regions.parquet` and `classes.parquet`.

**classify** trains one classifier and records its per-class metrics and per-sample predictions. `fit.balance=undersample`, the default, undersamples the training split; `fit.balance=none` leaves it intact, and a deep classifier then weights its loss by the original class frequencies instead. The two are alternatives — applying both would correct the same imbalance twice. Setting `fit.n_samples` also caps every class at the same size, so it takes the place of either. One model is trained on the training split — a classical classifier's hyperparameters chosen by a grid search cross-validated there (`grid_search.cv` folds), a deep one's epoch by the validation loss — and it predicts every split. The metrics come from the test predictions; `predictions.parquet` holds, for every sample of every split, the predicted class, the probability of every class (`proba_<class_id>`) and `in_fit`, which marks the training samples left after balancing, the ones the model was fitted on. `training.json` holds the grid search's best parameters and every candidate's scores, and a deep classifier's loss at every training step. A deep classifier also saves the latent embedding of a stratified sample of the test samples, at most 2000 per class, in `latent.parquet`, which `render` projects. `classify` knows nothing of the regions.

**regress** counts each region's errors on the test samples routed to it, from `classify`'s predictions and `regions`' assignments, assembles the table — a region's descriptors, its class's descriptors and its observed error rate — and fits four estimators on it, a Random Forest, ridge regression, XGBoost and a multilayer perceptron, over the same five outer folds, tuning each on five inner folds. Each is listed under `failure_regressor.models` with its fixed `params` and the `param_grid` its search draws from; ridge and the perceptron see the descriptors imputed and standardised, the trees as they are, and every prediction is clipped to [0, 1], the range of a rate. The primary estimator, `failure_regressor.primary`, the Random Forest by default, is the estimate: its held-out predictions are `predicted_rate`, it is the regressor the baselines are compared with, and its scores, best parameters per fold (`per_fold`) and feature importances head `results.json`, while `models` and `model_folds` there score all four, overall and per fold. Only an estimator that reports feature importances can be primary: naming ridge or the perceptron stops the stage with an error. A region with no test sample, or fewer than `failure_regressor.min_eval_support`, is left out and counted (`n_regions_total`, `n_regions_used` in `results.json`). It also compares the estimate against confidence-based baselines (MCP, one minus the classifier's top class probability, ATC, and rank-averaged combinations of each with the regressor), with ATC's threshold chosen on the validation split, against two empirical rates, and against MCP, ATC and the training rate calibrated on the validation split. When the error rates leave nothing to learn, fewer than two usable regions or no variance among them, the estimators are skipped and `baselines.json` is not written.

The two empirical baselines need no estimator: `train_rate_region` is each region's error rate on the training samples the classifier was fitted on, `val_rate_region` its rate on the validation samples routed to it. Both count samples by their label-free `nearest_region`, as the target counts test samples: a region's training samples are all of its own class, while the test samples routed to it are not, and those of other classes carry a large share of its errors. `regions.parquet` keeps both as observed, `fit_failure_rate` and `val_failure_rate`, empty where a region has no such sample, with their sample counts `n_fit` and `n_val` (routed) and the region's training-sample count `n_train` (clustered); none of them reaches the estimators. As a baseline, a region with no such sample takes the rate pooled over its class's regions, a class with none the rate pooled over every region, and `n_regions_without_fit` and `n_regions_without_val` in `baselines.json` count the scored regions that fell back. Neither is a clean reference. A classifier that makes no mistake on the samples it was fitted on — a Random Forest grown to pure leaves, which the demo's grid search picks, is one — has a training rate of zero in every region, so its ρ is undefined, `null`, and its oracle benefit, near zero, reflects only the order of tied samples. A deep classifier keeps the epoch with the lowest validation loss, so its validation rate is slightly optimistic. `error_by_size` sorts the scored regions by `n_train` into five bins of equal count and gives, per bin, for each of eight predictions — the regressor, both rates, MCP and ATC averaged over each region, and the three calibrated variants below — the mean squared and signed error (predicted − observed) with their standard errors, and the Spearman ρ between prediction and observed rate over the bin's regions, `null` where either is constant there; with fewer than five scored regions it is empty.

MCP, ATC and the training rate need not lie on the scale of the test error rate, so each has a calibrated variant, `mcp_region_cal`, `atc_region_cal` and `train_rate_region_cal`. Each is a Platt map, the logistic `expit(a + b·score)` with `b ≥ 0`, fitted over the regions that hold a validation sample by the cross-entropy between the map and each region's validation error rate. Every region weighs the same in the fit, as it does in the ρ and the MSE the variant is judged by; weighted by their samples, a few large regions can pull the slope to zero for all the others. The score is the region's mean MCP over its validation samples, the share of them under the ATC threshold, or its training rate, standardised for the fit, and the intercept and slope reported apply to it unstandardised. The map is then applied to the score on test — the mean MCP and the share under the threshold over the region's test samples, the training rate as it is — so no test sample sets the scale it is judged on. A slope that cannot turn negative keeps the raw variant's order, so the calibrated variant has the same ρ and oracle benefit recovered up to float resolution — raw scores closer than about 1e-16 to each other, as MCP's near 0, tie once mapped — and only `region_rate_mse` tells the two apart; a slope of zero makes the calibrated variant constant instead, and the stage logs a warning when the score itself was not. `calibration` in `baselines.json` holds each map's `variant`, `intercept` and `slope`, and `n_regions_val` counts the regions the maps were fitted over. When the validation samples hold no error, or nothing but errors, there is no scale to fit: the stage logs a warning and writes everything else, and the calibrated variants' intercept, slope, ρ, oracle benefit, `region_rate_mse` and by-size errors are `null`. The regressor is not calibrated, being fitted on the test rate already, and neither is the validation rate, which is the very rate the maps are fitted to.

**render** turns the `classify`, `complexity` and `regress` artefacts into figures. Under `classification/` it draws the classifier's own: the confusion matrix, the per-class F1 and a t-SNE projection of the test samples, and for a deep classifier a t-SNE of their latent embedding and the training-loss curve, all of them even when the estimators were skipped. The others include every estimator's ρ and MSE from `results.json`'s `models`, in `correlation/regressor_comparison`, and `baselines.json`'s error and ρ by region size, under `baselines/`, where the squared and signed error figures set each calibrated variant beside its raw one. `figure_format` selects `pdf`, the default, or `png`.

### Sweeps

```bash
make run NAME=x                             # every dataset × classifier × clustering algorithm
make run NAME=x DATA=covertype              # one dataset, every compatible classifier
make run NAME=x CLASSIFIER=random_forest    # every dataset, one classifier
make compare FIGURES_DIR=paper/figures
```

Omit `CLUSTERING` and the sweep runs every algorithm into a separate `NAME_<algo>` tree. `compare` then reduces every tree under `SWEEP_DIR` (`resources/experiments` by default) to the cross-run figures (ρ per configuration, ρ against region count, family importance, per-classifier and per-dataset baseline comparisons, error and ρ by region size), written to `FIGURES_DIR`, and the matching JSON tables, written to `SWEEP_DIR/compare/`. A calibrated baseline ranks as its raw one, so it appears only where the error is compared: the MSE by classifier and by baseline, and the error by region size. It stops, asking for `make regress`, when a run's `regress` was built from other stages than those on disk.

## Configuration

The root configuration is [configs/config.yaml](configs/config.yaml); it and every group file under [configs/](configs/) document their parameters inline. Any key may be overridden on the command line, through `make` or by calling the script directly:

```bash
make classify DATA=bot_iot_v2 NAME=my_exp SEED=123 CLASSIFIER=random_forest ARGS="fit.n_samples=10000 grid_search.max_samples=5000"
PYTHONPATH=. python stages/classify.py data=bot_iot_v2 name=my_exp seed=123 classifier=random_forest fit.n_samples=10000 grid_search.max_samples=5000
```

`make` turns `DATA`, `NAME`, `SEED`, `CLASSIFIER`, `CLUSTERING` and `DISTANCE` into their Hydra keys. `ARGS` comes last on every stage, so its overrides win, `force` included. The six keys that have a variable of their own are refused there, while nested keys such as `data.label_col` are fine. A variable the Makefile does not recognise stops it with an error, so a mistyped override is never silently ignored.

A shell reads `ARGS` before Hydra does, so single-quote an override whose value holds spaces or braces, as in `ARGS="'failure_regressor.models.random_forest.param_grid.n_estimators=[100, 300]'"`. Pass a `${…}` interpolation by calling the script directly: make and the shell would both expand it first.

| Group | Options |
|---|---|
| `data` | network traffic: `nb15_v2`, `bot_iot_v2`, `cic_2018_v2`, `ton_iot_v2` · benchmarks: `bank_marketing`, `covertype`, `letter_recognition`, `statlog_landsat_satellite`, `thyroid_disease` · `synthetic_test`, the only one needing no external CSV |
| `classifier` | deep: `mlp` (adapts to the dataset's numerical/categorical feature counts) · classical: `decision_tree`, `random_forest`, `hist_gradient_boosting`, `xgboost`, `knn`, `lda`, `logistic_regression`, `naive_bayes`, `linear_svc` |
| `space` | `default` — the one space every distance is measured in: one-hot slots per categorical column (`top_k`) and the cost of a mismatch (`cat_cost`) |
| `graph` | `default` — the training graph's node cap (`max_samples`) and its neighbours per node (`k`) |
| `clustering` | `kmeans`, `hdbscan`, `birch`, `spectral` — each with the region cap over all classes (`max_regions`), the merge floor and the hardness neighbours |
| `complexity` | `default` — rival regions per region (`top_k_clusters`), the samples a region and a class are measured on above `graph.max_samples` (`max_sample_per_region`, `max_sample_per_class`), silhouette subsample size and its per-cluster floor |
| `failure_regressor` | `default` — the estimators (`models`, each with its fixed `params` and searched `param_grid`), the `primary` one, nested-CV folds, the draws per search (`n_iter`; a smaller grid is searched whole) and the test samples a region needs (`min_eval_support`) |
| `fit` | `default` — how the classifier is trained: training-split balancing (`balance`, `n_samples`) and, for deep classifiers, `device`, epochs, gradient clipping, early stopping and data loaders |
| `grid_search` | `default` — scoring, CV folds and sample cap for classifier tuning |
| `loss` / `optimizer` / `scheduler` | deep learning only: `cross_entropy` \| `focal` / `adamw` / `one_cycle` |
| `path` | `default` |

Results are written to:

```
resources/experiments/${name}/${data.file_name}_${seed}/
├── split/                  # the splits, the fitted preprocessing, meta.json — shared
├── graph/                  # the space and the training graph — shared
├── regions/                # centroids, every sample's region, the clustering report — shared
├── complexity/             # descriptors per region and per class — shared
└── ${classifier.name}/
    ├── classify/           # model, predictions, metrics, training record, latent sample (deep only)
    ├── regress/            # error rates per region, estimator results, baselines
    └── render/             # figures
```

Each stage owns one folder and rewrites it whole whenever it recomputes. Its `record.json`, written last, holds the configuration keys the stage depends on (overrides included), the ids of the stages it read and an id of its own, new on every write, so a run interrupted before the end is recomputed rather than trusted.

## Repository layout

```
intrusion-forge/
├── stages/                       # entry points — own the config, I/O, logging and paths
│   ├── __init__.py               #   what each stage's record holds and reads, upstream checks
│   ├── split.py                  #   filter, split, preprocess
│   ├── graph.py                  #   space + training graph (uniform sample, k-NN, MST)
│   ├── regions.py                #   divide each class into regions, route every sample
│   ├── complexity.py             #   region and class descriptors
│   ├── classify.py               #   train + evaluate one classifier
│   ├── regress.py                #   descriptors → error rate, baselines
│   ├── render.py                 #   figures
│   └── compare.py                #   cross-run aggregation
├── generate_synthetic.py         # synthetic dataset generator
├── Makefile                      # experiment runner
├── configs/                      # Hydra hierarchy
├── src/                          # pure library — no config, no I/O, no path building
│   ├── core/                     # config, Factory, file I/O, stage records, RunPaths, logging, timing
│   ├── domain/
│   │   ├── data/                 # cleaning, splitting, scaling, encoding, the one space
│   │   ├── clustering/           # the four algorithms, the loss-scored grid search, kDN hardness
│   │   ├── analysis/complexity/  # the F / N / ND / T / G families
│   │   ├── analysis/             # metadata, classification metrics, confidence & risk-coverage scores, the failure regressor, baseline calibration
│   │   ├── training/             # ml.py (sklearn / XGBoost), dl.py (Ignite loop), weighting.py (class weights)
│   │   ├── plot/                 # Plot payload, chart primitives, classify/analysis/comparison composers, metrics, palette
│   │   └── projection.py         # t-SNE
│   └── engine/
│       ├── dl/                   # models, modules, losses, dataset, Ignite engine builder
│       └── ml/                   # sklearn / XGBoost wrappers, column preprocessing
└── resources/                    # not tracked by git: raw_data/ in, experiments/ out
```

Three conventions to know before editing:

- `src/` is input to output. Configuration loading, file I/O and path building belong to `stages/` and `src/core` alone.
- A stage writes only into its own folder, directly through `src/core` (`save_df`, `save_figures`, `save_arrays`, `save_to_json`, `save_to_joblib`), and writes its `record.json` last; model persistence belongs to the `Trainer`.
- Classifiers, failure regressors and losses register themselves with a factory (`@DLClassifierFactory.register()`, `@LossFactory.register()`, `MLClassifierFactory.register("name")(SklearnClass)` or `MLRegressorFactory.register("name")(SklearnClass)`) and are discovered automatically at import; a failure regressor also needs its row in `REGRESSOR_PREPROCESS` in `src/engine/ml/preprocessing.py`. Each factory is defined in the `factory.py` of the package that holds its components (`src/engine/ml/model/`, `src/engine/dl/model/`, `src/engine/dl/loss/`, `src/domain/clustering/`), and a new module imports it from there.
