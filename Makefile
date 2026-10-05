# ──────────────────────────────────────────────────────────────────────────────
# Intrusion Forge — Experiment Runner
#
# One parametric command for sweeps. Variables passed on the command line are
# FIXED; those omitted are ITERATED.
#
#   make run            NAME=my_exp                                # all datasets × ML+DL × all clustering algos
#   make run            NAME=my_exp DATA=letter_recognition        # 1 dataset × all classifiers × all clustering
#   make run            NAME=my_exp CLASSIFIER=random_forest       # all datasets × 1 classifier × all clustering
#   make run            NAME=my_exp CLUSTERING=kmeans              # all datasets × all classifiers × 1 clustering
#   make run            NAME=my_exp DATA=cic_2018_v2 CLASSIFIER=mlp CLUSTERING=kmeans       # single (ds, clf, clustering)
#
# Single-stage targets (dataset-level stages need no CLASSIFIER):
#   make split             DATA=cic_2018_v2 NAME=my_exp
#   make graph             DATA=cic_2018_v2 NAME=my_exp
#   make regions           DATA=cic_2018_v2 NAME=my_exp
#   make complexity        DATA=cic_2018_v2 NAME=my_exp
#   make classify          DATA=cic_2018_v2 NAME=my_exp CLASSIFIER=random_forest
#   make regress           DATA=cic_2018_v2 NAME=my_exp CLASSIFIER=random_forest
#   make render            DATA=cic_2018_v2 NAME=my_exp CLASSIFIER=random_forest
#
# Flags:
#   FORCE=1               re-run cached stages (split, graph, regions, complexity) and retrain the
#                         classifier even when a model for this exact config already exists
#   CLUSTERING=<name>     fix the clustering strategy (kmeans/hdbscan/birch/spectral);
#                         omit it in `run` to sweep all of CLUSTERING_ALGOS into NAME_<algo>
#   ARGS="k=v ..."        extra Hydra overrides, passed last to every stage, so they win
#                         (e.g. ARGS="fit.n_samples=10000 grid_search.max_samples=5000");
#                         data, name, seed, classifier, clustering and distance keep their
#                         own variables, and any other variable on the make line is an error
# ──────────────────────────────────────────────────────────────────────────────

# Use venv if present; falls back to the active conda (or system) python otherwise.
# Override explicitly: make <target> PYTHON=python
PYTHON     ?= $(if $(wildcard venv/bin/python),venv/bin/python,python)
DATA       ?= cic_2018_v2
NAME       ?= exp_euc
SEED       ?= 42
CLASSIFIER ?= mlp
DISTANCE   ?= euclidean
CLUSTERING ?= kmeans
CLUSTERING_ALGOS ?= kmeans spectral birch hdbscan
FORCE      ?=
# := rather than ?=: an ARGS exported in the shell would otherwise reach every stage unseen.
ARGS       :=

# A variable make does not know is a Hydra override typed in the wrong place: stop, don't drop it.
MAKE_VARS    := DATA NAME SEED CLASSIFIER DISTANCE CLUSTERING CLUSTERING_ALGOS FORCE ARGS \
                DATASETS ML_CLASSIFIERS DL_CLASSIFIERS \
                PYTHON SWEEP_DIR FIGURES_DIR ROWS
UNKNOWN_VARS := $(filter-out $(MAKE_VARS),$(foreach v,$(.VARIABLES),$(if $(filter command line,$(origin $(v))),$(v))))
ifneq ($(UNKNOWN_VARS),)
$(error Unknown make variable(s): $(UNKNOWN_VARS). Hydra overrides go through ARGS="key=value ...")
endif

# ARGS may override force, but not the keys make itself decides from — the NAME_<algo>
# directories of `run`: those have their own variables.
ARGS_CLASH := $(filter $(foreach k,data name seed classifier clustering distance,$(k)=% ++$(k)=%),$(ARGS))
ifneq ($(ARGS_CLASH),)
$(error ARGS cannot set $(ARGS_CLASH): use DATA, NAME, SEED, CLASSIFIER, CLUSTERING or DISTANCE)
endif

# `run` distinguishes "passed on the command line" from "default" via $(origin).
DATA_GIVEN    := $(if $(filter command line,$(origin DATA)),1,)
CLF_GIVEN     := $(if $(filter command line,$(origin CLASSIFIER)),1,)
CLUSTER_GIVEN := $(if $(filter command line,$(origin CLUSTERING)),1,)

ML_CLASSIFIERS := \
    naive_bayes \
    logistic_regression \
    lda \
    knn \
    decision_tree \
    random_forest \
    hist_gradient_boosting \
    linear_svc \
    xgboost

DL_CLASSIFIERS := mlp

DATASETS := \
    statlog_landsat_satellite \
    thyroid_disease \
    letter_recognition \
    bank_marketing \
    covertype \
    nb15_v2 \
    ton_iot_v2 \
    cic_2018_v2 \
    bot_iot_v2 \
    synthetic_test

# Every stage gets $(HYDRA); FORCE_FLAG reaches only the cached stages.
# Recipes put $(ARGS) last, so an explicit override wins.
HYDRA       := data=$(DATA) name=$(NAME) seed=$(SEED) classifier=$(CLASSIFIER) \
               clustering=$(CLUSTERING) distance=$(DISTANCE)
FORCE_FLAG  := $(if $(FORCE),force=true,)

# Cross-run comparisons: aggregate the full experiment tree under SWEEP_DIR into the
# cross-run figures (rho by config / vs regions, family importance, per-classifier and
# per-dataset baseline comparisons, error by region size) written to FIGURES_DIR, and the
# JSON tables (perconfig, nregions, datasets, variant comparisons, error by size), written
# under SWEEP_DIR/compare.
SWEEP_DIR       ?= resources/experiments
FIGURES_DIR     ?= paper/figures

.PHONY: split graph regions complexity classify regress render compare run generate help

## split:              Step 1 — filter, split and preprocess the raw CSV   (DATA, NAME, SEED, FORCE)
split:
	PYTHONPATH=. $(PYTHON) stages/split.py $(HYDRA) $(FORCE_FLAG) $(ARGS)

## graph:              Step 2 — uniform sample of train, its k-NN graph and its MST   (DATA, NAME, SEED, DISTANCE, FORCE)
graph:
	PYTHONPATH=. $(PYTHON) stages/graph.py $(HYDRA) $(FORCE_FLAG) $(ARGS)

## regions:            Step 3 — cluster train per class, route every split   (DATA, NAME, SEED, CLUSTERING, DISTANCE, FORCE)
regions:
	PYTHONPATH=. $(PYTHON) stages/regions.py $(HYDRA) $(FORCE_FLAG) $(ARGS)

## complexity:         Step 4 — region and class complexity descriptors   (DATA, NAME, SEED, CLUSTERING, DISTANCE, FORCE)
complexity:
	PYTHONPATH=. $(PYTHON) stages/complexity.py $(HYDRA) $(FORCE_FLAG) $(ARGS)

## classify:           Step 5 — train & evaluate one classifier (ML or DL)   (DATA, NAME, SEED, CLASSIFIER, FORCE)
classify:
	PYTHONPATH=. $(PYTHON) stages/classify.py $(HYDRA) $(FORCE_FLAG) $(ARGS)

## regress:            Step 6 — regressors predicting each region's failure rate   (DATA, NAME, SEED, CLASSIFIER)
regress:
	PYTHONPATH=. $(PYTHON) stages/regress.py $(HYDRA) $(ARGS)

## render:             Step 7 — render plots from the classify and regress artifacts   (DATA, NAME, SEED, CLASSIFIER)
render:
	PYTHONPATH=. $(PYTHON) stages/render.py $(HYDRA) $(ARGS)

## compare:            Aggregate the experiment tree into cross-run figures + result tables  (SWEEP_DIR, FIGURES_DIR)
compare:
	PYTHONPATH=. $(PYTHON) stages/compare.py sweep=$(SWEEP_DIR) out=$(FIGURES_DIR)
	@echo ""; echo "compare done -> $(FIGURES_DIR)/{rho_by_config,rho_vs_regions,family_importance,spearman_by_classifier,mse_by_classifier,oracle_benefit_by_variant,spearman_by_dataset,mse_by_region_size,bias_by_region_size,spearman_by_region_size}.pdf + $(SWEEP_DIR)/compare/{perconfig,nregions,datasets,variant_spearman,variant_region_mse,variant_spearman_by_dataset,error_by_size}_table.json"

## run:                Whole flow — fix passed vars, iterate the rest (DATA?, CLASSIFIER?, CLUSTERING?)  (NAME, SEED, DISTANCE, FORCE)
run:
	@data_given="$(DATA_GIVEN)"; \
	clf_given="$(CLF_GIVEN)"; \
	clu_given="$(CLUSTER_GIVEN)"; \
	requested_data="$(DATA)"; \
	if [ -n "$$clu_given" ]; then clu_list="$(CLUSTERING)"; else clu_list="$(CLUSTERING_ALGOS)"; fi; \
	if [ -n "$$data_given" ]; then \
		ds_list=""; \
		for ds in $(DATASETS); do \
			if [ "$$ds" = "$$requested_data" ]; then ds_list="$$ds"; break; fi; \
		done; \
		if [ -z "$$ds_list" ]; then \
			echo "ERROR: DATA='$$requested_data' not in DATASETS."; exit 1; \
		fi; \
	else \
		ds_list="$(DATASETS)"; \
	fi; \
	if [ -n "$$clf_given" ]; then clf_list="$(CLASSIFIER)"; else clf_list="$(ML_CLASSIFIERS) $(DL_CLASSIFIERS)"; fi; \
	for clu in $$clu_list; do \
		if [ -n "$$clu_given" ]; then name="$(NAME)"; else name="$(NAME)_$$clu"; fi; \
		echo ""; \
		echo "##############################################"; \
		echo " CLUSTERING = $$clu   →   name=$$name"; \
		echo "##############################################"; \
		for ds in $$ds_list; do \
			echo ""; \
			echo "══════════════════════════════════════════════"; \
			echo " Dataset: $$ds  |  name=$$name  seed=$(SEED)"; \
			echo "══════════════════════════════════════════════"; \
			for step in split graph regions complexity; do \
				$(MAKE) --no-print-directory $$step \
					DATA=$$ds NAME=$$name SEED=$(SEED) CLUSTERING=$$clu \
					DISTANCE=$(DISTANCE) FORCE=$(FORCE) || exit 1; \
			done; \
			for clf in $$clf_list; do \
				echo ""; \
				echo "── classifier: $$clf ─────────────────────────────"; \
				$(MAKE) --no-print-directory classify \
					DATA=$$ds NAME=$$name SEED=$(SEED) CLASSIFIER=$$clf \
					CLUSTERING=$$clu DISTANCE=$(DISTANCE) FORCE=$(FORCE) || exit 1; \
				for step in regress render; do \
					$(MAKE) --no-print-directory $$step \
						DATA=$$ds NAME=$$name SEED=$(SEED) CLASSIFIER=$$clf \
						CLUSTERING=$$clu DISTANCE=$(DISTANCE) || exit 1; \
				done; \
			done; \
		done; \
	done
	@echo ""
	@echo "Done."

## generate:           Generate synthetic test dataset                        (ROWS)
generate:
	$(PYTHON) generate_synthetic.py $(if $(ROWS),--rows $(ROWS),)

## help:               Show this help message
help:
	@echo "Usage: make <target> [DATA=<dataset>] [NAME=<name>] [SEED=<n>] [CLASSIFIER=<name>] [DISTANCE=<dist>] [ARGS=\"k=v ...\"]"
	@echo ""
	@echo "Targets:"
	@grep -E '^## ' Makefile | sed 's/## /  /'
	@echo ""
	@echo "Defaults:  DATA=$(DATA)  NAME=$(NAME)  SEED=$(SEED)  CLASSIFIER=$(CLASSIFIER)  DISTANCE=$(DISTANCE)  CLUSTERING=$(CLUSTERING)"
	@echo "Python:    $(PYTHON)  (override with PYTHON=)"
	@echo "ML classifiers: $(ML_CLASSIFIERS)"
	@echo "DL classifiers: $(DL_CLASSIFIERS)"
	@echo "Clustering strategies:  kmeans hdbscan birch spectral"
	@echo ""
	@echo "Datasets (smallest → largest):"
	@echo "  statlog_landsat_satellite  thyroid_disease  letter_recognition  bank_marketing  covertype  nb15_v2  ton_iot_v2  cic_2018_v2  bot_iot_v2"
	@echo ""
	@echo "Run examples (omitted vars iterate; passed vars are fixed):"
	@echo "  make run NAME=x                                      # all datasets × all classifiers × all clustering algos"
	@echo "  make run NAME=x DATA=letter_recognition              # 1 dataset, all classifiers, all clustering"
	@echo "  make run NAME=x CLASSIFIER=random_forest             # all datasets, 1 classifier, all clustering"
	@echo "  make run NAME=x CLUSTERING=kmeans                    # all datasets × all classifiers, 1 clustering"
	@echo "  make run NAME=x DATA=cic_2018_v2 CLASSIFIER=mlp CLUSTERING=kmeans      # single"
	@echo "  (clustering swept → artifacts land under NAME_<algo>; clustering fixed → under NAME)"
