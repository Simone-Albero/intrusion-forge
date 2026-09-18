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
#   make run            NAME=my_exp DATA=cic_2018_v2 CLASSIFIER=tabular CLUSTERING=kmeans   # single (ds, clf, clustering)
#
# Single-stage targets (DATA + CLASSIFIER explicit):
#   make prepare           DATA=cic_2018_v2 NAME=my_exp
#   make classify          DATA=cic_2018_v2 NAME=my_exp CLASSIFIER=random_forest
#   make complexity        DATA=cic_2018_v2 NAME=my_exp                     # shared, dataset-level
#   make failure-regress   DATA=cic_2018_v2 NAME=my_exp CLASSIFIER=random_forest
#   make render            DATA=cic_2018_v2 NAME=my_exp CLASSIFIER=random_forest
#
# Flags:
#   FORCE=1               re-run shared stages (prepare, complexity), ignoring skip markers
#   CLUSTERING=<name>     fix the clustering strategy (kmeans/hdbscan/birch/spectral);
#                         omit it in `run` to sweep all of CLUSTERING_ALGOS into NAME_<algo>
#
# k-fold note: k-fold evaluation (kfold=true) is disabled automatically for LARGE_DATASETS
#   (nb15_v2, bot_iot_v2, cic_2018_v2, ton_iot_v2) because millions of rows make it impractical.
#   Override per-call: make classify DATA=cic_2018_v2 ... KFOLD=true
# ──────────────────────────────────────────────────────────────────────────────

# Use venv if present; falls back to the active conda (or system) python otherwise.
# Override explicitly: make <target> PYTHON=python
PYTHON    ?= $(if $(wildcard venv/bin/python),venv/bin/python,python)
STREAMLIT ?= $(if $(wildcard venv/bin/streamlit),venv/bin/streamlit,streamlit)
DATA       ?= cic_2018_v2
NAME       ?= exp_euc
SEED       ?= 42
CLASSIFIER ?= tabular
DISTANCE   ?= euclidean
CLUSTERING ?= kmeans
CLUSTERING_ALGOS ?= kmeans spectral birch hdbscan
FORCE      ?=

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

DL_CLASSIFIERS := tabular

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

# Datasets too large for k-fold evaluation (millions of rows → hours per classifier).
# kfold=false is injected automatically for these; override with KFOLD=true if needed.
LARGE_DATASETS := nb15_v2 bot_iot_v2 cic_2018_v2 ton_iot_v2

HYDRA       := data=$(DATA) name=$(NAME) seed=$(SEED) classifier=$(CLASSIFIER) \
               clustering=$(CLUSTERING) distance=$(DISTANCE)
FORCE_FLAG  := $(if $(FORCE),force=true,)
KFOLD       ?= $(if $(filter $(DATA),$(LARGE_DATASETS)),false,true)
KFOLD_FLAG  := kfold=$(KFOLD)

# Cross-run paper comparisons: aggregate the full experiment tree under SWEEP_DIR into the
# cross-run figures (rho by config / vs clusters, family importance, per-classifier and
# per-dataset baseline comparisons) written to FIGURES_DIR, and the Results-section JSON
# tables (perconfig, nclusters, datasets, variant comparisons), written back under SWEEP_DIR.
SWEEP_DIR       ?= resources/experiments
FIGURES_DIR     ?= paper/figures

.PHONY: prepare classify complexity failure-regress render comparisons run generate dashboard help

## prepare:            Step 1 — preprocess raw CSV → parquet splits           (DATA, NAME, SEED, FORCE)
prepare:
	PYTHONPATH=. $(PYTHON) pipelines/prepare_data.py $(HYDRA) $(FORCE_FLAG)

## classify:           Step 2 — train & evaluate one classifier (ML or DL)    (DATA, NAME, SEED, CLASSIFIER)
classify:
	PYTHONPATH=. $(PYTHON) pipelines/classify.py $(HYDRA) $(KFOLD_FLAG)

## complexity:         Step 3a — cluster + class complexity (shared, idempotent)  (DATA, NAME, SEED, FORCE)
complexity:
	PYTHONPATH=. $(PYTHON) pipelines/compute_complexity.py $(HYDRA) $(FORCE_FLAG)

## failure-regress:    Step 3b — RF to detect problematic clusters            (DATA, NAME, SEED, CLASSIFIER)
failure-regress: complexity
	PYTHONPATH=. $(PYTHON) pipelines/fit_failure_regressor.py $(HYDRA)

## render:             Step 4 — render plots from analysis artifacts          (DATA, NAME, SEED, CLASSIFIER)
render:
	PYTHONPATH=. $(PYTHON) pipelines/render_plots.py $(HYDRA)

## comparisons:        Aggregate the experiment tree into cross-run paper figures + result tables  (SWEEP_DIR, FIGURES_DIR)
comparisons:
	PYTHONPATH=. $(PYTHON) pipelines/comparisons.py sweep=$(SWEEP_DIR) out=$(FIGURES_DIR)
	@echo ""; echo "comparisons done -> $(FIGURES_DIR)/{rho_by_config,rho_vs_clusters,family_importance,spearman_by_classifier,mse_by_classifier,oracle_benefit_by_variant,spearman_by_dataset}.pdf + $(SWEEP_DIR)/{perconfig,nclusters,datasets,variant_spearman,variant_cluster_mse,variant_spearman_by_dataset}_table.json"

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
			$(MAKE) --no-print-directory prepare \
				DATA=$$ds NAME=$$name SEED=$(SEED) CLUSTERING=$$clu \
				DISTANCE=$(DISTANCE) $(FORCE_FLAG) || exit 1; \
			$(MAKE) --no-print-directory complexity \
				DATA=$$ds NAME=$$name SEED=$(SEED) CLUSTERING=$$clu \
				DISTANCE=$(DISTANCE) $(FORCE_FLAG) || exit 1; \
			for clf in $$clf_list; do \
				echo ""; \
				echo "── classifier: $$clf ─────────────────────────────"; \
				$(MAKE) --no-print-directory classify \
					DATA=$$ds NAME=$$name SEED=$(SEED) CLASSIFIER=$$clf \
					CLUSTERING=$$clu DISTANCE=$(DISTANCE) || exit 1; \
				$(MAKE) --no-print-directory failure-regress \
					DATA=$$ds NAME=$$name SEED=$(SEED) CLASSIFIER=$$clf \
					CLUSTERING=$$clu DISTANCE=$(DISTANCE) || exit 1; \
				$(MAKE) --no-print-directory render \
					DATA=$$ds NAME=$$name SEED=$(SEED) CLASSIFIER=$$clf \
					CLUSTERING=$$clu DISTANCE=$(DISTANCE) || exit 1; \
			done; \
		done; \
	done
	@echo ""
	@echo "Done."

## generate:           Generate synthetic test dataset                        (ROWS)
generate:
	$(PYTHON) generate_synthetic.py $(if $(ROWS),--rows $(ROWS),)

## dashboard:          Open the experiment dashboard in browser
dashboard:
	$(STREAMLIT) run dashboard.py

## help:               Show this help message
help:
	@echo "Usage: make <target> [DATA=<dataset>] [NAME=<name>] [SEED=<n>] [CLASSIFIER=<name>] [DISTANCE=<dist>]"
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
	@echo "Datasets (smallest → largest, kfold auto-disabled for large):"
	@echo "  small (kfold=true):   statlog_landsat_satellite  thyroid_disease  letter_recognition  bank_marketing  covertype"
	@echo "  large (kfold=false):  nb15_v2  ton_iot_v2  cic_2018_v2  bot_iot_v2"
	@echo ""
	@echo "Run examples (omitted vars iterate; passed vars are fixed):"
	@echo "  make run NAME=x                                      # all datasets × all classifiers × all clustering algos"
	@echo "  make run NAME=x DATA=letter_recognition              # 1 dataset, all classifiers, all clustering"
	@echo "  make run NAME=x CLASSIFIER=random_forest             # all datasets, 1 classifier, all clustering"
	@echo "  make run NAME=x CLUSTERING=kmeans                    # all datasets × all classifiers, 1 clustering"
	@echo "  make run NAME=x DATA=cic_2018_v2 CLASSIFIER=tabular CLUSTERING=kmeans  # single"
	@echo "  (clustering swept → artifacts land under NAME_<algo>; clustering fixed → under NAME)"
