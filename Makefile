# ──────────────────────────────────────────────────────────────────────────────
# Intrusion Forge — Experiment Runner
#
# Usage:
#   make prepare   DATA=cic_2018_v2 NAME=my_exp
#   make classify  DATA=cic_2018_v2 NAME=my_exp CLASSIFIER=random_forest
#   make ml-all    DATA=cic_2018_v2 NAME=my_exp     # every ML classifier
#   make dl-all    DATA=cic_2018_v2 NAME=my_exp     # every DL classifier
#   make analyze   DATA=cic_2018_v2 NAME=my_exp CLASSIFIER=random_forest
#   make render    DATA=cic_2018_v2 NAME=my_exp CLASSIFIER=random_forest
#   make run       DATA=cic_2018_v2 NAME=my_exp CLASSIFIER=tabular
#   make all       NAME=my_exp                       # all datasets, default DL classifier
# ──────────────────────────────────────────────────────────────────────────────

PYTHON     := venv/bin/python
DATA       ?= cic_2018_v2
NAME       ?= exp
SEED       ?= 42
CLASSIFIER ?= tabular
DISTANCE   ?= cosine

ML_CLASSIFIERS := \
    naive_bayes \
    logistic_regression \
    lda \
    knn \
    decision_tree \
    random_forest \
    hist_gradient_boosting \
    svm_rbf \
    xgboost

DL_CLASSIFIERS := \
    tabular \
    numerical \
    categorical

DATASET_CLASSIFIERS := \
    nb15_v2:tabular \
    bot_iot_v2:tabular \
    cic_2018_v2:tabular \
    ton_iot_v2:tabular \
    bank_marketing:tabular \
    covertype:numerical \
    letter_recognition:numerical \
    statlog_landsat_satellite:numerical \
    thyroid_disease:numerical

HYDRA := data=$(DATA) name=$(NAME) seed=$(SEED) classifier=$(CLASSIFIER) \
         complexity.distance=$(DISTANCE) clustering.distance=$(DISTANCE)

.PHONY: prepare classify ml-all dl-all analyze render run all generate dashboard help

## prepare:            Step 1 — preprocess raw CSV → parquet splits           (DATA, NAME, SEED)
prepare:
	PYTHONPATH=. $(PYTHON) pipelines/prepare_data.py $(HYDRA)

## classify:           Step 2 — train & evaluate one classifier (ML or DL)    (DATA, NAME, SEED, CLASSIFIER)
classify:
	PYTHONPATH=. $(PYTHON) pipelines/classify.py $(HYDRA)

## ml-all:             Step 2 — train & evaluate every ML classifier in turn  (DATA, NAME, SEED)
ml-all:
	@for clf in $(ML_CLASSIFIERS); do \
		echo ""; \
		echo "── ML classifier: $$clf ─────────────────────────────"; \
		$(MAKE) --no-print-directory classify \
			DATA=$(DATA) NAME=$(NAME) SEED=$(SEED) CLASSIFIER=$$clf \
			DISTANCE=$(DISTANCE) || exit 1; \
	done

## dl-all:             Step 2 — train & evaluate every DL classifier in turn  (DATA, NAME, SEED)
dl-all:
	@for clf in $(DL_CLASSIFIERS); do \
		echo ""; \
		echo "── DL classifier: $$clf ─────────────────────────────"; \
		$(MAKE) --no-print-directory classify \
			DATA=$(DATA) NAME=$(NAME) SEED=$(SEED) CLASSIFIER=$$clf \
			DISTANCE=$(DISTANCE) || exit 1; \
	done

## analyze:            Step 3 — post-hoc analysis (compute only)              (DATA, NAME, SEED, CLASSIFIER)
analyze:
	PYTHONPATH=. $(PYTHON) pipelines/analyze_data.py $(HYDRA)

## render:             Step 4 — render plots from analysis artifacts          (DATA, NAME, SEED, CLASSIFIER)
render:
	PYTHONPATH=. $(PYTHON) pipelines/render_plots.py $(HYDRA)

## run:                Run all four steps for a single (dataset, classifier)  (DATA, NAME, SEED, CLASSIFIER)
run: prepare classify analyze render

## all:                Run the full pipeline for every dataset                (NAME, SEED, DISTANCE)
all:
	@for entry in $(DATASET_CLASSIFIERS); do \
		dataset=$${entry%%:*}; classifier=$${entry##*:}; \
		echo ""; \
		echo "══════════════════════════════════════════════"; \
		echo " Dataset: $$dataset  |  classifier=$$classifier  |  name=$(NAME)  seed=$(SEED)"; \
		echo "══════════════════════════════════════════════"; \
		$(MAKE) --no-print-directory run \
			DATA=$$dataset CLASSIFIER=$$classifier NAME=$(NAME) SEED=$(SEED) \
			DISTANCE=$(DISTANCE); \
	done
	@echo ""
	@echo "All datasets processed."

## generate:           Generate synthetic test dataset                        (ROWS)
generate:
	$(PYTHON) generate_synthetic.py $(if $(ROWS),--rows $(ROWS),)

## dashboard:          Open the experiment dashboard in browser
dashboard:
	venv/bin/streamlit run dashboard.py

## help:               Show this help message
help:
	@echo "Usage: make <target> [DATA=<dataset>] [NAME=<name>] [SEED=<n>] [CLASSIFIER=<name>] [DISTANCE=<dist>]"
	@echo ""
	@echo "Targets:"
	@grep -E '^## ' Makefile | sed 's/## /  /'
	@echo ""
	@echo "Defaults:  DATA=$(DATA)  NAME=$(NAME)  SEED=$(SEED)  CLASSIFIER=$(CLASSIFIER)  DISTANCE=$(DISTANCE)"
	@echo "ML classifiers:  $(ML_CLASSIFIERS)"
	@echo "DL classifiers:  $(DL_CLASSIFIERS)"
	@echo "Datasets:        $(DATASET_CLASSIFIERS)"
