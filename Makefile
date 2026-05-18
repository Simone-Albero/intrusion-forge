# ──────────────────────────────────────────────────────────────────────────────
# Intrusion Forge — Experiment Runner
#
# Usage:
#   make prepare       DATA=cic_2018_v2 NAME=my_exp
#   make dl-classify   DATA=cic_2018_v2 NAME=my_exp
#   make ml-classify   DATA=cic_2018_v2 NAME=my_exp CLASSIFIER=random_forest
#   make ml-all        DATA=cic_2018_v2 NAME=my_exp   # every ML classifier
#   make analyze       DATA=cic_2018_v2 NAME=my_exp
#   make render        DATA=cic_2018_v2 NAME=my_exp
#   make run           DATA=cic_2018_v2 NAME=my_exp   # prepare + dl-classify + analyze + render
#   make all           NAME=my_exp                     # all datasets
# ──────────────────────────────────────────────────────────────────────────────

PYTHON     := venv/bin/python
EXPERIMENT ?= supervised
DATA       ?= cic_2018_v2
NAME       ?= exp
SEED       ?= 42
MODEL      ?= tabular_classifier
CLASSIFIER ?= random_forest
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

DATASET_MODELS := \
    nb15_v2:tabular_classifier \
    bot_iot_v2:tabular_classifier \
    cic_2018_v2:tabular_classifier \
    ton_iot_v2:tabular_classifier \
    bank_marketing:tabular_classifier \
    covertype:numerical_classifier \
    letter_recognition:numerical_classifier \
    statlog_landsat_satellite:numerical_classifier \
    thyroid_disease:numerical_classifier

HYDRA := experiment=$(EXPERIMENT) data=$(DATA) name=$(NAME) seed=$(SEED) \
         model=$(MODEL) \
         complexity.distance=$(DISTANCE) clustering.distance=$(DISTANCE)
HYDRA_ML := data=$(DATA) name=$(NAME) seed=$(SEED) classifier=$(CLASSIFIER) \
            complexity.distance=$(DISTANCE) clustering.distance=$(DISTANCE)
TB_LOGDIR  := resources/experiments/$(NAME)/$(DATA)_$(SEED)/tb

.PHONY: prepare dl-classify ml-classify ml-all analyze render run all generate tensorboard help

## prepare:            Step 1 — preprocess raw CSV → parquet splits           (DATA, NAME, SEED)
prepare:
	$(PYTHON) prepare_data.py $(HYDRA)

## dl-classify:        Step 2 — train & evaluate the DL classifier            (DATA, NAME, SEED, MODEL)
dl-classify:
	$(PYTHON) dl_classify.py $(HYDRA)

## ml-classify:        Step 2 — train & evaluate one ML classifier            (DATA, NAME, SEED, CLASSIFIER)
ml-classify:
	$(PYTHON) ml_classify.py $(HYDRA_ML)

## ml-all:             Step 2 — train & evaluate every ML classifier in turn  (DATA, NAME, SEED)
ml-all:
	@for clf in $(ML_CLASSIFIERS); do \
		echo ""; \
		echo "── ML classifier: $$clf ─────────────────────────────"; \
		$(MAKE) --no-print-directory ml-classify \
			DATA=$(DATA) NAME=$(NAME) SEED=$(SEED) CLASSIFIER=$$clf \
			DISTANCE=$(DISTANCE) || exit 1; \
	done

## analyze:            Step 3 — post-hoc analysis (compute only)              (DATA, NAME, SEED)
analyze:
	$(PYTHON) analyze_data.py $(HYDRA)

## render:             Step 4 — render plots from analysis artifacts          (DATA, NAME, SEED)
render:
	$(PYTHON) render_plots.py $(HYDRA)

## run:                Run all four steps for a single dataset (DL path)      (DATA, NAME, SEED)
run: prepare dl-classify analyze render

## all:                Run all four steps for every dataset in DATASET_MODELS (NAME, SEED, DISTANCE)
all:
	@for entry in $(DATASET_MODELS); do \
		dataset=$${entry%%:*}; model=$${entry##*:}; \
		echo ""; \
		echo "══════════════════════════════════════════════"; \
		echo " Dataset: $$dataset  |  model=$$model  |  name=$(NAME)  seed=$(SEED)"; \
		echo "══════════════════════════════════════════════"; \
		$(MAKE) --no-print-directory run \
			DATA=$$dataset MODEL=$$model NAME=$(NAME) SEED=$(SEED) EXPERIMENT=$(EXPERIMENT) \
			DISTANCE=$(DISTANCE); \
	done
	@echo ""
	@echo "All datasets processed."

## generate:           Generate synthetic test dataset                 (ROWS)
generate:
	$(PYTHON) generate_synthetic.py $(if $(ROWS),--rows $(ROWS),)

## dashboard:          Open the experiment dashboard in browser
dashboard:
	venv/bin/streamlit run dashboard.py

## tensorboard:        Open TensorBoard for the current experiment     (DATA, NAME, SEED)
tensorboard:
	venv/bin/tensorboard --logdir $(TB_LOGDIR)

## help:               Show this help message
help:
	@echo "Usage: make <target> [DATA=<dataset>] [NAME=<name>] [SEED=<n>] [EXPERIMENT=<exp>] [MODEL=<model>] [DISTANCE=<dist>]"
	@echo ""
	@echo "Targets:"
	@grep -E '^## ' Makefile | sed 's/## /  /'
	@echo ""
	@echo "Defaults:  DATA=$(DATA)  NAME=$(NAME)  SEED=$(SEED)  EXPERIMENT=$(EXPERIMENT)  MODEL=$(MODEL)  DISTANCE=$(DISTANCE)"
	@echo "Datasets:  $(DATASET_MODELS)"
	@echo "TensorBoard logdir:  $(TB_LOGDIR)"
