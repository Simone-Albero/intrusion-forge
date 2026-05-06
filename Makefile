# ──────────────────────────────────────────────────────────────────────────────
# Intrusion Forge — Experiment Runner
#
# Usage:
#   make prepare  DATA=cic_2018_v2 NAME=my_exp
#   make classify DATA=cic_2018_v2 NAME=my_exp
#   make analyze  DATA=cic_2018_v2 NAME=my_exp
#   make run      DATA=cic_2018_v2 NAME=my_exp    # all three phases
#   make all      NAME=my_exp                      # all datasets
# ──────────────────────────────────────────────────────────────────────────────

PYTHON     := venv/bin/python
EXPERIMENT ?= supervised
DATA       ?= cic_2018_v2
NAME       ?= exp
SEED       ?= 42

DATASETS := nb15_v2 bot_iot_v2 cic_2018_v2 ton_iot_v2

HYDRA := experiment=$(EXPERIMENT) data=$(DATA) name=$(NAME) seed=$(SEED)
TB_LOGDIR  := resources/experiments/$(NAME)/$(DATA)_$(SEED)/tb

.PHONY: prepare classify analyze run all generate tensorboard help

## prepare:            Step 1 — preprocess raw CSV → parquet splits  (DATA, NAME, SEED)
prepare:
	$(PYTHON) prepare_data.py $(HYDRA)

## classify:           Step 2 — train & evaluate classifier           (DATA, NAME, SEED)
classify:
	$(PYTHON) classify.py $(HYDRA)

## analyze:            Step 3 — post-hoc analysis                     (DATA, NAME, SEED)
analyze:
	$(PYTHON) analyze_data.py $(HYDRA)

## run:                Run all three steps for a single dataset        (DATA, NAME, SEED)
run: prepare classify analyze

## all:                Run all three steps for every dataset in DATASETS (NAME, SEED)
all:
	@for dataset in $(DATASETS); do \
		echo ""; \
		echo "══════════════════════════════════════════════"; \
		echo " Dataset: $$dataset  |  name=$(NAME)  seed=$(SEED)"; \
		echo "══════════════════════════════════════════════"; \
		$(MAKE) --no-print-directory run \
			DATA=$$dataset NAME=$(NAME) SEED=$(SEED) EXPERIMENT=$(EXPERIMENT); \
	done
	@echo ""
	@echo "All datasets processed."

## generate:           Generate synthetic test dataset                 (ROWS)
generate:
	$(PYTHON) generate_synthetic.py $(if $(ROWS),--rows $(ROWS),)

## tensorboard:        Open TensorBoard for the current experiment     (DATA, NAME, SEED)
tensorboard:
	venv/bin/tensorboard --logdir $(TB_LOGDIR)

## help:               Show this help message
help:
	@echo "Usage: make <target> [DATA=<dataset>] [NAME=<name>] [SEED=<n>] [EXPERIMENT=<exp>]"
	@echo ""
	@echo "Targets:"
	@grep -E '^## ' Makefile | sed 's/## /  /'
	@echo ""
	@echo "Defaults:  DATA=$(DATA)  NAME=$(NAME)  SEED=$(SEED)  EXPERIMENT=$(EXPERIMENT)"
	@echo "Datasets:  $(DATASETS)"
	@echo "TensorBoard logdir:  $(TB_LOGDIR)"
