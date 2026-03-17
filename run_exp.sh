#!/bin/bash

set -e

DATASETS=(
    "bot_iot_v2"
    "cic_2018_v2"
    "nb15_v2"
    "ton_iot_v2"
)

NAME="cluster_separability_euclidean"
source venv/bin/activate

for dataset in "${DATASETS[@]}"; do
    echo ">>> Running experiments on dataset: $dataset"
    # python3 prepare_data.py experiment=supervised data="$dataset" name="$NAME"
    # python3 sup_classify.py experiment=supervised data="$dataset" name="$NAME"
    # python3 run_inference.py experiment=supervised data="$dataset" name="$NAME"
    python3 analyze_data.py experiment=supervised data="$dataset" name="$NAME"
    echo ">>> Done: $dataset"
done

echo "All datasets processed."
