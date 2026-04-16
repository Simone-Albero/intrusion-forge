#!/bin/bash

set -e

DATASETS=(
    "nb15_v2"
    "bot_iot_v2"
    "cic_2018_v2"
    "ton_iot_v2"
)

DISTANCE_METRICS=(
    "euclidean"
    "cosine"
)

NAME="failure_from_separability"
source venv/bin/activate

for dataset in "${DATASETS[@]}"; do
    for distance_metric in "${DISTANCE_METRICS[@]}"; do
        echo ">>> Running experiments on dataset: $dataset with distance metric: $distance_metric"
        # python3 prepare_data.py experiment=supervised data="$dataset" name="${NAME}_$distance_metric" distance_metric="$distance_metric"
        # python3 sup_classify.py experiment=supervised data="$dataset" name="${NAME}_$distance_metric" distance_metric="$distance_metric"
        # python3 run_inference.py experiment=supervised data="$dataset" name="${NAME}_$distance_metric" distance_metric="$distance_metric"
        python3 analyze_data.py experiment=supervised data="$dataset" name="${NAME}_$distance_metric" distance_metric="$distance_metric"
        echo ">>> Done: $dataset with distance metric: $distance_metric"
    done
done

echo "All datasets processed."
