#!/usr/bin/env bash
HOSTNUM="$1"
MODEL_DIR="$2"
CHECKPOINT_NUM="$3"

# Get Latest Config by Timestamp in Filename
CONFIG_PATH="$(ls -1 "${MODEL_DIR}"/config_*.json | sort -V | tail -n 1)"
if [[ "$MODEL_PATH" == "0" && -z "$CONFIG_PATH" ]]; then
    echo "ERROR: No config_*.json found in ${MODEL_DIR}"
    exit 1
fi
echo "Using config: ${CONFIG_PATH}"

datasets=(
    popqa
    selfaware
    math
    sciq
    simpleqa
    halueval
    umwp
    mmlu
    arc_challenge
    superglue
)
echo "Running all datasets"

for dataset in "${datasets[@]}"; do

    echo "Running inference for model ${MODEL_DIR} checkpoint ${CHECKPOINT_NUM} dataset ${dataset}"

    python ./exp2_rlmf/d_evaluate/inference.py  --config_path=$CONFIG_PATH  --with_instruction  --hostnum=$HOSTNUM  --checkpoint=$CHECKPOINT_NUM  --dataset_name=$dataset
    
done