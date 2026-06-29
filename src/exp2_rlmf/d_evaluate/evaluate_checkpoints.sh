#!/usr/bin/env bash
HOSTNUM="$1"
MODEL_DIR="$2"
START_CHECKPOINT="${3:- 550}"  
END_CHECKPOINT="${4:- 1500}"   
INCREMENT="${5:- 50}"  

# Wait until MODEL_DIR exists
echo "Checking for MODEL_DIR ${MODEL_DIR} ..."
while [[ ! -d "${MODEL_DIR}" ]]; do
    echo "[$(date)] ${MODEL_DIR} does not exist yet; sleeping for 10 minutes..."
    sleep 600
done
echo "Found ${MODEL_DIR} — proceeding."

# Get Latest Config by Timestamp in Filename
CONFIG_PATH="$(ls -1 "${MODEL_DIR}"/config_*.json | sort -V | tail -n 1)"
if [[ -z "$CONFIG_PATH" ]]; then
    echo "ERROR: No config_*.json found in ${MODEL_DIR}"
    exit 1
fi
echo "Using config: ${CONFIG_PATH}"

for ((CHECKPOINT_NUM=START_CHECKPOINT; CHECKPOINT_NUM<=END_CHECKPOINT; CHECKPOINT_NUM+=INCREMENT)); do

    CHECKPOINT_DIR="${MODEL_DIR}/checkpoint-${CHECKPOINT_NUM}"

    # Wait until checkpoint >=550 exists
    echo "Checking for ${CHECKPOINT_DIR}..."
    while [[ ! -d "$CHECKPOINT_DIR" ]]; do
        echo "[$(date)] ${CHECKPOINT_DIR} not found yet; sleeping 10 minutes..."
        sleep 600
    done

    echo "Running inference for model ${MODEL_DIR} checkpoint ${CHECKPOINT_NUM}"

    python ./exp2_rlmf/d_evaluate/inference.py  --config_path=$CONFIG_PATH  --with_instruction  --hostnum=$HOSTNUM  --checkpoint=$CHECKPOINT_NUM

done