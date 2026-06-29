#!/bin/bash

PREDS_PATH=$1
REWR_MODEL=$2
REWR_MODE=$3
MARKERS_PER_BIN=${4:-10}
SCORE=$5

datasets=(
    popqa
    sciq
    selfaware
    math
    simpleqa
    halueval
    umwp
    mmlu
    arc_challenge
    superglue
)

for dataset in "${datasets[@]}"; do
    echo "Processing ${PREDS_PATH}/test_preds_${dataset}.json"

    if [ "$SCORE" == "0" ]; then
        python ./exp3_rewriting/rewrite.py --markers_df_path=./exp3_rewriting/hedge_extraction/score_hedge_map.csv --preds_path=$PREDS_PATH/test_preds_${dataset}.json --rewrite_model_name=$REWR_MODEL --rewrite_mode=$REWR_MODE --max_markers_per_bin=$MARKERS_PER_BIN --output_modifier="_${REWR_MODE}_${MARKERS_PER_BIN}_${REWR_MODEL}"
    else
        python ./exp3_rewriting/rewrite.py --markers_df_path=./exp3_rewriting/hedge_extraction/score_hedge_map.csv --preds_path=$PREDS_PATH/test_preds_${dataset}.json --rewrite_model_name=$REWR_MODEL --rewrite_mode=$REWR_MODE --max_markers_per_bin=$MARKERS_PER_BIN --output_modifier="_${REWR_MODE}_${MARKERS_PER_BIN}_${REWR_MODEL}" --score_faithfulness
    fi
done
