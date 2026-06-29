#!/bin/bash

MODEL_NAME=$1
NUMSAMPS=${2:-1000}
SCORE_MODE=${3:-0}
HOSTNUM=${4:-0}

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

for dataset in "${datasets[@]}"; do

    if [[ "$SCORE_MODE" -eq 2 ]]; then
        python ./exp1_metafaith/inference_and_score.py --model_name=$MODEL_NAME --dataset_name=$dataset  --gpu_mem_utilization=0.9  --tensor_parallel_size=1  --num_samples=$NUMSAMPS  --split=train  --max_output_tokens=512 --sys_prompt=sys0  --score_mode=$SCORE_MODE --hostnum=$HOSTNUM    # use 256 tokens if no reasoning used
    else 
        python ./exp1_metafaith/inference_and_score.py --model_name=$MODEL_NAME --dataset_name=$dataset  --gpu_mem_utilization=0.9  --tensor_parallel_size=1  --num_samples=$NUMSAMPS  --split=train  --max_output_tokens=512 --sys_prompt=sys0  --score_mode=$SCORE_MODE  # use 256 tokens if no reasoning used
    fi

done

