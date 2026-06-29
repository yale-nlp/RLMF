#!/bin/bash

MODEL_NAME=$1
NUMSAMPS=${2:-1000}

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

    python ./exp1_metafaith/inference_and_score.py --model_name=$MODEL_NAME --dataset_name=$dataset  --gpu_mem_utilization=0.9  --tensor_parallel_size=1  --num_samples=$NUMSAMPS  --split=train  --max_output_tokens=512 --sys_prompt=sys0  --no_score # use 256 tokens if no reasoning used

done

