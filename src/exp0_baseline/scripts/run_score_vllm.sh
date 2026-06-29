#!/bin/bash

dataset_name=$1
hedge_names=(basic blank genuine human perception)
model_names=(
    # meta-llama/Llama-3.1-8B-Instruct
    Qwen/Qwen3-0.6B
    Qwen/Qwen3-1.7B
    Qwen/Qwen3-4B-Instruct-2507
    Qwen/Qwen3-8B
)

# Loop through each hedge_name
for hedge_name in "${hedge_names[@]}"; do
    for model_name in "${model_names[@]}"; do
        bash ./exp0_baseline/scripts/score_cmd_vllm.sh  $dataset_name  $hedge_name  $model_name
    done
done
