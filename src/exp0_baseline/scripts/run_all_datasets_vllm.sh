#!/bin/bash

# Command-line arguments
MODEL_NAME=$1
HEDGE_PROMPT=$2
MAX_TOKENS=${3:-256}
GMU=${4:-0.9}
TPS=${5:-1}

### QA Datasets

python ./exp0_baseline/scripts/run_exp_vllm.py --model_name="$MODEL_NAME" --use_vllm  --dataset_name="popqa"  --num_samples=1000  --num_candidates=20  --hedge_prompt="$HEDGE_PROMPT"  --input_prompt="qa"  --task_prompt="qa_short"  --output_dir="./exp0_baseline/results"  --tensor_parallel_size=$TPS --gpu_mem_utilization=$GMU --max_output_tokens=$MAX_TOKENS

python ./exp0_baseline/scripts/run_exp_vllm.py --model_name="$MODEL_NAME" --use_vllm  --dataset_name="selfaware"  --num_samples=1000  --num_candidates=20  --hedge_prompt="$HEDGE_PROMPT"  --input_prompt="qa"  --task_prompt="qa_short"  --output_dir="./exp0_baseline/results"  --tensor_parallel_size=$TPS --gpu_mem_utilization=$GMU --max_output_tokens=$MAX_TOKENS

python ./exp0_baseline/scripts/run_exp_vllm.py --model_name="$MODEL_NAME"  --use_vllm  --dataset_name="simpleqa"  --num_samples=1000  --num_candidates=20  --hedge_prompt="$HEDGE_PROMPT"  --input_prompt="qa"  --task_prompt="qa_short"  --output_dir="./exp0_baseline/results" --tensor_parallel_size=$TPS --gpu_mem_utilization=$GMU --max_output_tokens=$MAX_TOKENS

### Hallucination Detection

python ./exp0_baseline/scripts/run_exp_vllm.py --model_name="$MODEL_NAME" --use_vllm  --dataset_name="halueval"  --num_samples=1000  --num_candidates=20  --hedge_prompt="$HEDGE_PROMPT"  --input_prompt="hd"  --task_prompt="hd" --output_dir="./exp0_baseline/results"  --tensor_parallel_size=$TPS --gpu_mem_utilization=$GMU  --max_output_tokens=$MAX_TOKENS

### Mathematics & STEM Challenges

python ./exp0_baseline/scripts/run_exp_vllm.py --model_name="$MODEL_NAME"  --use_vllm  --dataset_name="math"  --num_samples=1000  --num_candidates=20  --hedge_prompt="$HEDGE_PROMPT"  --input_prompt="qa"  --task_prompt="math" --output_dir="./exp0_baseline/results" --tensor_parallel_size=$TPS --gpu_mem_utilization=$GMU --max_output_tokens=$MAX_TOKENS

python ./exp0_baseline/scripts/run_exp_vllm.py --model_name="$MODEL_NAME" --use_vllm  --dataset_name="umwp"  --num_samples=1000  --num_candidates=20  --hedge_prompt="$HEDGE_PROMPT"  --input_prompt="qa"  --task_prompt="umwp" --output_dir="./exp0_baseline/results"  --tensor_parallel_size=$TPS --gpu_mem_utilization=$GMU  --max_output_tokens=$MAX_TOKENS

python ./exp0_baseline/scripts/run_exp_vllm.py --model_name="$MODEL_NAME" --use_vllm  --dataset_name="sciq"  --num_samples=1000  --num_candidates=20  --hedge_prompt="$HEDGE_PROMPT"  --input_prompt="mcq"  --task_prompt="mcq_unique" --output_dir="./exp0_baseline/results"  --tensor_parallel_size=$TPS --gpu_mem_utilization=$GMU  --max_output_tokens=$MAX_TOKENS

python ./exp0_baseline/scripts/run_exp_vllm.py --model_name="$MODEL_NAME" --use_vllm  --dataset_name="arc_challenge"  --num_samples=1000  --num_candidates=20  --hedge_prompt="$HEDGE_PROMPT"  --input_prompt="mcq"  --task_prompt="mcq_unique_letters" --output_dir="./exp0_baseline/results"  --tensor_parallel_size=$TPS --gpu_mem_utilization=$GMU  --max_output_tokens=$MAX_TOKENS

### General Task Suites

python ./exp0_baseline/scripts/run_exp_vllm.py --model_name="$MODEL_NAME" --use_vllm  --dataset_name="mmlu"  --num_samples=1000  --num_candidates=20  --hedge_prompt="$HEDGE_PROMPT"  --input_prompt="mcq"  --task_prompt="mcq_unique" --output_dir="./exp0_baseline/results"  --tensor_parallel_size=$TPS --gpu_mem_utilization=$GMU  --max_output_tokens=$MAX_TOKENS

python ./exp0_baseline/scripts/run_exp_vllm.py --model_name="$MODEL_NAME" --use_vllm  --dataset_name="superglue"  --num_samples=1000  --num_candidates=20  --hedge_prompt="$HEDGE_PROMPT"  --input_prompt="qa"  --task_prompt="superglue" --output_dir="./exp0_baseline/results" --tensor_parallel_size=$TPS --gpu_mem_utilization=$GMU  --max_output_tokens=$MAX_TOKENS
