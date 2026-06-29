#!/bin/bash

# Experiment params passed as arguments
dataset_name=$1
hedge_name=$2
model_name=$3

# Python function to generate model identifier (modelname)
get_model_identifier() {
    model_name=$1
    temperature=$2
    top_p=$3

    model_identifier=$(echo "$model_name" | sed 's/-/_/g' | sed 's/\//_/g')

    if [[ "$temperature" != "null" ]]; then
        model_identifier="${model_identifier}__temp_${temperature}"
    fi

    if [[ "$top_p" != "null" ]]; then
        model_identifier="${model_identifier}__top_p_${top_p}"
    fi

    echo "$model_identifier"
}

# Function to check if metrics.json contains the key predxn_runtime_seconds and does not contain scoring_runtime_seconds
check_metrics_file() {
    metrics_file=$1
    if [[ -f "$metrics_file" && $(jq 'has("predxn_runtime_seconds")' "$metrics_file") == "true" && $(jq 'has("scoring_runtime_seconds")' "$metrics_file") == "false" ]]; then
        echo 0  # Return 0 if predxn_runtime_seconds exists and scoring_runtime_seconds does not exist => start scoring
    elif [[ -f "$metrics_file" && $(jq 'has("predxn_runtime_seconds")' "$metrics_file") == "true" && $(jq 'has("scoring_runtime_seconds")' "$metrics_file") == "true" ]]; then
        echo 2  # Return 2 if scoring_runtime_seconds exists => skip
    else
        echo 1  # Return 1 if the file doesn't meet the conditions => wait for file
    fi
}

# Get model identifier using the passed model name
model_identifier=$(get_model_identifier "$model_name" "null" "null")

# Loop through each folder path
declare -A dataset_map=(
    ["arc_challenge"]="./exp0_baseline/results/arc_challenge__mcq__mcq_unique_letters__${hedge_name}__1000_samps/20_cands/${model_identifier}"
    ["halueval"]="./exp0_baseline/results/halueval__hd__hd__${hedge_name}__1000_samps/20_cands/${model_identifier}"
    ["math"]="./exp0_baseline/results/math__qa__math__${hedge_name}__1000_samps/20_cands/${model_identifier}"
    ["mmlu"]="./exp0_baseline/results/mmlu__mcq__mcq_unique__${hedge_name}__1000_samps/20_cands/${model_identifier}"
    ["popqa"]="./exp0_baseline/results/popqa__qa__qa_short__${hedge_name}__1000_samps/20_cands/${model_identifier}"
    ["sciq"]="./exp0_baseline/results/sciq__mcq__mcq_unique__${hedge_name}__1000_samps/20_cands/${model_identifier}"
    ["selfaware"]="./exp0_baseline/results/selfaware__qa__qa_short__${hedge_name}__1000_samps/20_cands/${model_identifier}"
    ["simpleqa"]="./exp0_baseline/results/simpleqa__qa__qa_short__${hedge_name}__1000_samps/20_cands/${model_identifier}"
    ["superglue"]="./exp0_baseline/results/superglue__qa__superglue__${hedge_name}__1000_samps/20_cands/${model_identifier}"
    ["umwp"]="./exp0_baseline/results/umwp__qa__umwp__${hedge_name}__1000_samps/20_cands/${model_identifier}"
)

results_path="${dataset_map[$dataset_name]#*:}"

# Define paths for args.json and metrics.json, including model name in paths
args_file="$results_path/args.json"
metrics_file="$results_path/metrics.json"

# Check if check_metrics_file returns 2 (skip the folder)
if [[ $(check_metrics_file "$metrics_file") -eq 2 ]]; then
    echo -e "Skipping folder $results_path because scoring_runtime_seconds exists in metrics.json.\n"
else
    # Continuously wait for the necessary files to exist and meet the condition
    while [[ ! -f "$args_file" || ! -f "$metrics_file" || $(check_metrics_file "$metrics_file") -eq 1 ]]; do
        echo -e "Waiting for $metrics_file to be ready...\n"
        sleep 600  # Check every 10 seconds
    done

    echo -e "Predictions completed for $results_path, proceeding to score.\n"

    # Start the command
    cmd="python ./exp0_baseline/scripts/score_exp_vllm.py "

    # Add the arguments from args.json, excluding dtype
    for key in $(jq -r 'keys_unsorted[]' "$args_file"); do
        value=$(jq -r --arg key "$key" '.[$key]' "$args_file")
        
        # Skip dtype
        if [[ "$key" == "dtype" ]]; then
            continue
        fi

        # Replace hyphens with underscores in keys
        key=$(echo "$key" | tr '-' '_')

        # Add the argument to the command
        if [[ "$value" == "true" ]]; then
            cmd="$cmd --$key"
        elif [[ "$value" == "false" ]]; then
            continue  # Skip false boolean flags
        elif [[ "$value" != "null" ]]; then
            cmd="$cmd --$key $value"
        fi
    done

    # Print and run the command
    echo -e "Running: $cmd\n"
    $cmd
fi


