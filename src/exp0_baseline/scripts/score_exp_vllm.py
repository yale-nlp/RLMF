### Adapted from MetaFaith https://github.com/yale-nlp/MetaFaith/blob/main/src/score_exp_vllm.py
import argparse
import torch
import pandas as pd
import json
import ast 
import os
import time

import google.generativeai as genai

from src.exp0_baseline.prompts.input_prompts import *
from src.exp0_baseline.prompts.task_prompts import *
from src.exp0_baseline.utilities.utils import *
from src.exp0_baseline.metrics.faithfulness_batch import get_faithfulness_batch
from src.exp0_baseline.tasks._task import Task

from src.exp0_baseline.tasks import TASK_REGISTRY, MODEL_DEFAULTS


###################################
#### Set Up Gemini Access
###################################
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

###################################
#### Login to HF
###################################
from huggingface_hub import login
login(os.getenv("HF_TOKEN"))   # HF login


def parse_args():
    """
    Define and parse script arguments.
    """

    def add_arg(parser, arg, defaults, **kwargs):
        if defaults is None:
            default = None
        else:
            default = defaults.get(arg)
        parser.add_argument(f"{arg}", default=default, **kwargs)

    parser = argparse.ArgumentParser()

    ### Set model args
    add_arg(
        parser, 
        "--model_name", 
        MODEL_DEFAULTS, 
        type=str, 
        help="Name of model (e.g. in HF hub)"
    )
    parser.add_argument(
        "--dtype",
        default=torch.float16,
        help="Data type to use for model loading",
        choices = [
            torch.bfloat16,
            torch.float16,
        ]
    )
    parser.add_argument(
        "--trust_remote_code", 
        action='store_true',
        default=False,
        help="Whether to trust remote code in HF hub"
    )
    parser.add_argument(
        "--use_vllm", 
        action='store_true',
        default=False,
        help="Whether to use VLLM for inference for HF model"
    )
    parser.add_argument(
        "--gpu_mem_utilization",
        type=float,
        default=0.9,
        help="Proportion of GPU memory to use for VLLM model loading",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=float,
        default=1,
        help="Number of GPUs to use for VLLM model loading",
    )

    ### Set inference args
    parser.add_argument(
        "--start_idx", 
        type=int, 
        default=-1,
        help="Start index for inference"
    )
    parser.add_argument(
        "--model_max_len", 
        type=int, 
        default=4096,
        help="Max length of model input"
    )
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=256,
        help="Max tokens model can generate at inference time",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature for model inference",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top_p for model inference",
    )

    ### Set faithfulness/experiment args
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=20,
        help="Number of samplings to use when scoring uncertainty/confidence of the model",
    )
    parser.add_argument(
        "--sys_prompt",
        default=None,
        help="System prompt to use",
    )
    parser.add_argument(
        "--input_prompt",
        default=None,
        required=True,
        help="Lowercase string name of the task prompt to use; full list is in src/exp0_baseline/prompts/__init__.py",
    )
    parser.add_argument(
        "--hedge_prompt",
        default="blank",
        required=True,
        help="Lowercase string name of the hedge direction to use; full list is in src/exp0_baseline/prompts/__init__.py",
    )
    parser.add_argument(
        "--task_prompt",
        default="qa_short_answerability",
        required=True,
        help="Lowercase string name of the hedge prompt to use; full list is in src/exp0_baseline/prompts/__init__.py",
    )

    ### Set dataset args
    parser.add_argument(
        "--dataset_name",
        type=str,
        # nargs="+",
        required=True,
        choices=list(TASK_REGISTRY.keys()),
        help="Name of the dataset/task to use, full list is in src/exp0_baseline/tasks/__init__.py",
    )
    parser.add_argument(
        "--num_samples",
        default=None,
        type=int,
        help="Max number of instances to run inference on",
    )
    parser.add_argument(
        "--random_seed",
        default=42,
        type=int,
        help="Seed for sampling when `num_samples` is smaller than dataset size",
    )
    
    ### Set saving/output args
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Directory for output files"
    )
    parser.add_argument(
        "--score_only", 
        action='store_true',
        default=False,
        help="Score existing saved results."
    )

    args = parser.parse_args()
    return args

def get_identifier(args):
    """
    Get identifying string for the present experiment to use in saving.
    """

    num_samples, dataset_name, input_prompt, task_prompt, hedge_prompt = args.num_samples, args.dataset_name, args.input_prompt, args.task_prompt.lower(), args.hedge_prompt.lower()

    task_identifier = f"{dataset_name}__{input_prompt}__{task_prompt}__{hedge_prompt}__{num_samples}_samps"

    if args.sys_prompt:
        task_identifier += f"__fprompt_{args.sys_prompt}"

    return task_identifier

def get_model_identifier(args):

    model_identifier = args.model_name.replace("-", "_").replace("/", "_")

    if args.temperature:
        model_identifier += f"__temp_{args.temperature}"
    if args.top_p:
        model_identifier += f"__temp_{args.top_p}"

    return model_identifier

def run(args):
    """
    Run full inference experiment and save predictions and scoring results.
    """

    start_time = time.time()

    # Prepare output directories
    output_dir = args.output_dir
    candidate_str = f"{args.num_candidates}_cands"
    task_identifier = get_identifier(args)
    model_identifier = get_model_identifier(args)
    task_dir = os.path.join(output_dir, task_identifier)
    save_dir = os.path.join(output_dir, task_identifier, candidate_str, model_identifier)

    print("\n\n"+"#"*50)
    print(f"Starting scoring for experiment {task_identifier} for {model_identifier}...")

    # Load task
    if args.dataset_name not in TASK_REGISTRY:               
        raise ValueError("Invalid dataset/task provided")

    task: Task = TASK_REGISTRY[args.dataset_name](args=args)
    
    # Get data samples for inference
    inputs_file = os.path.join(task_dir, "formatted_inputs.csv")   # if existent
    data_df = pd.read_csv(inputs_file, index_col=0)
    data_df["input_args"] = data_df["input_args"].apply(ast.literal_eval)
    data_df["inputs"] = data_df["inputs"].apply(ast.literal_eval)
    if task.task_type=="qa":
        data_df["targets"] = data_df["targets"].apply(ast.literal_eval)
    task.set_data_df(data_df)

    # Open results files
    results_file = os.path.join(save_dir, "results.csv")
    sampled_outputs_file = os.path.join(save_dir, "outputs_sampled.json")
    assertions_file = os.path.join(save_dir, "assertions.json")
    metrics_file = os.path.join(save_dir, "metrics.json")
    uncertainty_outputs_file = os.path.join(save_dir, "conf_scores.json")

    results_df = pd.read_csv(results_file, index_col=0)
    try:
        results_df["targets"] = results_df["targets"].apply(ast.literal_eval)
    except: 
        pass
    # results_df["outputs"] = results_df["outputs"].apply(pd.to_numeric, errors='ignore') 
    def safe_to_numeric(x):
        try:
            return pd.to_numeric(x)
        except:
            return x
    results_df["outputs"] = results_df["outputs"].apply(safe_to_numeric)
    with open(sampled_outputs_file) as f:
        sampled_outputs_dict = json.load(f) 
    with open(metrics_file) as f:
        metrics = json.load(f) 

    if not args.score_only:

        faithfulness_scores, avg_conf_scores, overall_decisiveness_scores, assertions_dict, conf_responses = get_faithfulness_batch(
            answers=results_df["outputs"], 
            sampled_answers_dict=sampled_outputs_dict, 
        ) 

        results_df['faithfulness'] = faithfulness_scores
        results_df['confidence'] = avg_conf_scores
        results_df['decisiveness'] = overall_decisiveness_scores

        # Save results
        results_df = results_df[['targets', 'outputs', 'faithfulness', 'confidence', 'decisiveness']]
        results_df.to_csv(results_file)
        with open(assertions_file, 'w') as file:
            json.dump(assertions_dict, file, indent=4)
        with open(uncertainty_outputs_file, 'w') as file:
            json.dump(conf_responses, file, indent=4)


    # Evaluate task performance, cMFG, MFG
    print(f"\nComputing performance and faithfulness metrics...")
    try:
        metrics_dict, errors = task.score(results_df['outputs'])
        metrics.update(metrics_dict)
    except: 
        errors = np.zeros(results_df.shape[0])
    results_df['errors'] = errors

    cmfg, mfg, stats = get_cmfg_mfg(
        results_df['faithfulness'].tolist(), 
        results_df['confidence'].tolist(), 
        num_bins=10,
    )
    cmfg_star, var_cmfg_star, star_stats = get_cmfg_star(
        results_df['faithfulness'].tolist(), 
        results_df['confidence'].tolist(), 
        num_bins=10,
    )
    
    metrics['cmfg'] = cmfg
    metrics['mfg'] = mfg
    metrics['stats'] = stats
    metrics['cmfg_star'] = cmfg_star
    metrics['var_cmfg_star'] = var_cmfg_star
    # metrics['star_stats'] = star_stats
    
    # Get runtime
    processing_time = time.time() - start_time  
    print(f"\nCompleted scoring in {processing_time:.2f} seconds")
    metrics['scoring_runtime_seconds'] = processing_time

    # Save results
    with open(metrics_file, 'w') as file:
        json.dump(metrics, file, indent=4)
    if not args.score_only:
        results_df.to_csv(results_file)
        with open(assertions_file, 'w') as file:
            json.dump(assertions_dict, file, indent=4)
    print(f"\nFinished saving assertions, metrics to\n{save_dir}!\n")

    # Score performance
    if not args.score_only:
        start_time = time.time()

        results_df["accuracy"] = results_df.apply(lambda row: llm_eval(str(row["targets"]), str(row["outputs"])), axis=1)

        # Save results
        results_df.to_csv(os.path.join(save_dir, "results.csv"))

        # Get runtime
        processing_time = time.time() - start_time  
        print(f"\nCompleted LLM scoring of performance in {processing_time:.2f} seconds!")
 
if __name__ == "__main__":
    args = parse_args()
    run(args)




