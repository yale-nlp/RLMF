import os
import json
import numpy as np
from termcolor import colored
import hashlib

import torch
import gc 
from vllm import LLM
from vllm.lora.request import LoRARequest

from src.exp0_baseline.utilities.utils import get_cmfg_mfg, get_cmfg_star
from src.exp2_rlmf.utils.data_utils import get_dataset
from src.exp2_rlmf.utils.utils import load_config, extract_sentences_with_confidence_new, get_sentence_internal_confidence, llm_eval
from src.exp2_rlmf.c_rl_training.rewards import linear_reward, quadratic_reward, binary_reward, log_reward, stretched_log_reward

from argparse import ArgumentParser

dtypes = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16
}

def run_extract_sentences_with_confidence(extracted_responses, format_version, get_meta_score_from_completions=False, metascore_as_percentage=False):

    sentences_per_completion = []   # List of list of strs
    pred_confs_per_completion = []  # List of list of floats
    metascore_per_completion = []
    for response in extracted_responses:
        extracted_sentences_with_confidence = extract_sentences_with_confidence_new(response, percentage=('percentage' in format_version), get_meta_score=get_meta_score_from_completions, metascore_as_percentage=metascore_as_percentage)
        sentences, confidences = extracted_sentences_with_confidence[0], extracted_sentences_with_confidence[1]
        if get_meta_score_from_completions:
            metascore = extracted_sentences_with_confidence[2]
        else: 
            metascore = None
        if len(sentences)==0:
            # print(colored("Warning: No sentences found in the completion:", "red"), f"{completion}")
            sentences_per_completion.append(['no output'])
            pred_confs_per_completion.append([None])
            metascore_per_completion.append([None])
        else:
            sentences_per_completion.append(sentences)
            pred_confs_per_completion.append(confidences)
            metascore_per_completion.append(metascore)
    return sentences_per_completion, pred_confs_per_completion, metascore_per_completion

def get_metrics(extracted_responses, sampled_answers_lists, prompts, targets, format_version, reward_weights, hostnum, get_meta_score_from_completions=False, get_mc_prediction_as_percentage=False):

    ### Extract Sentences & Predicted Confidence Floats
    sentences_per_response, pred_confs_per_response, metascore_per_response = run_extract_sentences_with_confidence(extracted_responses, format_version=format_version, get_meta_score_from_completions=get_meta_score_from_completions, metascore_as_percentage=get_mc_prediction_as_percentage)
    assert(len(sentences_per_response)==len(pred_confs_per_response))
        
    ### Compute Gold Confidence
    assert(len(sentences_per_response)==len(sampled_answers_lists))
    # if hostnum==8001:
    #     import ipdb; ipdb.set_trace()
    gold_confs_per_response = get_sentence_internal_confidence(
            sent_lists=sentences_per_response, 
            sampled_answers_lists=sampled_answers_lists,
            hostname=f"http://localhost:{hostnum}/v1",
        )   # List of list of floats

    ### Compute Faithfulness Rewards + Overall Faithfulness
    faithfulness_rewards = {
        'linear': [],
        'binary': [],
        'quadratic': [],
        'simple_log': [],
        'stretched_log': [],
    }
    faithfulness_scores = []
    for sentences, pred_confs, gold_confs in zip(sentences_per_response, pred_confs_per_response, gold_confs_per_response):
        assert(len(sentences)==len(pred_confs))
        assert(len(sentences)==len(gold_confs))

        ### Compute Faithfulness Rewards
        faithfulness_rewards['linear'].append(-10.*reward_weights[-1] if (len(sentences)==1 and sentences[0]=='no output') else sum([linear_reward(c, ic)*reward_weights[-1] for c, ic in zip(pred_confs, gold_confs)]) / len(sentences))
        faithfulness_rewards['quadratic'].append(-10.*reward_weights[-1] if (len(sentences)==1 and sentences[0]=='no output') else sum([quadratic_reward(c, ic)*reward_weights[-1] for c, ic in zip(pred_confs, gold_confs)]) / len(sentences))
        faithfulness_rewards['binary'].append(-10.*reward_weights[-1] if (len(sentences)==1 and sentences[0]=='no output') else sum([binary_reward(c, ic)*reward_weights[-1] for c, ic in zip(pred_confs, gold_confs)]) / len(sentences))
        faithfulness_rewards['simple_log'].append(-10.*reward_weights[-1] if (len(sentences)==1 and sentences[0]=='no output') else sum([log_reward(c, ic)*reward_weights[-1] for c, ic in zip(pred_confs, gold_confs)]) / len(sentences))
        faithfulness_rewards['stretched_log'].append(-10.*reward_weights[-1] if (len(sentences)==1 and sentences[0]=='no output') else sum([stretched_log_reward(c, ic)*reward_weights[-1] for c, ic in zip(pred_confs, gold_confs)]) / len(sentences))

        ### Compute Faithfulness Score for cMFG/cMFG*
        # skip indices where len(sentences)==1 and sentences[0]=='no output'
        # skip indices where pred confs pred_conf is None or not (0.<=pred_conf<=1.)
        # skip indices where gold conf is -1
        pred_confs_np = np.array(pred_confs)
        gold_confs_np = np.array(gold_confs)

        no_output = (len(sentences) == 1 and sentences[0] == "no output")
        pred_valid = np.array([(p is not None and 0 <= p <= 1) for p in pred_confs_np], bool)
        gold_valid = (gold_confs_np != -1)
        if no_output:
            mask = np.zeros_like(gold_valid)
        else:
            mask = pred_valid & gold_valid
        
        if len(gold_confs_np[mask])>0:
            f_score = 1. - (1. / len(sentences)) * np.abs(pred_confs_np[mask].astype(float) - gold_confs_np[mask]).sum()
        else: 
            f_score = -1.
        faithfulness_scores.append(f_score)
    
    assert(len(faithfulness_scores)==len(faithfulness_rewards['linear']))

    responses_without_confidences = [
        " ".join(sent_list) for sent_list in sentences_per_response
    ]
    correctness = llm_eval(prompts, targets, responses_without_confidences, hostname=f"http://localhost:{hostnum}/v1") # 1.0 if correct 0.0 else; proxy eval by local model
    avg_confidences = [
        -1. if not (vals := [c for c in conf_list if c is not None]) else np.mean(vals)
        for conf_list in pred_confs_per_response
    ]
    avg_gold_confidences = [
        -1. if not (vals := [c for c in conf_list if c is not None]) else np.mean(vals)
        for conf_list in gold_confs_per_response
    ]

    all_correctness = np.array(correctness, dtype=np.float64)
    all_avg_confidences = np.array(avg_confidences, dtype=np.float64)
    all_avg_gold_confidences = np.array(avg_gold_confidences, dtype=np.float64)
    brier_scores = (all_correctness - all_avg_gold_confidences) ** 2.
    mask = all_avg_gold_confidences == -1
    brier_scores[mask] = -1.

    ### Compute cMFG/cMFG* & Avg Acc & BS
    cmfg, mfg, stats = get_cmfg_mfg(
        faithfulness_scores, 
        all_avg_gold_confidences, 
        num_bins=10,
    )
    cmfg_star, var_cmfg_star, star_stats = get_cmfg_star(
        faithfulness_scores, 
        all_avg_gold_confidences, 
        num_bins=10,
    )
    metrics = {
        'cmfg_numeric': cmfg,
        'mfg_numeric': mfg,
        'stats_numeric': stats,
        'cmfg_star_numeric': cmfg_star,
        'var_cmfg_star_numeric': var_cmfg_star,
        # 'star_stats_numeric': star_stats,
        'avg_acc': all_correctness.mean(),
        'avg_bs': brier_scores[brier_scores!=-1].mean(),
        'avg_lin_reward': np.array(faithfulness_rewards['linear'])[np.array(faithfulness_rewards['linear'])!=-1].mean(),
        'avg_quad_reward': np.array(faithfulness_rewards['quadratic'])[np.array(faithfulness_rewards['quadratic'])!=-1].mean(),
        'avg_bin_reward': np.array(faithfulness_rewards['binary'])[np.array(faithfulness_rewards['binary'])!=-1].mean(),
        'avg_simp_log_reward': np.array(faithfulness_rewards['simple_log'])[np.array(faithfulness_rewards['simple_log'])!=-1].mean(),
        'avg_stret_log_reward': np.array(faithfulness_rewards['stretched_log'])[np.array(faithfulness_rewards['stretched_log'])!=-1].mean(),
        'acc_scores': correctness,
        'brier_scores': brier_scores.tolist(),
        'f_scores': faithfulness_scores,
        'avg_pred_confs': all_avg_confidences.tolist(),
        'avg_gold_confs': all_avg_gold_confidences.tolist(),
        'pred_confs_per_response': pred_confs_per_response,
        'gold_confs_per_response': gold_confs_per_response,
        'metascore_per_response': metascore_per_response,
    }
    print(metrics)

    return faithfulness_rewards, metrics


def inference(args):

    ### Load Config
    with open(args.config_path, 'r', encoding='utf-8') as file:
        config_path = json.load(file)['config_path']
    config = load_config(config_path)

    ### Check Run Condition
    dtype = dtypes[config.dtype]
    model_identifier = config.model_name.replace("/", "-").replace("-", "_")
    run_name = config.run_name if config.run_name else model_identifier
    output_dir = os.path.join(config.output_dir, model_identifier, run_name)
    if args.checkpoint_num==None:
        save_path = os.path.join(output_dir, f'test_preds{args.output_modifier}.json')
        metrics_path = os.path.join(output_dir, f'test_scores{args.output_modifier}.json')
        rewards_path = os.path.join(output_dir, f'test_rewards{args.output_modifier}.json')
    else: 
        if args.dev_mode:   
            results_dir = 'test_results_dev'
        else:
            results_dir = 'test_results'
        output_dir2 = os.path.join(output_dir, results_dir, f'checkpoint_{args.checkpoint_num}')
        os.makedirs(output_dir2, exist_ok=True)
        if args.use_length_direction:
            output_dir2 = os.path.join(output_dir2, "_with_length_dirxn")
        os.makedirs(output_dir2, exist_ok=True)
        if args.dataset_name==None:
            ds_str = "_".join(config.dataset_names)
        else: 
            ds_str = str(args.dataset_name)
        save_path = os.path.join(output_dir2, f'test_preds{args.output_modifier}_{ds_str}.json')
        metrics_path = os.path.join(output_dir2, f'test_scores{args.output_modifier}_{ds_str}.json')
        rewards_path = os.path.join(output_dir2, f'test_rewards{args.output_modifier}_{ds_str}.json')
    if os.path.exists(save_path) and os.path.exists(metrics_path) and os.path.exists(rewards_path):
        print(colored(f"Running {config.run_name} checkpoint {args.checkpoint_num} on dataset {ds_str}...", "cyan"))
        print(colored("Already computed and saved faithfulness scores, skipping this dataset...", "cyan"))
        return
    else:
        print(colored(f"Running {config.run_name} checkpoint {args.checkpoint_num} on dataset {ds_str}...", "cyan"))
    
    from vllm import SamplingParams

    print(colored("Loading original model to VLLM...", "cyan"))
    if 'emma' in config.model_name:
        llm = LLM(
            model=config.model_name, 
            dtype=dtype, 
            max_model_len=config.max_seq_length + 4096, 
            enable_lora=True,
            max_lora_rank=config.lora_rank, 
            seed=42,
            gpu_memory_utilization=0.85,
            enforce_eager=True,
        )
    else:
        llm = LLM(
            model=config.model_name, 
            dtype=dtype,
            max_model_len=config.max_seq_length + 4096,
            enable_lora=True,
            max_lora_rank=config.lora_rank, 
            seed=42,
            gpu_memory_utilization=0.85,
            enforce_eager=True,
        )
    
    if args.checkpoint_num==None:
        lora_name = f'{config.run_name}_grpo_saved_lora'
        lora_path = os.path.join(output_dir, "grpo_saved_lora")
    else:
        lora_name = f'{config.run_name}_checkpoint_{args.checkpoint_num}'
        lora_path = os.path.join(output_dir, f"checkpoint-{args.checkpoint_num}")
    lora_id = int(hashlib.sha1(lora_name.encode()).hexdigest(), 16) % 2**31-1
    lora = LoRARequest(
        lora_name=lora_name, 
        lora_int_id=lora_id, 
        lora_path=lora_path,
    )

    tokenizer = llm.get_tokenizer()
    tokenizer_kwargs = {}
    if "Qwen3" in config.model_name:
        tokenizer_kwargs["enable_thinking"] = False

    ### Load Test Data
    _, _, test_dataset = get_dataset(
        dataset_names=[args.dataset_name] if args.dataset_name!=None else config.dataset_names, 
        tokenizer=tokenizer, 
        num_train_samples=5000,
        sys_prompt_id=config.sys_prompt_id if config.sys_prompt_id else 1,
        include_raw_prompts=True,
        inference=True,
        scores_df_path=os.path.join(f"./exp2_rlmf/b_metacog_data_selection/score_dfs", "meta-llama/Meta-Llama-3.1-8B-Instruct".replace("-", "_").replace("/", "_")) if "llama" in config.model_name and "3.1" in config.model_name else (os.path.join(f"./exp2_rlmf/b_metacog_data_selection/score_dfs", "Qwen/Qwen3-8B".replace("-", "_").replace("/", "_")) if "wen3" in config.model_name else os.path.join(f"./exp2_rlmf/b_metacog_data_selection/score_dfs", config.model_name.replace("-", "_").replace("/", "_"))),
        mc_threshold=None if config.sys_prompt_id not in [11,13] else config.mc_threshold,
        use_length_direction=args.use_length_direction,
    )

    ### Prepare Prompts
    targets = [x['targets'] for x in test_dataset]
    prompts = [x['prompt'] for x in test_dataset]
    raw_prompts = [x['raw_prompt'] for x in test_dataset]
    if args.dev_mode:
        prompts = prompts[:100]
    
    ### Format Prompts
    chat_format_prompts = []
    for prompt in prompts:
        
        chat_format_prompts.append(
            tokenizer.apply_chat_template(
                prompt, 
                tokenize=False, 
                add_generation_prompt=True,
                **tokenizer_kwargs
            )
        )
    
    ### Run Inference via VLLM if Not Done Yet
    gen_kwargs = {}
    if "wen3" in config.model_name:
        gen_kwargs.update({
            'temperature': 0.7,
            'top_p': 0.8,
            'top_k': 20,
            "min_p": 0,
        })
    else:
        if args.temperature:
            gen_kwargs['temperature'] = args.temperature
        if args.top_p:
            gen_kwargs['top_p'] = args.top_p
        if args.top_k:
            gen_kwargs['top_k'] = args.top_k
    SAMPLING_PARAMS = SamplingParams(
        n=21,
        max_tokens=256,
        # max_tokens=config.max_seq_length, 
        **gen_kwargs,
    )
    if not os.path.isfile(save_path):
        responses = llm.generate(chat_format_prompts, SAMPLING_PARAMS, lora_request=lora)

        ### Extract Responses
        extracted_responses = []
        sampled_answers_lists = []
        for sub_response in responses:
            extracted_responses.append([x.text for x in sub_response.outputs[:1]][0])
            sampled_answers_lists.append([x.text for x in sub_response.outputs[1:]])
        assert(type(extracted_responses[0])==str)
        assert(len(sampled_answers_lists[0])==20)
        
        ### Save Results
        results = {
            'with_instruction': str(args.with_instruction), 
            'gen_kwargs': gen_kwargs,
            'tokenizer_kwargs': tokenizer_kwargs,
            'prompts': prompts,
            'targets': targets,
            'answers': extracted_responses,
            'sampled_answers': sampled_answers_lists,
            'raw_prompts': raw_prompts,
            'chat_format_prompts': chat_format_prompts,
        }
        with open(save_path, 'w') as f:
            f.write(json.dumps(results, indent=4, ensure_ascii=False))
    else: 
        print(colored("Already ran and saved predictions, loading from json to dict...", "cyan"))
        with open(save_path) as f:
            results = json.load(f)
        extracted_responses = results['answers']
        sampled_answers_lists = results['sampled_answers']

    ### Compute Faithfulness, Factual Calibration, Accuracy if Needed
    if not os.path.isfile(metrics_path): # or os.path.isfile(metrics_path):
        print(colored("Computing faithfulness scores...", "cyan"))
        faithfulness_rewards, metrics = get_metrics(extracted_responses, sampled_answers_lists, prompts, targets, config.format_rewards_version, config.reward_weights, hostnum=args.hostnum, 
        get_meta_score_from_completions=True if config.sys_prompt_id in [11, 12, 13] else False,
        get_mc_prediction_as_percentage=config.get_mc_prediction_as_percentage if hasattr(config, "get_mc_prediction_as_percentage") else (False if config.sys_prompt_id in [11, 12, 13] else True),
        )
        
        print(colored("Saving faithfulness scores...", "cyan"))
        with open(metrics_path, 'w') as f:
            f.write(json.dumps(metrics, indent=4, ensure_ascii=False))
        with open(rewards_path, 'w') as f:
            f.write(json.dumps(faithfulness_rewards, indent=4, ensure_ascii=False))
    else: 
        print(colored("Already computed and saved faithfulness scores, skipping this dataset...", "cyan"))

def inference_baseline(args):

    output_dir = os.path.join(f"./exp2_rlmf/c_rl_training/__models/", args.model_name.replace(f"./exp2_rlmf/c_rl_training/__models/meta_llama_Meta_Llama_3.1_8B_Instruct/", ""), f"_baseline_sys6")
    os.makedirs(output_dir, exist_ok=True)
    if args.use_length_direction:
        output_dir = os.path.join(output_dir, "_with_length_dirxn")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'test_preds_{args.dataset_name}.json')
    metrics_path = os.path.join(output_dir, f'test_scores_{args.dataset_name}.json')
    rewards_path = os.path.join(output_dir, f'test_rewards_{args.dataset_name}.json')
    
    if os.path.exists(save_path) and os.path.exists(metrics_path) and os.path.exists(rewards_path):
        print(colored(f"Running model {args.model_name} on dataset {args.dataset_name}...", "cyan"))
        print(colored("Already computed and saved faithfulness scores, skipping this dataset...", "cyan"))
        return
    else:
        print(colored(f"Running {args.model_name} on dataset {args.dataset_name}...", "cyan"))
    
    ### Load Model to VLLM
    print(colored("Porting model to VLLM...", "cyan"))
    if 'emma' in args.model_name:
        llm = LLM(
            model=args.model_name, 
            max_model_len=1024 + 4096,
            enforce_eager=True,
            seed=42,
            gpu_memory_utilization=0.85,
        )
    else:
        llm = LLM(
            model=args.model_name, 
            max_model_len=1024 + 4096,
            seed=42,
            gpu_memory_utilization=0.85,
        )
    tokenizer = llm.get_tokenizer()
    tokenizer_kwargs = {}
    if "Qwen3" in args.model_name:
        tokenizer_kwargs["enable_thinking"] = False

    ### Load Test Data
    _, _, test_dataset = get_dataset(
        dataset_names=[args.dataset_name], 
        tokenizer=tokenizer, 
        num_train_samples=5000,
        sys_prompt_id=6,
        include_raw_prompts=True,
        inference=True,
        scores_df_path=os.path.join(f"./exp2_rlmf/b_metacog_data_selection/score_dfs", args.model_name.replace("-", "_").replace("/", "_")) if "fut" not in args.model_name and "home" not in args.model_name else f"./exp2_rlmf/b_metacog_data_selection/score_dfs/meta_llama_Meta_Llama_3.1_8B_Instruct", 
        use_length_direction=args.use_length_direction,
    )
    print(colored(f"Using {len(test_dataset)} test samples!", "yellow"))
    print(test_dataset['prompt'][0])

    ### Prepare Prompts
    targets = [x['targets'] for x in test_dataset]
    prompts = [x['prompt'] for x in test_dataset]
    raw_prompts = [x['raw_prompt'] for x in test_dataset]
    if args.dev_mode:
        prompts = prompts[:10]
    
    ### Format Prompts
    chat_format_prompts = []
    for prompt in prompts:
        chat_format_prompts.append(
            tokenizer.apply_chat_template(
                prompt, 
                tokenize=False, 
                add_generation_prompt=True,
                **tokenizer_kwargs
            )
        )

    ### Run Inference via VLLM
    gen_kwargs = {}
    if "wen3" in args.model_name:
        gen_kwargs.update({
            'temperature': 0.7,
            'top_p': 0.8,
            'top_k': 20,
            "min_p": 0,
        })
    else:
        if args.temperature:
            gen_kwargs['temperature'] = args.temperature
        if args.top_p:
            gen_kwargs['top_p'] = args.top_p
        if args.top_k:
            gen_kwargs['top_k'] = args.top_k
    if not os.path.isfile(save_path):
    
        from vllm import SamplingParams
        SAMPLING_PARAMS = SamplingParams(
            n=21,
            max_tokens=8192,
            **gen_kwargs,
        )
        responses = llm.generate(chat_format_prompts, SAMPLING_PARAMS)

        ### Extract Responses
        extracted_responses = []
        sampled_answers_lists = []
        for sub_response in responses:
            extracted_responses.append([x.text for x in sub_response.outputs[:1]][0])
            sampled_answers_lists.append([x.text for x in sub_response.outputs[1:]])
        assert(type(extracted_responses[0])==str)
        assert(len(sampled_answers_lists[0])==20)
        
        ### Save Results
        results = {
            'with_instruction': str(args.with_instruction), 
            'gen_kwargs': gen_kwargs,
            'tokenizer_kwargs': tokenizer_kwargs,
            'prompts': prompts,
            'targets': targets,
            'answers': extracted_responses,
            'sampled_answers': sampled_answers_lists,
            'raw_prompts': raw_prompts,
            'chat_format_prompts': chat_format_prompts,
        }
        
        with open(save_path, 'w') as f:
            f.write(json.dumps(results, indent=4, ensure_ascii=False))
    else: 
        print(colored("Already ran and saved predictions, loading from json to dict...", "cyan"))
        with open(save_path) as f:
            results = json.load(f)
        extracted_responses = results['answers']
        sampled_answers_lists = results['sampled_answers']

    ### Compute Faithfulness, Factual Calibration, Accuracy
    if not os.path.isfile(metrics_path): #or os.path.isfile(metrics_path):
        print(colored("Computing faithfulness scores...", "cyan"))
        faithfulness_rewards, metrics = get_metrics(extracted_responses, sampled_answers_lists, prompts, targets, "new", reward_weights=[3., 1., 1., 12.,], hostnum=args.hostnum, get_meta_score_from_completions=False, get_mc_prediction_as_percentage=False)
        
        with open(metrics_path, 'w') as f:
            f.write(json.dumps(metrics, indent=4, ensure_ascii=False))
        with open(rewards_path, 'w') as f:
            f.write(json.dumps(faithfulness_rewards, indent=4, ensure_ascii=False))
    else: 
        print(colored("Already computed and saved faithfulness scores, skipping this dataset...", "cyan"))

if __name__ == '__main__':
    args = ArgumentParser()
    
    ### PoT Args
    args.add_argument('--merged_model_path', type=str, default=None)    # opt'l
    args.add_argument('--config_path', type=str, default=None)       
    args.add_argument('--with_instruction', action='store_true', default=False)
    args.add_argument('--hostnum', type=int, required=True)
    args.add_argument('--checkpoint_num', type=int, default=None)
    args.add_argument('--output_modifier', type=str, default="")

    ### Baseline Args
    args.add_argument('--model_name', type=str, default=None)    # opt'l
    args.add_argument('--dataset_name', type=str, default=None)    # opt'l

    args.add_argument('--temperature', type=float, default=None)
    args.add_argument('--top_p', type=float, default=None)
    args.add_argument('--top_k', type=int, default=None)
    args.add_argument('--dev_mode', action='store_true', default=False)
    args.add_argument('--use_length_direction', action='store_true', default=False)
    
    args = args.parse_args()
    
    if args.config_path!=None or args.merged_model_path!=None:
        inference(args)
    else: 
        inference_baseline(args)
