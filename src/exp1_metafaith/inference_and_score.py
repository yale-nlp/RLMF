import argparse
import torch
import pandas as pd
import numpy as np
import json
import re
import spacy
import os
import time
import pickle
from termcolor import colored

from google.genai import types
import google.generativeai as genai
from google.generativeai import GenerationConfig
from vllm import LLM, SamplingParams

from src.exp0_baseline.utilities.utils import get_cmfg_mfg, llm_eval, get_cmfg_star
from src.exp0_baseline.metrics.decisiveness import get_decisiveness
from src.exp1_metafaith.sys_prompts import SYS_PROMPT_REGISTRY
from src.exp1_metafaith.data_utils import get_dataset
from src.exp2_rlmf.utils.utils import get_sentence_internal_confidence


from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def serialize(obj):
    """
    Serialize args values to enable writing to json file.
    """
    if isinstance(obj, (np.dtype, torch.dtype)):  # Handle data types
        return str(obj)
    elif isinstance(obj, set):  # Convert sets to lists
        return list(obj)
    elif hasattr(obj, "__dict__"):  # Try to serialize objects with a __dict__ attribute
        return vars(obj)
    else:
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def parse_args():
    """
    Define and parse script arguments.
    """

    parser = argparse.ArgumentParser()

    ### Set model args
    parser.add_argument("--model_name", type=str, help="Name of model (e.g. in HF hub)", required=True)
    parser.add_argument("--dtype", default=torch.float16, help="Data type to use for model loading", choices=[torch.bfloat16, torch.float16])
    parser.add_argument("--trust_remote_code", type=bool, default=True, help="Whether to trust remote code in HF hub")
    parser.add_argument("--gpu_mem_utilization", type=float, default=0.9, help="Proportion of GPU memory to use for VLLM model loading")
    parser.add_argument("--tensor_parallel_size", type=float, default=1, help="Number of GPUs to use for VLLM model loading")

    parser.add_argument("--model_max_len", type=int, default=4096, help="Max length of model input")
    parser.add_argument("--max_output_tokens", type=int, default=256, help="Max tokens model can generate at inference time")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for model inference")
    parser.add_argument("--top_p", type=float, default=None, help="Top_p for model inference")
    parser.add_argument("--top_k", type=float, default=None, help="Top_p for model inference")

    parser.add_argument("--sys_prompt", help="System prompt to use", default="sys1")

    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset/task to use")
    parser.add_argument("--split", type=str, required=True, choices=["train", "test"], help="Whether to use train or test split of dataset")
    parser.add_argument("--num_samples", default=1000, type=int, help="Max number of instances to run inference on")
    parser.add_argument("--random_seed", default=42, type=int, help="Seed for sampling when `num_samples` is smaller than dataset size")

    parser.add_argument("--output_dir", type=str, default="./exp1_metafaith/_results", help="Directory for output files")
    parser.add_argument("--dev_mode", action='store_true', default=False)
    parser.add_argument("--score_mode", type=int, help="Scoring mode", choices=[1,2,3,4])
    parser.add_argument("--hostnum", type=int, help="Host ID for local judgments")
    parser.add_argument("--no_score", action='store_true', default=False, help="Skip scoring, just get predictions")

    args = parser.parse_args()
    return args

def get_proprietary_response(sys_prompt, prompt, model_name, model, gen_kwargs, stop_seqs, k, max_output_tokens):

    # Gemini inference
    if "gemini" in model_name:
        responses = []
        for i in range(k):
            try:
                if "gemini-3" in model_name:
                    if "pro" in model_name:
                        config = types.GenerateContentConfig(
                            system_instruction=sys_prompt,
                            max_output_tokens=max_output_tokens,
                            candidate_count=1,
                            stop_sequences=stop_seqs,
                            thinking_config=types.ThinkingConfig(thinking_level="low"),
                            temperature=gen_kwargs["temperature"],
                        )
                    else: 
                        config = types.GenerateContentConfig(
                            system_instruction=sys_prompt,
                            max_output_tokens=max_output_tokens,
                            candidate_count=1,
                            stop_sequences=stop_seqs,
                            thinking_config=types.ThinkingConfig(thinking_level="low"),
                            temperature=gen_kwargs["temperature"],
                        )
                else: 
                    config = types.GenerateContentConfig(
                        system_instruction=sys_prompt,
                        max_output_tokens=max_output_tokens,
                        candidate_count=1,
                        stop_sequences=stop_seqs,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                        temperature=gen_kwargs["temperature"],
                    )
                response = model.models.generate_content(
                    model=model_name, 
                    config=config,
                    contents=prompt,
                )
                response_text = response.text.strip()
                responses.append(response_text)
            except Exception as e: 
                print(e)
                responses.append("GENERATION ERROR")
        
    # GPT inference
    elif "gpt" in model_name:

        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # If NOT Reasoning Model
        if "5" not in model_name and "o" not in model_name:  
            response = client.chat_completions_create(
                model=model_name,
                messages=messages,
                n=k,
                max_completion_tokens=max_output_tokens,
                stop=stop_seqs,
                **gen_kwargs,
            )['choices']
            responses = [x['message']['content'] for x in response]
       
        # If YES Reasoning Model
        else:  # not yet incorporated: responses API for reasoning summary
            try:
                response1 = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    n=8,
                    max_completion_tokens=max_output_tokens,
                    reasoning_effort="minimal",
                    # stop=stop_seqs,
                    **gen_kwargs,
                ).choices
                responses1 = [x.message.content for x in response1]

                response2 = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    n=8,
                    max_completion_tokens=max_output_tokens,
                    reasoning_effort="minimal",
                    # stop=stop_seqs,
                    **gen_kwargs,
                ).choices
                responses2 = [x.message.content for x in response2]

                response3 = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    n=k-16,
                    max_completion_tokens=max_output_tokens,
                    reasoning_effort="minimal",
                    # stop=stop_seqs,
                    **gen_kwargs,
                ).choices
                responses3 = [x.message.content for x in response3]
                response = response1+response2+response3
                responses = responses1 + responses2 + responses3
            except Exception as e:
                print(e)
                responses = ["GENERATION ERROR"] * k

    return responses

def load_model(args):
    """
    Load model as specified in args.
    """

    # Prepare Gemini model
    if "gemini" in args.model_name:
        tokenizer = None
        google_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=google_key)
        from google import genai as ga
        model = ga.Client(api_key=google_key)

    # Prepare GPT model
    elif "gpt" in args.model_name:
        tokenizer = None
        model = None

    # Prepare HF / VLLM model
    else: 
        ###################################
        #### Login to HF
        ###################################
        from huggingface_hub import login
        login(os.getenv("HF_TOKEN"))
        model = LLM(
            model=args.model_name, 
            trust_remote_code=args.trust_remote_code,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_mem_utilization,
            tensor_parallel_size=int(args.tensor_parallel_size),
            max_model_len=args.model_max_len if "OLMo" not in args.model_name else min(args.model_max_len, 4096),
            seed=42,
            enforce_eager=True,
            # max_num_seqs=128,   
        )
        tokenizer = model.get_tokenizer()

    return tokenizer, model

def get_exp_identifier(args):
    return f"{args.sys_prompt}_{args.dataset_name}_{args.num_samples}samps"

def get_model_identifier(args):
    model_identifier = args.model_name.replace("-", "_").replace("/", "_")
    return model_identifier

def run(args):

    start_time = time.time()
    args.max_output_tokens = int(args.max_output_tokens)
    assert(type(args.max_output_tokens)==int)

    # Prepare output directories
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    model_identifier = get_model_identifier(args)
    model_dir = os.path.join(output_dir, model_identifier)
    os.makedirs(model_dir, exist_ok=True)

    task_identifier = get_exp_identifier(args)

    # Define output files
    results_file = os.path.join(model_dir, f"{args.split}_preds_{task_identifier}.json")
    metrics_file = os.path.join(model_dir, f"{args.split}_scores_{task_identifier}.json")

    run_preds = None
    create_data_from_scratch = True
    
    # If already finished preds + scoring
    if os.path.exists(results_file) and os.path.exists(metrics_file):
        print(colored(f"Running {args.model_name} on task {task_identifier} {args.split} split...", "cyan"))
        print(colored("Already computed and saved scores, skipping this run...", "cyan"))
        return
    else:
        # If finished preds (or partially for GPT/Gemini)
        if os.path.exists(results_file):

            create_data_from_scratch = False

            # If VLLM
            if "gemini" not in args.model_name and "gpt" not in args.model_name:

                print(colored(f"Already completed predictions for task {task_identifier} for {model_identifier}, loading from json to dict...", "cyan"))
                
                with open(results_file) as f:
                    results = json.load(f)
                
                extracted_responses = results['answers']
                sampled_answers_lists = results['sampled_answers']
                prompts = results['prompts']
                targets = results['targets']

                run_preds = False
                
            # If Proprietary
            else: 

                with open(results_file) as f:
                    results = json.load(f)
                
                finished_all = results['finished_all']
                prompts = results['prompts']
                targets = results['targets']
                extracted_responses = results['answers']
                sampled_answers_lists = results['sampled_answers']
                raw_prompts = results['raw_prompts']
                chat_format_prompts = results['chat_format_prompts']

                # If done
                if finished_all=="true":
                    print(colored(f"Already completed predictions for task {task_identifier} for {model_identifier}, loading from json to dict...", "cyan"))
                    run_preds = False

                # If failed partway
                else: 
                    start_idx = int(finished_all) + 1 
                    print(colored(f"Running {args.model_name} on task {task_identifier} {args.split} split, starting from index {start_idx}...", "cyan"))
                    run_preds = True
                    extracted_responses = extracted_responses[:start_idx]
                    sampled_answers_lists = sampled_answers_lists[:start_idx]

        # If not yet run preds
        else:
            print(colored(f"Running {args.model_name} on task {task_identifier} {args.split} split...", "cyan"))
            
            run_preds = True

            if "gemini" in args.model_name or "gpt" in args.model_name:
                start_idx = 0

    if run_preds == True:

        # Save arguments
        args_file = os.path.join(model_dir, f"_args.json")
        if not os.path.exists(args_file):
            with open(args_file, "w") as f:
                json.dump(vars(args), f, indent=4, default=serialize)  

        # Load task
        sys_prompt = SYS_PROMPT_REGISTRY[args.sys_prompt.lower()]
        if create_data_from_scratch == True:
            train_dataset, test_dataset = get_dataset(dataset_names=[args.dataset_name], num_samples=args.num_samples, sys_prompt=sys_prompt, use_length_direction=False)

            if args.split=="train": 
                dataset = train_dataset
            else: 
                dataset = test_dataset
            if dataset is None: return

            # Prepare prompts
            raw_prompts = [x['raw_prompt'] for x in dataset]
            prompts = [x['prompt'] for x in dataset]
            targets = [x['targets'] for x in dataset]

            if args.dev_mode:
                prompts = prompts[:5]
                raw_prompts = raw_prompts[:5]

        # Load model and tokenizer
        tokenizer, model = load_model(args)
        stop_seqs = ["Question:"]

        if "gemini" not in args.model_name and "gpt" not in args.model_name:

            tokenizer_kwargs = {}
            if "Qwen3" in args.model_name and args.model_name!="Qwen/Qwen3-4B-Instruct-2507":
                tokenizer_kwargs["enable_thinking"] = False

            gen_kwargs = {}
            if "wen3" in args.model_name:
                gen_kwargs.update({
                    'temperature': 0.7,
                    'top_p': 0.8,
                    'top_k': 20,
                    "min_p": 0.,
                })
            else:
                if args.temperature:
                    gen_kwargs['temperature'] = args.temperature
                if args.top_p:
                    gen_kwargs['top_p'] = args.top_p
                if args.top_k:
                    gen_kwargs['top_k'] = args.top_k
            gen_kwargs['max_tokens'] = args.max_output_tokens

            SAMPLING_PARAMS = SamplingParams(
                n=21,
                stop=stop_seqs,
                **gen_kwargs,
            )

            # Prepare prompts
            chat_format_prompts = []
            for prompt in raw_prompts:
                if 'emma' in args.model_name:
                    message = [
                        {'role': 'user', 'content': sys_prompt + prompt + "\nAnswer:"}
                    ]
                elif 'wen3' in args.model_name:
                    message = [
                        {'role': 'system', 'content': sys_prompt},
                        {'role': 'user', 'content': prompt + "\nAnswer:"},
                    ]
                else: 
                    message = [
                        {'role': 'system', 'content': sys_prompt},
                        {'role': 'user', 'content': prompt + "\nAnswer:"},
                    ]
                chat_format_prompts.append(
                    tokenizer.apply_chat_template(
                        message, 
                        tokenize=False, 
                        add_generation_prompt=True,
                        **tokenizer_kwargs
                    )
                )

            # Run model on samples in dataset
            responses = model.generate(chat_format_prompts, SAMPLING_PARAMS)

            # Extract responses
            extracted_responses = []
            sampled_answers_lists = []
            for sub_response in responses:
                extracted_responses.append([x.text for x in sub_response.outputs[:1]][0])
                sampled_answers_lists.append([x.text for x in sub_response.outputs[1:]])
            assert(type(extracted_responses[0])==str)
            assert(len(sampled_answers_lists[0])==20)

        else: 
            tokenizer_kwargs = {}
            gen_kwargs = {}
            if args.temperature:
                gen_kwargs['temperature'] = args.temperature
            if args.top_p:
                gen_kwargs['top_p'] = args.top_p
            if args.top_k:
                gen_kwargs['top_k'] = args.top_k

            # Prepare prompts
            if create_data_from_scratch == True:
                chat_format_prompts = []
                extracted_responses = []
                sampled_answers_lists = []

            # Run model on samples in dataset
            for idx, prompt in enumerate(raw_prompts):

                if idx < start_idx: continue
                print(colored(f"On index {idx}", "yellow"))

                responses = get_proprietary_response(sys_prompt=sys_prompt, prompt=prompt+"\nAnswer:", model_name=args.model_name, model=model, gen_kwargs=gen_kwargs, stop_seqs=stop_seqs, k=21, max_output_tokens=args.max_output_tokens)
                extracted_responses.append(responses[0])
                sampled_answers_lists.append(responses[1:])
                assert(type(extracted_responses[0])==str)
                assert(len(sampled_answers_lists[0])==20)

                # Save ongoing results
                results = {
                    'gen_kwargs': gen_kwargs,
                    'tokenizer_kwargs': tokenizer_kwargs,
                    'prompts': prompts,
                    'targets': targets,
                    'answers': extracted_responses,
                    'sampled_answers': sampled_answers_lists,
                    'raw_prompts': raw_prompts,
                    'chat_format_prompts': chat_format_prompts,
                }
                if "gemini" in args.model_name or "gpt" in args.model_name and responses[0]!="GENERATION ERROR":
                    results['finished_all'] = str(idx)
                with open(results_file, 'w') as f:
                    f.write(json.dumps(results, indent=4, ensure_ascii=False))

        # Save results
        results = {
            'predxn_runtime_seconds': time.time() - start_time,
            'gen_kwargs': gen_kwargs,
            'tokenizer_kwargs': tokenizer_kwargs,
            'prompts': prompts,
            'targets': targets,
            'answers': extracted_responses,
            'sampled_answers': sampled_answers_lists,
            'raw_prompts': raw_prompts,
            'chat_format_prompts': chat_format_prompts,
        }
        if "gemini" in args.model_name or "gpt" in args.model_name:
            results['finished_all'] = "true"
        with open(results_file, 'w') as f:
            f.write(json.dumps(results, indent=4, ensure_ascii=False))
        print(colored(f"\nSaved predictions and runtime to {results_file}...", "cyan"))

    if args.no_score==False:
        if os.path.exists(metrics_file):
            print(colored("Already computed and saved scores for all scoring modes, skipping this dataset...", "cyan"))
        else: 
            if args.score_mode==2:
                print(colored(f"Computing scores (mode {args.score_mode} -- gold conf per sentence per response, acc per response)...", "cyan"))
            elif args.score_mode==3:
                print(colored(f"Computing scores (mode {args.score_mode} -- f, d, c per response, dc gap per sentence per response)...", "cyan"))
            elif args.score_mode==1:
                print(colored(f"Computing scores (mode {args.score_mode} -- dec per sentence per response)...", "cyan"))
        
            hostnum = args.hostnum if args.score_mode==2 else None
            get_metrics(extracted_responses, sampled_answers_lists, prompts, targets, hostnum, args.score_mode, args.model_name, model_dir, args.split, task_identifier, metrics_file)

def get_metrics(extracted_responses, sampled_answers_lists, prompts, targets, hostnum, mode, model_name, model_dir, split, task_identifier, metrics_file):

    output_file = os.path.join(model_dir, f"_chkpt_{split}_scores_{task_identifier}", f"output_mode_{mode}.pkl")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if os.path.exists(output_file): 
        print(colored(f"Already finished so skipping...", "cyan"))
        return

    # Separate each response into sentences
    nlp = spacy.load("en_core_web_sm")
    sentences_per_response = [
        [sent.text for sent in nlp(x).sents if "please" not in sent.text.lower()] # and "?" not in sent.text
        for x in extracted_responses
    ]

    if mode==1:
        # Compute decisiveness per sentence
        dec_per_sentence_per_response = [] 
        for i, sent_list in enumerate(sentences_per_response): 
            print(f"Scoring decisiveness for sample {i}")
            sent_dec_scores = [
                get_decisiveness(answer=sent) for sent in sent_list
            ]
            dec_per_sentence_per_response.append(sent_dec_scores)
            with open(output_file.replace("output", "checkpoint"), "wb") as f:
                pickle.dump(dec_per_sentence_per_response, f)

            with open(output_file.replace("output", "checkpoint").replace("pkl", "txt"), "a") as f:
                f.write(f"idx {i} " + str(sent_list) + "\n")
                f.write(f"    " + str(sent_dec_scores) + "\n")

        with open(output_file, "wb") as f:
            pickle.dump(dec_per_sentence_per_response, f)

        print(colored(f"Finished Mode {mode} scoring for task {task_identifier} split {split} for model {model_name} & saved result to {output_file}!", "green"))

    elif mode==2:

        # Compute internal confidence for each sentence
        gold_conf_per_sentence_per_response = get_sentence_internal_confidence(
                sent_lists=sentences_per_response, 
                sampled_answers_lists=sampled_answers_lists,
                hostname=f"http://localhost:{hostnum}/v1",
            )   # List of list of floats

        avg_gold_confidences = [
            -1.0 if not (vals := [c for c in conf_list if c is not None]) else np.mean(vals)
            for conf_list in gold_conf_per_sentence_per_response
        ]
        all_avg_gold_confidences = np.array(avg_gold_confidences, dtype=np.float64)
        
        # Compute accuracy per response
        responses_without_please = [
            " ".join(sent_list) for sent_list in sentences_per_response
        ]

        acc_per_response = []
        for idx, (targs, pred) in enumerate(zip(targets, responses_without_please)):
            acc_per_response.append(llm_eval(targs, pred))
            if idx%100==0: print(f"    Finished index {idx}!")

        all_accs = np.array(acc_per_response, dtype=np.float64)
        avg_acc  = float(all_accs.mean())

        brier_scores = (all_accs - all_avg_gold_confidences) ** 2.0
        mask = all_avg_gold_confidences == -1.0
        brier_scores[mask] = -1.0
        valid_brier = brier_scores[brier_scores != -1.0]
        avg_bs = float(valid_brier.mean()) if len(valid_brier) > 0 else float("nan")

        with open(output_file, "wb") as f:
            pickle.dump((gold_conf_per_sentence_per_response, responses_without_please, acc_per_response, avg_acc, avg_bs), f)
        
        print(colored(f"Finished Mode {mode} scoring for task {task_identifier} split {split} for model {model_name} & saved result to {output_file}!", "green"))

    elif mode==3:

        mode1_done = file_exists(output_file.replace("mode_3", "mode_1"))
        mode2_done = file_exists(output_file.replace("mode_3", "mode_2"))

        assert(mode1_done==mode2_done==True)

        if os.path.exists(metrics_file): 
            print(colored(f"Already finished so skipping...", "cyan"))
            return
        
        print(colored(f"    Found all 2 output files! Computing mode 4 metrics...", "yellow"))

        with open(output_file.replace("mode_3", "mode_2"), "rb") as f:
            gold_conf_per_sentence_per_response, responses_without_please, acc_per_response, avg_acc, avg_bs = pickle.load(f)

        with open(output_file.replace("mode_3", "mode_1"), "rb") as f:
            dec_per_sentence_per_response = pickle.load(f)
            
        # 6. Compute faithfulness score per response & DC gap per sentence per response
        dc_gap_per_sentence_per_response = []
        f_score_per_response = []
        d_per_response = []
        c_per_response = []
        for sentences, decs, gold_confs in zip(sentences_per_response, dec_per_sentence_per_response, gold_conf_per_sentence_per_response):
            
            decs_np = np.array(decs)
            gold_confs_np = np.array(gold_confs)

            invalid_or_no_output = (len(sentences) == 0 or sentences[0]=="GENERATION ERROR")
            dec_valid = (decs_np != -1)
            gold_valid = (gold_confs_np != -1)

            if invalid_or_no_output:
                mask = np.zeros_like(gold_valid)
            else:
                mask = dec_valid & gold_valid
            
            if len(gold_confs_np[mask])>0:
                f_score = 1. - (1. / len(sentences)) * np.abs(decs_np[mask] - gold_confs_np[mask]).sum()
            else: 
                f_score = -1.
            f_score_per_response.append(f_score)

            mean_dec = decs_np[mask].mean() if mask.sum() > 0 else -1.
            mean_conf = gold_confs_np[mask].mean() if mask.sum() > 0 else -1.

            dc_gaps = np.abs(decs_np[mask] - gold_confs_np[mask]).tolist() if mask.sum() > 0 else []

            d_per_response.append(mean_dec)
            c_per_response.append(mean_conf)
            dc_gap_per_sentence_per_response.append(dc_gaps)

        # 7. Compute cmfg score for dataset
        cmfg, mfg, stats = get_cmfg_mfg(
            f_score_per_response, 
            c_per_response, 
            num_bins=10,
        )
        cmfg_star, var_cmfg_star, bin_info = get_cmfg_star(
            f_score_per_response,
            c_per_response,   # plain Python list, as computed above
            num_bins=10,
        )

        metrics = {
            "avg_acc": avg_acc,
            "avg_bs": avg_bs,
            "cmfg_star": cmfg_star, 
            "var_cmfg_star": var_cmfg_star,
            # "bin_info": bin_info,
            "cmfg": cmfg,
            "mfg": mfg,
            "stats": stats,
            "sentences_per_response": sentences_per_response,
            "responses_without_please": responses_without_please,
            "dec_per_sentence_per_response": dec_per_sentence_per_response,
            "gold_conf_per_sentence_per_response": gold_conf_per_sentence_per_response,
            "dc_gap_per_sentence_per_response": dc_gap_per_sentence_per_response,
            "acc_per_response": acc_per_response,
            "f_score_per_response": f_score_per_response,
            "d_per_response": d_per_response,
            "c_per_response": c_per_response,
        }

        with open(metrics_file, 'w') as f:
            f.write(json.dumps(metrics, indent=4, ensure_ascii=False))

        print(colored(f"Finished Mode {mode} scoring for task {task_identifier} split {split} for model {model_name} & saved result to {output_file}!", "green"))


def file_exists(file_path, interval=300):
    """
    Waits for a file to appear at the specified path.
    Args:
        file_path (str): The path to the file.
        interval (int): Time to sleep between checks in seconds.
    """
    while not os.path.exists(file_path):
        print(f"    Waiting for file {file_path}...")
        time.sleep(interval)
    return True

if __name__ == "__main__":
    args = parse_args()
    run(args)