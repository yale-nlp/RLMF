import os
import json
import numpy as np
from termcolor import colored
from argparse import ArgumentParser

import torch
from vllm import LLM
from vllm.lora.request import LoRARequest

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

from src.exp2_rlmf.utils.data_utils import get_dataset
from src.exp2_rlmf.utils.utils import get_sentence_internal_confidence


dtypes = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16
}

def get_metrics(extracted_responses, sampled_answers_lists, hostnum):

    ### Extract Sentences & Predicted Confidence Floats
    sentences_per_response = [
        sent_tokenize(text) for text in extracted_responses
    ]
    assert(len(sentences_per_response)==len(sampled_answers_lists))

    lengths = [len(s) for s in sentences_per_response]
    p5  = int(np.percentile(lengths, 5))   # robust "min"
    p95 = int(np.percentile(lengths, 85))
        
    ### Compute Gold Confidence
    gold_confs_per_response = get_sentence_internal_confidence(
        sent_lists=sentences_per_response, 
        sampled_answers_lists=sampled_answers_lists,
        hostname=f"http://localhost:{hostnum}/v1",
    )   # List of list of floats
   
    avg_gold_confidences = [
        -1. if not (vals := [c for c in conf_list if c is not None]) else np.mean(vals)
        for conf_list in gold_confs_per_response
    ]

    return gold_confs_per_response, avg_gold_confidences, p5, p95


def get_predictions(args):

    model_identifier = args.model_name.replace("-","_").replace("/", "_")
    output_dir = os.path.join(f"./exp2_rlmf/a_pre_sft", model_identifier)
    os.makedirs(output_dir, exist_ok=True)

    pred_path = os.path.join(output_dir, f'preds_{args.dataset_name}_{args.num_samples}samps.json')
    result_path = os.path.join(output_dir, f'preds_with_scores_{args.dataset_name}_{args.num_samples}samps.json')
    
    if os.path.exists(pred_path) and os.path.exists(result_path):
        print(colored(f"Already got predictions for {args.model_name} on dataset {args.dataset_name}, skipping...", "cyan"))
        return
    else:
        print(colored(f"Getting predictions for {args.model_name} on dataset {args.dataset_name}...", "cyan"))
    
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
    train_dataset, _, _ = get_dataset(
        dataset_names=[args.dataset_name], 
        tokenizer=tokenizer, 
        num_train_samples=5000,
        sys_prompt_id=-2,
        include_raw_prompts=True,
        inference=True,
        scores_df_path=os.path.join(f"./exp2_rlmf/b_metacog_data_selection/score_dfs", args.model_name.replace("-", "_").replace("/", "_")),
    )

    ### Prepare Prompts
    targets = [x['targets'] for x in train_dataset][-args.num_samples:]
    prompts = [x['prompt'] for x in train_dataset][-args.num_samples:]
    raw_prompts = [x['raw_prompt'] for x in train_dataset][-args.num_samples:]
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
    if "wen3" in args.model_name and '2507' not in args.model_name:
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
   
    if not os.path.isfile(pred_path): 
        from vllm import SamplingParams
        SAMPLING_PARAMS = SamplingParams(
            n=21,
            max_tokens=1024,
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
            'gen_kwargs': gen_kwargs,
            'tokenizer_kwargs': tokenizer_kwargs,
            'prompts': prompts,
            'targets': targets,
            'answers': extracted_responses,
            'sampled_answers': sampled_answers_lists,
            'raw_prompts': raw_prompts,
            'chat_format_prompts': chat_format_prompts,
        }
        
        with open(pred_path, 'w') as f:
            f.write(json.dumps(results, indent=4, ensure_ascii=False))
    else: 
        print(colored("Already ran and saved predictions, loading from json to dict...", "cyan"))
        with open(pred_path) as f:
            results = json.load(f)
        extracted_responses = results['answers']
        sampled_answers_lists = results['sampled_answers']

    ### Compute Gold Confidence
    if not os.path.isfile(result_path):
        print(colored("Computing gold confidence...", "cyan"))
        gold_confs_per_response, avg_gold_confidences, min_sentences, max_sentences = get_metrics(extracted_responses, sampled_answers_lists, hostnum=args.hostnum)
        
        results['gold_confs_per_response'] = gold_confs_per_response
        results['avg_gold_confidences'] = avg_gold_confidences
        results['min_sentences'] = min_sentences
        results['max_sentences'] = max_sentences

        with open(result_path, 'w') as f:
            f.write(json.dumps(results, indent=4, ensure_ascii=False))
    else: 
        print(colored("Already computed and saved faithfulness scores, skipping this dataset...", "cyan"))

if __name__ == '__main__':
    args = ArgumentParser()
    
    args.add_argument('--model_name', type=str, default=None)    
    args.add_argument('--dataset_name', type=str, default=None)    
    args.add_argument('--hostnum', type=int, required=True)
    args.add_argument('--num_samples', type=int, default=100)

    args.add_argument('--temperature', type=float, default=None)
    args.add_argument('--top_p', type=float, default=None)
    args.add_argument('--top_k', type=int, default=None)
    args.add_argument('--dev_mode', action='store_true', default=False)
    
    args = args.parse_args()
    get_predictions(args)