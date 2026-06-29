import os
import json
import argparse
from termcolor import colored

from vllm import LLM, SamplingParams

from src.exp2_rlmf.utils.data_utils import get_dataset

from huggingface_hub import login
login(os.getenv("HF_TOKEN"))

def get_sampled_answers(dataset, llm, tokenizer, tokenizer_kwargs, gen_kwargs, dev_mode):

    ############################## 
    ### Predict Answers
    ############################## 
    questions = [x['raw_prompt'] for x in dataset]
    prompts = [x['prompt'] for x in dataset]
    if dev_mode:
        questions = questions[:10]
        prompts = prompts[:10]
    chat_format_prompts = [
        tokenizer.apply_chat_template(
            msg_list, 
            tokenize=False, 
            add_generation_prompt=True,
            **tokenizer_kwargs
        )
        for msg_list in prompts
    ]
    assert(len(questions)==len(chat_format_prompts)==len(prompts))
    
    ### Get Predictions for All Samples
    SAMPLING_PARAMS = SamplingParams(
        n=20,
        max_tokens=1024,
        **gen_kwargs,
    )
    print(colored("Getting 20 predictions per sample...", "cyan"))
    raw_outputs = llm.generate(chat_format_prompts, SAMPLING_PARAMS)
    sampled_answers_lists = []
    for sub_response in raw_outputs:
        sampled_answers_lists.append([x.text for x in sub_response.outputs[0:]])
    assert(len(sampled_answers_lists[0])==20)

    sampled_answers_by_prompt = {
        str(p): sampled_answer_list 
        for p, sampled_answer_list in zip(prompts, sampled_answers_lists)
    }
    sampled_answers_by_raw_prompt = {
        q: sampled_answer_list 
        for q, sampled_answer_list in zip(questions, sampled_answers_lists)
    }

    return sampled_answers_by_prompt, sampled_answers_by_raw_prompt
    
def main(args):

    ### Load Model
    print(colored("Loading model to VLLM...", "cyan"))
    if 'emma' in args.model_name:
        llm = LLM(
            model=args.model_name, 
            dtype=args.dtype, 
            max_model_len=args.max_model_len, 
            seed=42,
            gpu_memory_utilization=args.gpu_mem_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            enforce_eager=True,
        )
    else:
        llm = LLM(
            model=args.model_name, 
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            seed=42,
            gpu_memory_utilization=args.gpu_mem_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
        )
    
    ### Prepare Generation Kwargs
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
    
    ### Load Tokenizer
    tokenizer = llm.get_tokenizer()
    tokenizer_kwargs = {}
    if "Qwen3" in args.model_name and args.model_name!="Qwen/Qwen3-4B-Instruct-2507":
        tokenizer_kwargs["enable_thinking"] = False

    ### Load Data
    train_dataset, val_dataset, test_dataset = get_dataset(
        dataset_names=[args.dataset_name], 
        tokenizer=tokenizer, 
        num_train_samples=None, # use all
        sys_prompt_id=args.sys_prompt_id,
        include_raw_prompts=True,
        inference=False,
    )

    ### Get Sampled Answers
    print(colored("Getting sampled answers for training and validation data...", "cyan"))
    sampled_answers_by_prompt_train, sampled_answers_by_raw_prompt_train = get_sampled_answers(train_dataset, llm, tokenizer, tokenizer_kwargs, gen_kwargs, args.dev_mode)
    sampled_answers_by_prompt_val, sampled_answers_by_raw_prompt_val = get_sampled_answers(val_dataset, llm, tokenizer, tokenizer_kwargs, gen_kwargs, args.dev_mode)

    ### Combine Dicts
    sampled_answers_by_prompt = {
        **sampled_answers_by_prompt_train,
        **sampled_answers_by_prompt_val,
    }
    sampled_answers_by_raw_prompt = {
        **sampled_answers_by_raw_prompt_train,
        **sampled_answers_by_raw_prompt_val,
    }
    assert all(isinstance(v, list) for v in sampled_answers_by_prompt.values())
    assert all(isinstance(v, list) for v in sampled_answers_by_raw_prompt.values())

    ### Save Scores
    print(colored("Saving to CSVs...", "cyan"))
    output_dir = os.path.join(f"./exp2_rlmf/b_metacog_data_selection/sampled_answers_lists", args.model_name.replace("/", "_").replace("-", "_"))
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, f"sampled_answers_by_prompt_{args.dataset_name}_sys{args.sys_prompt_id}.json"), 'w') as f:
        json.dump(sampled_answers_by_prompt, f, indent=4)

    with open(os.path.join(output_dir, f"sampled_answers_by_raw_prompt_{args.dataset_name}_sys{args.sys_prompt_id}.json"), 'w') as f:
        json.dump(sampled_answers_by_raw_prompt, f, indent=4)

    print(colored("Saved to...", "green"), f"./exp2_rlmf/b_metacog_data_selection/sampled_answers_lists"+colored("!", "green"))

def parse_args():
    parser = argparse.ArgumentParser()

    ### MODEL ARGS
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--gpu_mem_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=float, default=1)

    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument("--max_model_len", type=int, default=8192, help="Max length of model input")

    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--dev_mode", action="store_true", default=False)
    parser.add_argument("--sys_prompt_id", type=int, default=0)

    args = parser.parse_args()
    return args
 
if __name__ == "__main__":
    args = parse_args()
    main(args)