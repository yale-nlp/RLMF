import os
import json
import argparse
import numpy as np
import pandas as pd
from termcolor import colored

from vllm import LLM, SamplingParams

from src.exp2_rlmf.utils.data_utils import get_dataset

from huggingface_hub import login
login(os.getenv("HF_TOKEN"))

judgment_sys_prompt = "You are an agent with high metacognitive sensitivity and excellent self-awareness of your internal confidence and uncertainty."
judgment_prompt = """Question: {q}\nYour Answer: {a}\nAbove is a question and your own response to it. On a scale of 0-100, how confident are you that the linguistic decisiveness of your answer above matches your true internal confidence in that answer? Respond with a single integer between 0-100 and no other text."""

def get_meta_scores(dataset, llm, tokenizer, tokenizer_kwargs, gen_kwargs, dev_mode, sampled_answers_path, slice):

    ############################## 
    ### Load Sampled Answers
    ############################## 
    questions = [x['raw_prompt'] for x in dataset]
    prompts = [x['prompt'] for x in dataset]
    if dev_mode:
        questions = questions[:2]
        prompts = prompts[:2]
    else: 
        slice_size = len(questions) //4
        if slice<3:
            questions = questions[slice_size*slice:slice_size*(slice+1)]
            prompts = prompts[slice_size*slice:slice_size*(slice+1)]
        else: 
            questions = questions[slice_size*slice:]
            prompts = prompts[slice_size*slice:]
    
    # Load sampled answers from file
    with open(sampled_answers_path, 'r') as f:
        sampled_answers_dict = json.load(f)
    
    # Determine which key type to use based on file name
    if 'sampled_answers_by_prompt' in sampled_answers_path:
        # Use prompts as keys
        sampled_answers_lists = [
            sampled_answers_dict[str(p)] 
            for p in prompts
        ]
    elif 'sampled_answers_by_raw_prompt' in sampled_answers_path:
        # Use raw questions as keys
        sampled_answers_lists = [
            sampled_answers_dict[q] 
            for q in questions
        ]
    
    ############################## 
    ### Predict Meta-Scores for All Sampled Answers
    ############################## 
    print(colored("Getting meta-level confidence scores for all sampled answers...", "cyan"))

    # Process all sampled answers for all questions
    all_answers = []
    all_questions_expanded = []
    all_sampled_lists_expanded = []
    
    for q, sampled_list in zip(questions, sampled_answers_lists):
        for answer in sampled_list:
            all_answers.append(answer)
            all_questions_expanded.append(q)
            all_sampled_lists_expanded.append(sampled_list)
    
    judgment_prompts = [
        [
            {"role": "system", "content": judgment_sys_prompt},
            {"role": "user", "content": judgment_prompt.format(q=q, a=a) }
        ]
        for q, a in zip(all_questions_expanded, all_answers)
    ]
    judgment_prompts_list = tokenizer.apply_chat_template(
        judgment_prompts, 
        add_generation_prompt=True, 
        tokenize=False,
        **tokenizer_kwargs,
    )
    SAMPLING_PARAMS = SamplingParams(
        n=1,
        max_tokens=5,
        stop=["Question:"],
        **gen_kwargs,
    )
    outputs = []
    response = llm.generate(judgment_prompts_list, SAMPLING_PARAMS)
    for sub_response in response:
        outputs.append([x.text for x in sub_response.outputs])

    ### Extract Meta-confidence Scores from Predictions
    print(colored("Extracting meta-level confidence scores...", "cyan"))
    scores = []
    for x in outputs:
        try:
            score = float(x[0])
            scores.append(score)
        except Exception as e: 
            scores.append(-2.)
            print(colored("Error while parsing meta-level confidence score: ", "red"), e)
    scores = np.array(scores) / 100.    # convert to 0-1 range
    assert(len(scores) == len(all_answers))

    ############################## 
    ### Create DataFrames
    ############################## 
    # Create raw scores dataframe (one row per answer)
    raw_scores_df = pd.DataFrame({
        "raw_prompt": all_questions_expanded,
        "answer": all_answers,
        "meta_confidence_score": scores,
    })

    # Create one per prompt scores dataframe (one row per question)
    first_answer_df = raw_scores_df.groupby("raw_prompt").first().reset_index()
    
    # Create aggregated scores dataframe (one row per question)
    scores_aggr_over_sampled_answers_df = raw_scores_df.groupby("raw_prompt").agg({
        "meta_confidence_score": "mean",
    }).reset_index()

    return raw_scores_df, first_answer_df, scores_aggr_over_sampled_answers_df

def main(args):

    ### Load Tokenizer
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
    if "Qwen3" in args.model_name:
        tokenizer_kwargs["enable_thinking"] = False

    ### Load Data
    train_dataset, _, _ = get_dataset(
        dataset_names=[args.dataset_name], 
        tokenizer=tokenizer, 
        num_train_samples=None, # use all
        sys_prompt_id=0,
        include_raw_prompts=True,
        inference=False,
    )

    ### Get Meta-level Confidence Scores
    print(colored("Getting scores for training data...", "cyan"))
    output_dir = os.path.join("./exp2_rlmf/b_metacog_data_selection/score_dfs", args.model_name.replace("/", "_").replace("-", "_"))
    os.makedirs(output_dir, exist_ok=True)

    raw_scores_df_train, single_scores_df_train, scores_aggr_over_sampled_answers_df_train = get_meta_scores(train_dataset, llm, tokenizer, tokenizer_kwargs, gen_kwargs, args.dev_mode, args.sampled_answers_path, args.slice)

    ### Save Scores
    print(colored("Saving to dataframe...", "cyan"))

    output_path = os.path.join(output_dir, f"scores_raw_per_sampled_answer_{args.dataset_name}_train_sys{args.sys_prompt_id}_{args.slice}.csv")
    raw_scores_df_train.to_csv(output_path)

    output_path = os.path.join(output_dir, f"scores_one_per_prompt_{args.dataset_name}_train_sys{args.sys_prompt_id}_{args.slice}.csv")
    single_scores_df_train.to_csv(output_path)

    output_path_final = os.path.join(output_dir, f"scores_aggr_over_sampled_answers_{args.dataset_name}_train_sys{args.sys_prompt_id}_{args.slice}.csv")
    scores_aggr_over_sampled_answers_df_train.to_csv(output_path_final)
    print(colored("Saved final dfs to...", "green"), output_path_final+colored("!", "green"))

def parse_args():
    parser = argparse.ArgumentParser()

    ### MODEL ARGS
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--sys_prompt_id", type=int)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--gpu_mem_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=float, default=1)

    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument("--max_model_len", type=int, default=8192, help="Max length of model input")

    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--dev_mode", action="store_true", default=False)
    parser.add_argument("--sampled_answers_path", type=str)

    parser.add_argument('--slice', type=int, choices=[0,1,2,3])

    args = parser.parse_args()
    return args
 
if __name__ == "__main__":
    args = parse_args()
    main(args)
