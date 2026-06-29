import os, pickle
import json
import argparse
import pandas as pd
from termcolor import colored
import time 

from transformers import AutoTokenizer

from src.exp2_rlmf.utils.data_utils import get_dataset
from src.exp0_baseline.metrics.faithfulness import get_faithfulness

from huggingface_hub import login
login(os.getenv("HF_TOKEN"))

def get_gold_faithfulness_scores(dataset, dev_mode, sampled_answers_path, dataset_name, sys_prompt_id, output_dir, slice):

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
    ### Expand Answers + Sampled Responses
    ############################## 
    print(colored("Computing gold faithfulnesses for all sampled answers...", "cyan"))
    
    # Process all sampled answers for all questions
    all_answers = []
    all_questions_expanded = []
    all_sampled_lists_expanded = []
    
    for q, sampled_list in zip(questions, sampled_answers_lists):
        for answer in sampled_list:
            all_answers.append(answer)
            all_questions_expanded.append(q)
            all_sampled_lists_expanded.append(sampled_list)
    
    ############################## 
    ### Predict Scores
    ############################## 
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "checkpoints", f"faithfulness_checkpoint_{dataset_name}_sys{sys_prompt_id}_{slice}.pkl")

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        start_idx = ckpt["idx"] + 1
        f_scores = ckpt["f_scores"]
        avg_gold_confs = ckpt["avg_gold_confs"]
        avg_decs = ckpt["avg_decs"]
        extracted_assertions = ckpt["extracted_assertions"]
        print(f"Resuming from checkpoint at index {start_idx}...")
    else:
        start_idx = 0
        f_scores, avg_gold_confs, avg_decs, extracted_assertions = [], [], [], []
        print(f"No checkpoint found, starting fresh...")

    start_time = time.time()
    for idx, (pred, sampled_answers_list) in enumerate(zip(all_answers[start_idx:], all_sampled_lists_expanded[start_idx:]), start=start_idx):

        elapsed = int(time.time() - start_time)
        hours, rem = divmod(elapsed, 3600)
        mins, secs = divmod(rem, 60)
        print(f"[{hours:02d} hr {mins:02d} min {secs:02d} s] Getting faithfulness for sample index {idx} of {len(all_answers)}")
        
        f_score, avg_conf_score, avg_dec_score, assertions, _ = get_faithfulness(answer=pred, sampled_answers=sampled_answers_list, confidence_score=None)
        f_scores.append(f_score)
        avg_gold_confs.append(avg_conf_score)
        avg_decs.append(avg_dec_score)
        extracted_assertions.append(assertions)

        with open(checkpoint_path, "wb") as f:
            pickle.dump({"idx": idx, "f_scores": f_scores, "avg_gold_confs": avg_gold_confs, "avg_decs": avg_decs, "extracted_assertions": extracted_assertions}, f)

    ############################## 
    ### Create DataFrames
    ############################## 
    # Create raw scores dataframe (one row per answer)
    raw_scores_df = pd.DataFrame({
        "raw_prompt": all_questions_expanded,
        "answer": all_answers,
        "gold_f_score": f_scores,
        "gold_conf": avg_gold_confs,
        "dec_score": avg_decs,
        "extracted_assertions": extracted_assertions,
    })
    
    # Create aggregated scores dataframe (one row per question)
    scores_aggr_over_sampled_answers_df = raw_scores_df.groupby("raw_prompt").agg({
        "gold_f_score": "mean",
        "gold_conf": "mean",
        "dec_score": "mean",
    }).reset_index()

    scores_first_df = raw_scores_df.groupby("raw_prompt", as_index=False).first()

    return raw_scores_df, scores_aggr_over_sampled_answers_df, scores_first_df

def main(args):

    # Create Output Dir
    output_dir = os.path.join(f"./exp2_rlmf/b_metacog_data_selection/score_dfs", args.model_name.replace("/", "_").replace("-", "_"))
    os.makedirs(output_dir, exist_ok=True)

    ### Load Tokenizer
    print(colored("Loading tokenizer...", "cyan"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    ### Load Data
    print(colored("Loading data...", "cyan"))
    train_dataset, _, _ = get_dataset(
        dataset_names=[args.dataset_name], 
        tokenizer=tokenizer, 
        num_train_samples=None, # use all
        sys_prompt_id=0,
        include_raw_prompts=True,
        inference=False,
    )

    ### Get Gold Faithfulness Scores
    print(colored(f"Getting gold faithfulness scores for training data...", "cyan"))
    raw_scores_df_train, scores_aggr_over_sampled_answers_df_train, scores_first_df_train = get_gold_faithfulness_scores(train_dataset, args.dev_mode, args.sampled_answers_path, args.dataset_name, args.sys_prompt_id, output_dir, args.slice)

    ### Save Scores
    print(colored(f"Saving to dataframe...", "cyan"))

    output_path = os.path.join(output_dir, f"faithfulness_scores_per_sampled_answer_{args.dataset_name}_train_sys{args.sys_prompt_id}_{args.slice}.csv")
    raw_scores_df_train.to_csv(output_path)

    output_path_final = os.path.join(output_dir, f"faithfulness_scores_aggr_over_sampled_answers_{args.dataset_name}_train_sys{args.sys_prompt_id}_{args.slice}.csv")
    scores_aggr_over_sampled_answers_df_train.to_csv(output_path_final)
    
    output_path_final_first = os.path.join(output_dir, f"faithfulness_scores_for_first_sampled_answer_{args.dataset_name}_train_sys{args.sys_prompt_id}_{args.slice}.csv")
    scores_first_df_train.to_csv(output_path_final_first)

    print(colored("Saved final dfs to...", "green"), output_path_final+colored("!", "green"))

def parse_args():
    parser = argparse.ArgumentParser()

    ### MODEL ARGS
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--sys_prompt_id", type=int)

    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--dev_mode", action="store_true", default=False)
    parser.add_argument("--sampled_answers_path", type=str)

    parser.add_argument('--slice', type=int, choices=[0,1,2,3])

    args = parser.parse_args()
    return args
 
if __name__ == "__main__":
    args = parse_args()
    main(args)