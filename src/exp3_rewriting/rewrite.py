import os
import json
import time, pickle
import ast
import argparse
import random
import numpy as np
import pandas as pd
from typing import List
from termcolor import colored

from openai import OpenAI
from google.genai import types
from google import genai

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

from src.exp0_baseline.utilities.utils import get_cmfg_mfg, get_cmfg_star
from src.exp0_baseline.metrics.faithfulness import get_faithfulness
from src.exp2_rlmf.d_evaluate.inference import run_extract_sentences_with_confidence
from src.exp3_rewriting.prompts import *

def get_gpt_response(rewrite_model_name, sys_prompt, user_prompt, max_output_tokens):
    response = openai_client.chat.completions.create(
        model=rewrite_model_name,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        n=1,
        max_completion_tokens=max_output_tokens,
        temperature=0.5,
        # stop="####",
        reasoning_effort="minimal",
    ).choices
    assert(len(response)==1)
    extracted_response = response[0].message.content
    if "####" in extracted_response: 
        return extracted_response.split("####")[0].strip()
    return extracted_response 

def get_gemini_response(rewrite_model_name, sys_prompt, user_prompt, max_output_tokens):

    try:
        response = gemini_client.models.generate_content(
            model=rewrite_model_name,
            config=types.GenerateContentConfig(
                system_instruction=sys_prompt,
                candidate_count=1,
                max_output_tokens=max_output_tokens,
                temperature=0.5,
                stop_sequences=['####'],
                # thinking_config=types.ThinkingConfig(thinking_budget=0)
            ),
            contents=user_prompt,
        ).text
    except Exception as e: 
        print(colored("Generation error:", "red"), e)
        response = ""

    # Process Response
    if response is None: 
        print(colored("Response is None, setting to empty string", "red"))
        response = ""
    if "####" in response: 
        return response.split("####")[0].strip()
    return response 

def get_options_str(markers_df_path, conf_float_or_list, max_markers_per_bin):
    
    df = pd.read_csv(markers_df_path)
    df.markers = df.markers.apply(ast.literal_eval)
    
    if type(conf_float_or_list)==float:  # get unique possible markers for this float
        markers = []
        for _, row in df.iterrows():

            # If confidence in the range start/end
            if row['range_start']<conf_float_or_list<=row['range_end']:
                markers += random.sample(row['markers'], len(row['markers']))
                # break   # break since found the range?

        markers = set(markers)  # get unique markers
        if max_markers_per_bin is not None:
            markers = random.sample(list(markers), min(max_markers_per_bin, len(markers)))
        options_str = ", ".join(markers)   # convert to comma separated list

    elif type(conf_float_or_list)==list: # get unique possible markers per float
        marker_strs = []
        
        for conf in sorted(set(conf_float_or_list)): # for each unique conf scr
            for _, row in df.iterrows():  # check if in range
                
                # If confidence in the range start/end
                if row['range_start']<conf<=row['range_end']: 
                    marker_str = ", ".join(set(random.sample(row['markers'], len(row['markers']))))
                    marker_strs.append(
                        f"CONFIDENCE {row['range_start']}-{row['range_end']}: {marker_str}"
                    )   # 2 decimals

        marker_strs = list(set(marker_strs))
        if max_markers_per_bin is not None:
            marker_strs = random.sample(marker_strs, min(max_markers_per_bin, len(marker_strs)))
        options_str = "\n".join(marker_strs)
        # options_str = "\n".join(set(marker_strs))
    
    else: 
        raise TypeError("Input to `get_options_str` must be a float or list of floats:", type(conf_float_or_list))
    
    return options_str

def rewrite_easy(markers_df_path, max_markers_per_bin, rewrite_model_name, generation_fxn, max_output_tokens, style_descrip, questions, answers, sentences_per_response, pred_confs_per_response, args, raw_only=False) -> List[str]:

    ### Prepare Inputs -- List[Tuple[str,float]]
    s_c_pairs = []
    for sent_list, conf_list in zip(sentences_per_response, pred_confs_per_response):
        temp = []
        for s, c in zip(sent_list, conf_list):
            if s=="no output":
                continue 
            if c==None:
                temp.append((s, -1))
            else: 
                temp.append((s, c))
        s_c_pairs.append(temp)
    assert(len(s_c_pairs)==len(questions))

    ### Run Rewriting
    checkpoint_path = os.path.join(args.output_dir, f'_chkpt_rewr{args.output_modifier}.json') if args.output_dir is not None else args.preds_path.replace("preds", f"chkpt_rewr").replace(".json", f"{args.output_modifier}.pkl").replace("test_", "linguistic_").replace("linguistic_results", "test_results")

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        start_idx = ckpt["idx"] + 1
        rewritten_sentences_per_sample = ckpt['rewritten_sentences_per_sample']
        sentences_per_sample = ckpt['sentences_per_sample']
        print(f"Resuming from rewriting step checkpoint at index {start_idx}...")
    else: 
        print(f"No checkpoint found for rewriting step, starting fresh...")
        start_idx = 0
        rewritten_sentences_per_sample = []
        sentences_per_sample = []

    # For Each Response (Sample)
    # for pairs_list in s_c_pairs:
    for idx, (q, pairs_list) in enumerate(zip(questions[start_idx:], s_c_pairs[start_idx:]), start=start_idx):

        print(f"On index {idx}...")

        # If No Sentences Skip
        if len(pairs_list)==0:
            rewritten_sentences_per_sample.append("")
            sentences_per_sample.append("")
        else: 
            cur_sample_rewritten = []
            cur_sample_to_rewrite = []

            # For Each Sentence & Associated Confidence
            for (s, c) in pairs_list:

                cur_sample_to_rewrite.append(f"{s} <conf> {c} </conf>")

                # Put Sentence As-Is if No Confidence
                if c==-1:
                    cur_sample_rewritten.append(s)
                    
                else:
                    dec_options = get_options_str(markers_df_path, c, max_markers_per_bin)
                    s = s.replace("\n", " ")

                    # Format Rewriting Prompt
                    if style_descrip!=None:
                        sys_prompt = EASY_SYS_PROMPT_WITH_STYLE
                        user_prompt = f"SENTENCE: {s}\nCONFIDENCE: {c}\nOPTIONS: {dec_options}\nTARGET STYLE: {style_descrip}\nREWRITTEN SENTENCE:"
                    else:
                        sys_prompt = EASY_SYS_PROMPT
                        user_prompt = f"SENTENCE: {s}\nCONFIDENCE: {c}\nOPTIONS: {dec_options}\nREWRITTEN SENTENCE:"
                    
                    # Get Rewritten Sentence
                    cur_sample_rewritten.append(
                        generation_fxn(
                            rewrite_model_name=rewrite_model_name,
                            sys_prompt=sys_prompt,
                            user_prompt=user_prompt,
                            max_output_tokens=max_output_tokens,
                        )
                    )
            rewritten_sentences_per_sample.append(cur_sample_rewritten)
            sentences_per_sample.append(cur_sample_to_rewrite)

        with open(checkpoint_path, "wb") as f:
            pickle.dump({
                "idx": idx, 
                "rewritten_sentences_per_sample": rewritten_sentences_per_sample,
                "sentences_per_sample": sentences_per_sample,
            }, f)

    if raw_only:
        return rewritten_sentences_per_sample, sentences_per_sample

    original_answers = [
        " ".join(orig_sent_list)    # Join rewritten sentences together
        for orig_sent_list in sentences_per_sample
    ]
    rewritten_answers = [
        " ".join(rewr_sent_list)    # Join rewritten sentences together
        for rewr_sent_list in rewritten_sentences_per_sample
    ]
    return rewritten_answers, original_answers

def rewrite_iterative(markers_df_path, max_markers_per_bin, rewrite_model_name, generation_fxn, max_output_tokens, style_descrip, questions, answers, sentences_per_response, pred_confs_per_response, args) -> List[str]:
    
    ### First Run Easy Rewriting
    rewritten_answers, _ = rewrite_easy(markers_df_path, max_markers_per_bin, rewrite_model_name, generation_fxn, max_output_tokens, None, questions, answers, sentences_per_response, pred_confs_per_response, args, raw_only=False)
    
    ### Prepare Revision Prompts & Run Iterative Rewriting
    revised_answers = []
    orig_answers = []

    # For Each Sample
    for idx, (q, s_list, c_list, rewr_answer) in enumerate(zip(questions, sentences_per_response, pred_confs_per_response, rewritten_answers)):

        print(f"On iter step index {idx}...")

        # Prepare to Format Original Answer
        orig_answer_decomposed = []
        relevant_c_list = []
        for s, c in zip(s_list, c_list):
            # Skip if Empty Answer
            if s=="no output":
                continue 
            # Default to Confidence 1.0 if Not Assigned/Error
            if c==None:
                orig_answer_decomposed.append(f"<sentence>{s}</sentence><confidence>{1.0}</confidence>")
                relevant_c_list.append(1.0)
            # Else Use Raw Sentence + Confidence Score
            else: 
                orig_answer_decomposed.append(f"<sentence>{s}</sentence><confidence>{c}</confidence>")
                relevant_c_list.append(c)
        # Skip if Invalid Original Answer
        if len(orig_answer_decomposed)==0:
            revised_answers.append("")
            orig_answers.append("")
        # elif len(orig_answer_decomposed)==1:
        #     revised_answers.append(rewr_answer)
        # Else Run Iterative Rewrite
        else: 
            orig_answer = " ".join(orig_answer_decomposed).replace("\n", " ")
            dec_options_str = get_options_str(markers_df_path, relevant_c_list, max_markers_per_bin)
            if style_descrip!=None:
                sys_prompt = ITERATIVE_SYS_PROMPT_WITH_STYLE
                user_prompt = f"QUESTION: {q}\nORIGINAL ANSWER: {orig_answer}\nREVISED ANSWER: {rewr_answer}\nOPTIONS: {dec_options_str}\nTARGET STYLE: {style_descrip}\nFINAL REVISED ANSWER:"
            else: 
                sys_prompt = ITERATIVE_SYS_PROMPT
                user_prompt = f"QUESTION: {q}\nORIGINAL ANSWER: {orig_answer}\nREVISED ANSWER: {rewr_answer}\nOPTIONS: {dec_options_str}\nFINAL REVISED ANSWER:"
            revised_answers.append(
                generation_fxn(
                    rewrite_model_name=rewrite_model_name,
                    sys_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    max_output_tokens=max_output_tokens,
                )
            )
            orig_answers.append(orig_answer)

    return revised_answers, orig_answers

def rewrite_all(markers_df_path, max_markers_per_bin, rewrite_model_name, generation_fxn, max_output_tokens, style_descrip, questions, answers, sentences_per_response, pred_confs_per_response) -> List[str]:
    
    ### Prepare Rewriting Prompts & Run Full Rewrite
    revised_answers = []
    orig_answers = []

    # For Each Sample
    for idx, (q, s_list, c_list) in enumerate(zip(questions, sentences_per_response, pred_confs_per_response)):
        
        print(f"On all mode index {idx}...")

        orig_answer_decomposed = []
        relevant_c_list = []
        for s, c in zip(s_list, c_list):
            # Skip if Empty Answer
            if s=="no output":
                continue 
            # Default to Confidence 1.0 if Not Assigned/Error
            if c==None:
                orig_answer_decomposed.append(f"<sentence>{s}</sentence><confidence>{1.0}</confidence>")
                relevant_c_list.append(1.0)
            # Else Use Raw Sentence + Confidence Score
            else: 
                orig_answer_decomposed.append(f"<sentence>{s}</sentence><confidence>{c}</confidence>")
                relevant_c_list.append(c)
        # Skip if Invalid Original Answer
        if len(orig_answer_decomposed)==0:
            revised_answers.append("")
            orig_answers.append("")
        # Else Run Full Rewrite
        else: 
            orig_answer = " ".join(orig_answer_decomposed).replace("\n", " ")
            dec_options_str = get_options_str(markers_df_path, relevant_c_list, max_markers_per_bin)
            if style_descrip!=None:
                sys_prompt = ALL_SYS_PROMPT_WITH_STYLE
                user_prompt = f"ORIGINAL ANSWER: {orig_answer}\nOPTIONS: {dec_options_str}\nTARGET STYLE: {style_descrip}\nREVISED ANSWER:"
            else: 
                sys_prompt = ALL_SYS_PROMPT
                user_prompt = f"ORIGINAL ANSWER: {orig_answer}\nOPTIONS: {dec_options_str}\nREVISED ANSWER:"
            revised_answers.append(
                generation_fxn(
                    rewrite_model_name=rewrite_model_name,
                    sys_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    max_output_tokens=max_output_tokens,
                )
            )
            orig_answers.append(orig_answer)

    return revised_answers, orig_answers


def main():

    ### Parse Run Arguments
    args = parse_args() 
    if args.output_dir is not None: 
        os.makedirs(args.output_dir, exist_ok=True)

    ### Read in Model Outputs
    print(colored(f"Loading raw predictions from", "cyan"), args.preds_path+colored("...", "cyan"))
    with open(args.preds_path, 'r') as file:
        preds_dict = json.load(file)
    questions = preds_dict['raw_prompts']
    answers = preds_dict['answers']
    if args.dev_mode:
        questions = questions[:100]
        answers = answers[:100]

    ### Parse Sentences & Confidence Scores
    print(colored("Extracting sentences and confidences per response...", "cyan"))
    sentences_per_response, pred_confs_per_response, _ = run_extract_sentences_with_confidence(answers, format_version='new')
    assert(len(sentences_per_response)==len(pred_confs_per_response))
    
    ### Run Rewriting
    results_save_path = os.path.join(args.output_dir, f'preds{args.output_modifier}.json') if args.output_dir is not None else args.preds_path.replace(".json", f"{args.output_modifier}.json").replace("test_", "linguistic_").replace("linguistic_results", "test_results")

    if os.path.exists(results_save_path):
        print(colored(f"Found rewriting results by {args.rewrite_model_name}, loading...", "cyan"))
        with open(results_save_path) as f:
            preds_dict = json.load(f)
        rewritten_answers = preds_dict['rewritten_answers']
    else: 
        print(colored(f"Running rewriting by model {args.rewrite_model_name}...", "cyan"))

        generation_fxn = get_gpt_response if "gemini" not in args.rewrite_model_name else get_gemini_response

        if args.rewrite_mode=="easy":
            rewritten_answers, orig_answers = rewrite_easy(args.markers_df_path, args.max_markers_per_bin, args.rewrite_model_name, generation_fxn, args.max_output_tokens, args.style_descrip, questions, answers, sentences_per_response, pred_confs_per_response, args)
        elif args.rewrite_mode=="iter":
            rewritten_answers, orig_answers = rewrite_iterative(args.markers_df_path, args.max_markers_per_bin, args.rewrite_model_name, generation_fxn, args.max_output_tokens, args.style_descrip, questions, answers, sentences_per_response, pred_confs_per_response, args)
        elif args.rewrite_mode=="all":
            rewritten_answers, orig_answers = rewrite_all(args.markers_df_path, args.max_markers_per_bin, args.rewrite_model_name, generation_fxn, args.max_output_tokens, args.style_descrip, questions, answers, sentences_per_response, pred_confs_per_response)

        assert(len(rewritten_answers)==len(questions))

        ### Save Results
        preds_dict['rewrite_mode'] = args.rewrite_mode
        preds_dict['rewrite_model'] = args.rewrite_model_name
        preds_dict['rewrite_style_descrip'] = args.style_descrip if args.style_descrip is not None else ""

        preds_dict['markers_df_path'] = args.markers_df_path
        preds_dict['max_markers_per_bin'] = args.max_markers_per_bin

        preds_dict['orig_answers'] = orig_answers
        preds_dict['rewritten_answers'] = rewritten_answers

        with open(results_save_path, 'w') as f:
            f.write(json.dumps(preds_dict, indent=4))
        print(colored("Saved rewriting results as json!", "green"))

    ### Compute cMFG if Desired
    if args.score_faithfulness:

        metrics_save_path = os.path.join(args.output_dir, f'scores{args.output_modifier}.json') if args.output_dir is not None else args.preds_path.replace("preds", f"scores").replace(".json", f"{args.output_modifier}.json").replace("test_", "linguistic_").replace("linguistic_results", "test_results")

        if os.path.exists(metrics_save_path):
            print(colored(f"Already scored faithfulness! Results saved at {metrics_save_path}", "cyan"))
        
        else: 

            print(colored("Scoring faithfulness...", "cyan"))

            with open(args.preds_path.replace("preds", "scores")) as f:
                scores_dict = json.load(f)
            gold_confs_per_response = scores_dict['gold_confs_per_response']
            avg_gold_confidences = [
                -1.0 if not (vals := [c for c in conf_list if c is not None]) else np.mean(vals)
                for conf_list in gold_confs_per_response
            ]
            # sampled_answers_lists = preds_dict['sampled_answers']

            checkpoint_path = os.path.join(args.output_dir, f'_chkpt_scr{args.output_modifier}.json') if args.output_dir is not None else args.preds_path.replace("preds", f"chkpt_scr").replace(".json", f"{args.output_modifier}.pkl").replace("test_", "linguistic_").replace("linguistic_results", "test_results")

            if os.path.exists(checkpoint_path) and args.rescore==False:
                with open(checkpoint_path, "rb") as f:
                    ckpt = pickle.load(f)
                start_idx = ckpt["idx"] + 1
                f_scores_with_assertions = ckpt['f_scores_with_assertions']
                f_scores_without_assertions = ckpt['f_scores_without_assertions']
                avg_conf_scores_with_assertions = ckpt['avg_conf_scores_with_assertions']
                avg_conf_scores_without_assertions = ckpt['avg_conf_scores_without_assertions']
                avg_dec_scores_with_assertions = ckpt['avg_dec_scores_with_assertions']
                avg_dec_scores_without_assertions = ckpt['avg_dec_scores_without_assertions']
                extracted_assertions = ckpt['extracted_assertions']
                conf_responses_all_with_assertions = ckpt['consistency_judgments_with_assertions']
                conf_responses_all_without_assertions = ckpt['consistency_judgments_without_assertions']
                print(f"Resuming from scoring checkpoint at index {start_idx}...")

            else:
                start_idx = 0
                f_scores_with_assertions, f_scores_without_assertions, avg_conf_scores_with_assertions, avg_conf_scores_without_assertions, avg_dec_scores_with_assertions, avg_dec_scores_without_assertions, extracted_assertions, conf_responses_all_with_assertions, conf_responses_all_without_assertions = [], [], [], [], [], [], [], [], []
                print(f"No checkpoint found for scoring, starting fresh...")
            
            ### Score Faithfulness Per Response
            start_time = time.time()
            for idx, (q, answer, gold_conf) in enumerate(zip(questions[start_idx:], rewritten_answers[start_idx:], avg_gold_confidences[start_idx:]), start=start_idx):

                elapsed = int(time.time() - start_time)
                hours, rem = divmod(elapsed, 3600)
                mins, secs = divmod(rem, 60)
                print(colored(f"[{hours:02d} hr {mins:02d} min {secs:02d} s]", "yellow") + f" Getting faithfulness for sample index {idx} of {len(questions[start_idx:])}")

                # print(colored("With assertion...", "magenta"))
                f_score_with_assertions, avg_conf_score_with_assertions, avg_dec_score_with_assertions, assertions, conf_responses_with_assertions = get_faithfulness(answer=answer, sampled_answers=None, confidence_score=gold_conf)

                # print(colored("Without assertions...", "magenta"))
                f_score_without_assertions, avg_conf_score_without_assertions, avg_dec_score_without_assertions, _, conf_responses_without_assertions = get_faithfulness(answer=answer, sampled_answers=None, confidence_score=gold_conf)

                f_scores_with_assertions.append(f_score_with_assertions)
                f_scores_without_assertions.append(f_score_without_assertions)

                avg_conf_scores_with_assertions.append(avg_conf_score_with_assertions)
                avg_conf_scores_without_assertions.append(avg_conf_score_without_assertions)

                avg_dec_scores_with_assertions.append(avg_dec_score_with_assertions)
                avg_dec_scores_without_assertions.append(avg_dec_score_without_assertions)

                extracted_assertions.append(assertions)

                conf_responses_all_with_assertions.append(conf_responses_with_assertions)
                conf_responses_all_without_assertions.append(conf_responses_without_assertions)

                with open(checkpoint_path, "wb") as f:
                    pickle.dump({
                        "idx": idx, 
                        "f_scores_with_assertions": f_scores_with_assertions,
                        "f_scores_without_assertions": f_scores_without_assertions,
                        "avg_conf_scores_with_assertions": avg_conf_scores_with_assertions,
                        "avg_conf_scores_without_assertions": avg_conf_scores_without_assertions,
                        "avg_dec_scores_with_assertions": avg_dec_scores_with_assertions,
                        "avg_dec_scores_without_assertions": avg_dec_scores_without_assertions,
                        "extracted_assertions": extracted_assertions,
                        "consistency_judgments_with_assertions": conf_responses_all_with_assertions,
                        "consistency_judgments_without_assertions": conf_responses_all_without_assertions,
                    }, f)
            
            scoring_artifacts_dict = {
                "f_scores_with_assertions": f_scores_with_assertions,
                "f_scores_without_assertions": f_scores_without_assertions,
                "avg_conf_scores_with_assertions": avg_conf_scores_with_assertions,
                "avg_conf_scores_without_assertions": avg_conf_scores_without_assertions,
                "avg_dec_scores_with_assertions": avg_dec_scores_with_assertions,
                "avg_dec_scores_without_assertions": avg_dec_scores_without_assertions,
                "extracted_assertions": extracted_assertions,
                "consistency_judgments_with_assertions": conf_responses_all_with_assertions,
                "consistency_judgments_without_assertions": conf_responses_all_without_assertions,
            }

            artifacts_save_path = os.path.join(args.output_dir, f'scoring_artifacts{args.output_modifier}.json') if args.output_dir is not None else args.preds_path.replace("preds", f"scoring_artifacts").replace(".json", f"{args.output_modifier}.json").replace("test_", "linguistic_").replace("linguistic_results", "test_results")
            
            with open(artifacts_save_path, 'w') as f:
                f.write(json.dumps(scoring_artifacts_dict, indent=4))
            
            print(colored("Saved faithfulness results to", "green"), artifacts_save_path+colored("!", "green"))

            ### Compute cMFG Score + Stats
            print(colored("Computing cMFG...", "cyan"))
            cmfg_with_assertions, mfg_with_assertions, stats_with_assertions = get_cmfg_mfg(
                f_scores_with_assertions, 
                avg_conf_scores_with_assertions, 
                num_bins=10,
            )
            cmfg_without_assertions, mfg_without_assertions, stats_without_assertions = get_cmfg_mfg(
                f_scores_without_assertions, 
                avg_conf_scores_without_assertions, 
                num_bins=10,
            )
            cmfg_star_with_assertions, var_cmfg_star_with_assertions, star_stats_with_assertions = get_cmfg_star(
                f_scores_with_assertions, 
                avg_conf_scores_with_assertions, 
                num_bins=10,
            )
            cmfg_star_without_assertions, var_cmfg_star_without_assertions, star_stats_without_assertions = get_cmfg_star(
                f_scores_without_assertions, 
                avg_conf_scores_without_assertions, 
                num_bins=10,
            )
            metrics_summary = {
                'cmfg_linguistic_with_assertions': cmfg_with_assertions,
                'mfg_linguistic_with_assertions': mfg_with_assertions,
                'stats_linguistic_with_assertions': stats_with_assertions,
                'cmfg_linguistic_without_assertions': cmfg_without_assertions,
                'mfg_linguistic_without_assertions': mfg_without_assertions,
                'stats_linguistic_without_assertions': stats_without_assertions,
                'cmfg_star_linguistic_with_assertions': cmfg_star_with_assertions,
                'cmfg_star_linguistic_without_assertions': cmfg_star_without_assertions,
                'var_cmfg_star_with_assertions': var_cmfg_star_with_assertions,
                'var_cmfg_star_without_assertions': var_cmfg_star_without_assertions,
            }

            ### Save cMFG Results
            with open(args.preds_path.replace("preds", "scores"), 'r') as file:
                metrics_dict = json.load(file)
            metrics_summary.update(metrics_dict)

            avg_gold_confidences = [
                -1. if not (vals := [c for c in conf_list if c is not None]) else np.mean(vals)
                for conf_list in metrics_dict['gold_confs_per_response']
            ]
            avg_gold_confidences = np.array(avg_gold_confidences, dtype=np.float64)
            metrics_summary['avg_gold_confidences'] = avg_gold_confidences.tolist()
                    
            with open(metrics_save_path, 'w') as f:
                f.write(json.dumps(metrics_summary, indent=4))

            print(colored("Saved cMFG metrics to updated JSON at", "green"), metrics_save_path+colored("!", "green"))

def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--rewrite_mode", type=str, default="easy", choices=["easy", "iter", "iterative", "all"])
    parser.add_argument("--rewrite_model_name", type=str, default="gemini-2.5-flash")
    parser.add_argument("--style_descrip", type=str, default=None)

    parser.add_argument("--preds_path", type=str)
    parser.add_argument("--markers_df_path", type=str)
    parser.add_argument("--max_markers_per_bin", type=int, default=10)

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_modifier", type=str, default=None)

    parser.add_argument("--max_output_tokens", type=int, default=2000)
    parser.add_argument("--dev_mode", default=False, action="store_true")
    
    parser.add_argument("--score_faithfulness", default=False, action="store_true")
    parser.add_argument("--rescore", default=False, action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    main()
