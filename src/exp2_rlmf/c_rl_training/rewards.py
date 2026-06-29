import json
import re
import math
import numpy as np

import wandb

from src.exp2_rlmf.utils.utils import llm_eval, get_sentence_internal_confidence, extract_sentences_with_confidence_new, get_mc_scores
from .rewards import *

from netcal.metrics import ECE
ece = ECE(bins=10)

def count_empty_sentences(completion):
    empty_count = 0
    
    # Find all tag positions
    tags = []
    for match in re.finditer(r"</?sentence>", completion):
        tags.append((match.start(), match.end(), match.group()))
    
    used_closes = set()  # Track which closing tags we've already matched
    
    for i, (start, end, tag) in enumerate(tags):
        if tag == "<sentence>":
            # Find the next unused </sentence>
            for j in range(i + 1, len(tags)):
                if tags[j][2] == "</sentence>" and j not in used_closes:
                    # Extract content between this pair
                    content = completion[end:tags[j][0]]
                    # Remove all tags
                    content_no_tags = re.sub(r"<[^>]+>", "", content)
                    # Remove punctuation and whitespace, check if any alphanumeric chars remain
                    content_cleaned = re.sub(r"[^\w]", "", content_no_tags, flags=re.UNICODE)
                    
                    if not content_cleaned:
                        empty_count += 1
                    used_closes.add(j)
                    break
    
    return empty_count

def count_empty_confidences(completion):
    empty_count = 0
    
    # Find all tag positions
    tags = []
    for match in re.finditer(r"</?confidence>", completion):
        tags.append((match.start(), match.end(), match.group()))
    
    used_closes = set()  # Track which closing tags we've already matched
    
    for i, (start, end, tag) in enumerate(tags):
        if tag == "<confidence>":
            # Find the next unused </confidence>
            for j in range(i + 1, len(tags)):
                if tags[j][2] == "</confidence>" and j not in used_closes:
                    # Extract content between this pair
                    content = completion[end:tags[j][0]]
                    # Remove all tags
                    content_no_tags = re.sub(r"<[^>]+>", "", content)
                    # Remove punctuation and whitespace, check if any alphanumeric chars remain
                    content_cleaned = re.sub(r"[^\w]", "", content_no_tags, flags=re.UNICODE)
                    
                    if not content_cleaned:
                        empty_count += 1
                    used_closes.add(j)
                    break
    
    return empty_count

def count_valid_sentences(completion):
    valid_count = 0
    
    # Find all tag positions
    tags = []
    for match in re.finditer(r"</?sentence>", completion):
        tags.append((match.start(), match.end(), match.group()))
    
    used_closes = set()  # Track which closing tags we've already matched
    
    for i, (start, end, tag) in enumerate(tags):
        if tag == "<sentence>":
            # Find the next unused </sentence>
            for j in range(i + 1, len(tags)):
                if tags[j][2] == "</sentence>" and j not in used_closes:
                    # Extract content between this pair
                    content = completion[end:tags[j][0]]
                    
                    # Check for nested sentence tags
                    if '<sentence>' in content or '</sentence>' in content:
                        used_closes.add(j)
                        break  # Invalid due to nested tags
                    
                    # Remove all other tags (like <confidence>, etc.)
                    content_no_tags = re.sub(r"<[^>]+>", "", content)
                    # Remove punctuation and whitespace, check if any alphanumeric chars remain
                    content_cleaned = re.sub(r"[^\w]", "", content_no_tags, flags=re.UNICODE)
                    
                    if content_cleaned:  # Has alphanumeric content
                        valid_count += 1
                    
                    used_closes.add(j)
                    break
    
    return valid_count

def format_reward_func_new(completions, format_version, sys9=False, sys_11_12_13=False, metascore_as_percentage=False, **kwargs) -> list[float]:
    """
    Strict format reward function for sentence + confidence tag pairs.
    
    Requirements:
    - Each <sentence>...</sentence> must be followed by <confidence>X</confidence>
    - Confidence tags must contain ONLY a valid number (0.0 to 1.0)
    - All tags must be properly paired and matched
    - At least one valid sentence-confidence pair must exist
    
    Returns:
    - 0.0 if format is perfect
    - -1.0 if format has any violations
    """
    
    # # Pattern for valid sentence-confidence pair
    # # Matches: <sentence>...</sentence><confidence>NUMBER</confidence>
    # sentence_confidence_pattern = (
    #     r"<sentence>.*?</sentence>\s*"
    #     r"<confidence>\s*([01](?:\.\d+)?|0?\.\d+)\s*</confidence>"
    # )
    
    # # Pattern for valid confidence value only (0.0 to 1.0)
    # valid_confidence_value = r"<confidence>\s*(0(\.\d*)?|1(\.0*)?)\s*</confidence>"

    # if "percentage" in format_version:
    #     valid_confidence_value = r"<confidence>\s*(\d{1,2}(\.\d*)?|100(\.0*)?|\.\d+)\s*</confidence>"
    #     sentence_confidence_pattern = (
    #         r"<sentence>.*?</sentence>\s*"
    #         r"<confidence>\s*(\d{1,2}(\.\d*)?|100(\.0*)?|\.\d+)\s*</confidence>"
    #     )
    if "percentage" in format_version:
        valid_confidence_value = r"<confidence>\s*(\d{1,2}(\.\d*)?|100(\.0*)?|\.\d+)\s*%?\s*</confidence>"
        sentence_confidence_pattern = (
            r"<sentence>.*?</sentence>\s*"
            r"<confidence>\s*(\d{1,2}(\.\d*)?|100(\.0*)?|\.\d+)\s*%?\s*</confidence>"
        )
    else:
        valid_confidence_value = r"<confidence>\s*(0(\.\d*)?|1(\.0*)?)\s*</confidence>"
        sentence_confidence_pattern = (
            r"<sentence>.*?</sentence>\s*"
            r"<confidence>\s*([01](?:\.\d+)?|0?\.\d+)\s*</confidence>"
        )
    
    scores = []
    for completion in completions:
        completion = completion[0]['content']

        if sys9:
            # Check 1: Must contain ####
            if "####" not in completion:
                scores.append(-1.0)
                continue
            
            # Check 2: No text after #### (only whitespace allowed)
            after_separator = completion.split("####", 1)[1]
            if after_separator.strip():  # If there's any non-whitespace text
                scores.append(-1.0)
                continue
            
            # Check 3: Tag immediately before #### must be </confidence>
            before_separator = completion.split("####", 1)[0]
            # Remove trailing whitespace and check if it ends with </confidence>
            if not before_separator.rstrip().endswith("</confidence>"):
                scores.append(-1.0)
                continue
        
        # Count all tags
        n_sentence_open = len(re.findall(r"<sentence>", completion))
        n_sentence_close = len(re.findall(r"</sentence>", completion))
        n_conf_open = len(re.findall(r"<confidence>", completion))
        n_conf_close = len(re.findall(r"</confidence>", completion))
        
        # Count valid confidence values (number only, in range 0-1)
        valid_conf_values = re.findall(valid_confidence_value, completion)
        n_valid_conf = len(valid_conf_values)
        
        # Count valid sentence-confidence pairs
        n_valid_pairs = len(re.findall(sentence_confidence_pattern, completion, re.DOTALL))
        
        # Check for text inside confidence tags (beyond just numbers)
        all_conf_content = re.findall(r"<confidence>(.*?)</confidence>", completion, re.DOTALL)
        has_text_in_conf = any(
            not re.match(r"^\s*([01](?:\.\d+)?|0?\.\d+)\s*$", content) 
            for content in all_conf_content
        )
        if "percentage" in format_version:
            n_conf_with_text = sum(1 for content in all_conf_content if content.strip() and not re.match(r"^\s*-?\d+(\.\d*)?\s*%?\s*$", content))
        else: 
            n_conf_with_text = sum(1 for content in all_conf_content if content.strip() and not re.match(r"^\s*-?\d+(\.\d*)?\s*$", content))

        # Check for metascore if sys_11_12_13 is True
        if sys_11_12_13:
            # Check only one metascore tag in entire completion
            all_metascore_matches = re.findall(r'<metascore>', completion)
            if len(all_metascore_matches) != 1:
                scores.append(-1.0)
                continue

            # Check only one metascore tag in entire completion
            all_metascore_close_matches = re.findall(r'</metascore>', completion)
            if len(all_metascore_close_matches) != 1:
                scores.append(-1.0)
                continue

            # # Check same # open and close metascore tags (only 1)
            # if len(all_metascore_matches)!=len(all_metascore_close_matches):
            #     scores.append(-1.0)
            #     continue

            # Find last </confidence> tag
            last_conf_idx = completion.rfind("</confidence>")
            if last_conf_idx == -1:
                scores.append(-1.0)
                continue
            
            # Get text after last </confidence>
            text_after_last_conf = completion[last_conf_idx + len("</confidence>"):]

            # Check if there's a metascore tag with valid range
            if metascore_as_percentage:
                metascore_pattern = r'<metascore>\s*(\d{1,2}(\.\d*)?|100(\.0*)?|\.\d+)\s*</metascore>'
            else:
                metascore_pattern = r'<metascore>\s*(0(\.\d*)?|1(\.0*)?)\s*</metascore>'
            
            # Check if there's a metascore tag
            metascore_match = re.search(metascore_pattern, text_after_last_conf)
            if not metascore_match:
                scores.append(-1.0)
                continue

            # Check nothing but whitespace between </confidence> and <metascore>
            text_between = text_after_last_conf[:metascore_match.start()]
            if any(c.strip().isalnum() for c in text_between):  # if nonpunc
            # if text_between.strip():  # If there's any non-whitespace
                scores.append(-1.0)
                continue

        # Perfect format conditions:
        # 1. At least one pair exists
        # 2. All tags are balanced (sentence and confidence)
        # 3. Number of sentence tags equals number of confidence tags
        # 4. All confidence tags contain valid values
        # 5. All pairs are properly formatted (sentence followed by confidence)
        # 6. No text inside confidence tags (only numbers)
        if (n_valid_pairs >= 1 and
            n_sentence_open == n_sentence_close and
            n_conf_open == n_conf_close and
            n_sentence_open == n_conf_open and
            n_valid_conf == n_conf_open and
            n_valid_pairs == n_sentence_open and
            n_conf_with_text==0):
            scores.append(1.0)
        else:
            scores.append(-1.0)
    
    return scores   # orig score 3.0

def approximate_format_reward_func_new(prompts, completions, format_version, sys9=False, sys_11_12_13=False, metascore_as_percentage=False, length_dirxn_used=False, **kwargs) -> list[float]:
    """
    Partial credit format reward function.
    
    Provides graduated penalties based on:
    - Presence of sentence-confidence pairs
    - Tag balance and matching
    - Proportion of valid pairs
    - Confidence values in valid range (0.0-1.0)
    - No extraneous text in confidence tags
    
    Returns:
    - 0.0 for perfect format
    - Negative scores proportional to format violations
    - -1.0 for severe violations (no pairs, completely mismatched tags)
    """

    def get_number_before_sentences(s):
        match = re.search(r'(\d+)\s*sentences', s.replace('–', '-'))
        matches = re.findall(r'(\d+)\s*sentences', s)
        return int(matches[-1]) if matches else None
    
    # if "percentage" in format_version:
    #     valid_confidence_value = r"<confidence>\s*(\d{1,2}(\.\d*)?|100(\.0*)?|\.\d+)\s*</confidence>"
    #     sentence_confidence_pattern = (
    #         r"<sentence>.*?</sentence>\s*"
    #         r"<confidence>\s*(\d{1,2}(\.\d*)?|100(\.0*)?|\.\d+)\s*</confidence>"
    #     )
    if "percentage" in format_version:
        valid_confidence_value = r"<confidence>\s*(\d{1,2}(\.\d*)?|100(\.0*)?|\.\d+)\s*%?\s*</confidence>"
        sentence_confidence_pattern = (
            r"<sentence>.*?</sentence>\s*"
            r"<confidence>\s*(\d{1,2}(\.\d*)?|100(\.0*)?|\.\d+)\s*%?\s*</confidence>"
        )
    else: 
        sentence_confidence_pattern = (
            r"<sentence>.*?</sentence>\s*"
            r"<confidence>\s*([01](?:\.\d+)?|0?\.\d+)\s*</confidence>"
        )
        valid_confidence_value = r"<confidence>\s*(0(\.\d*)?|1(\.0*)?)\s*</confidence>"

    if length_dirxn_used==True: 
        print(len(prompts), len(completions))
        assert(len(prompts)==len(completions))
        target_num_sents = [
            get_number_before_sentences(prompt[-1]['content'].rsplit("\n", 1)[1]) for prompt in prompts
        ]
    else: 
        target_num_sents = [-1]*len(completions)

    scores = []
    for target_length, completion in zip(target_num_sents, completions):
        completion = completion[0]['content']

        # Count all tags
        n_sentence_open = len(re.findall(r"<sentence>", completion))
        n_sentence_close = len(re.findall(r"</sentence>", completion))
        n_conf_open = len(re.findall(r"<confidence>", completion))
        n_conf_close = len(re.findall(r"</confidence>", completion))
        
        # Count valid elements
        valid_conf_values = re.findall(valid_confidence_value, completion)
        n_valid_conf = len(valid_conf_values)
        n_valid_pairs = len(re.findall(sentence_confidence_pattern, completion, re.DOTALL))
        
        # Check for text in confidence tags
        all_conf_content = re.findall(r"<confidence>(.*?)</confidence>", completion, re.DOTALL)
        if "percentage" in format_version:
            n_conf_with_text = sum(1 for content in all_conf_content if content.strip() and not re.match(r"^\s*-?\d+(\.\d*)?\s*%?\s*$", content))
        else:
            n_conf_with_text = sum(1 for content in all_conf_content if content.strip() and not re.match(r"^\s*-?\d+(\.\d*)?\s*$", content))

        # Check for empty tags
        empty_sentences = count_empty_sentences(completion)
        empty_confidences = count_empty_confidences(completion)
        
        # Severe violations: no pairs or completely mismatched tags
        if n_sentence_open==0 or n_valid_pairs == 0 or n_sentence_open != n_sentence_close or n_conf_open != n_conf_close or n_valid_pairs == 0 or (target_length!=-1 and max(n_sentence_open, n_sentence_close)+2 > target_length):    # length condition added 2/26
            if n_sentence_open==0:
                reward = -1.0
            elif n_sentence_open != n_sentence_close or n_conf_open != n_conf_close:
                reward = 0.
                if n_sentence_open != n_sentence_close:
                    reward += -0.5 * (1. - (count_valid_sentences(completion) / max(n_sentence_open, n_sentence_close))) 
                if n_conf_open != n_conf_close:
                    reward += - 0.5 * (1. - (n_valid_conf / max(n_conf_open, n_conf_close)))
            elif n_valid_pairs == 0:
                reward = -1.0
            else: 
                reward = -1.0
        else:
            # Calculate penalties for various issues
            penalties = []
            
            # Penalty for tag count mismatch (sentence vs confidence)
            if n_sentence_open != n_conf_open:
                tag_mismatch_penalty = abs(n_sentence_open - n_conf_open) / max(n_sentence_open, n_conf_open)
                penalties.append(tag_mismatch_penalty * -0.25)
            
            # Penalty for invalid confidence values
            if n_conf_open > 0:
                invalid_conf_penalty = (n_conf_open - n_valid_conf) / n_conf_open
                penalties.append(invalid_conf_penalty * -0.25)
            
            # Penalty for text in confidence tags
            if n_conf_open > 0 and n_conf_with_text > 0:
                text_penalty = n_conf_with_text / n_conf_open
                penalties.append(text_penalty * -0.2)
            
            # Penalty for unpaired elements (valid tags but not properly paired)
            expected_pairs = min(n_sentence_open, n_conf_open)
            if expected_pairs > 0:
                pairing_penalty = (expected_pairs - n_valid_pairs) / expected_pairs
                penalties.append(pairing_penalty * -0.3)
            
            # Penalty for empty sentence tags
            if n_sentence_open > 0 and empty_sentences > 0:
                empty_sentence_penalty = empty_sentences / n_sentence_open
                penalties.append(empty_sentence_penalty * -0.05)
            
            # Penalty for empty confidence tags
            if n_conf_open > 0 and empty_confidences > 0:
                empty_conf_penalty = empty_confidences / n_conf_open
                penalties.append(empty_conf_penalty * -0.05)
            
            # Total penalty (max magnitude 1.0)
            reward = sum(penalties)

            if sys9:
                if not ("####" in completion and not completion.split("####", 1)[1].strip() and completion.split("####", 1)[0].rstrip().endswith("</confidence>")):
                    reward -=0.5
        
        # Penalize metascore issues if needed
        if sys_11_12_13:
            metascore_penalties = []
            
            # Check for metascore tags
            n_metascore_open = len(re.findall(r'<metascore>', completion))
            n_metascore_close = len(re.findall(r'</metascore>', completion))
            
            # Severe: No metascore at all
            if n_metascore_open == 0:
                metascore_penalties.append(-1.0)

            # Severe: Multiple metascores
            elif n_metascore_open > 1:
                metascore_penalties.append(-1.0)

            # Moderate: Unbalanced metascore tags
            elif n_metascore_open != n_metascore_close:
                metascore_penalties.append(-0.5)
            else:
                # Metascore exists and is balanced, check position and validity
                last_conf_idx = completion.rfind("</confidence>")
                metascore_idx = completion.find("<metascore>")
                
                # Severe: Metascore not after last confidence
                if last_conf_idx == -1 or metascore_idx < last_conf_idx:
                    metascore_penalties.append(-1.0)
                else:
                    # Get text after last confidence
                    text_after_last_conf = completion[last_conf_idx + len("</confidence>"):]
                    
                    # Pattern for valid metascore
                    if metascore_as_percentage:
                        metascore_pattern = r'<metascore>\s*(\d{1,2}(\.\d*)?|100(\.0*)?|\.\d+)\s*</metascore>'
                    else:
                        metascore_pattern = r'<metascore>\s*(0(\.\d*)?|1(\.0*)?|\.\d+)\s*</metascore>'
                    
                    metascore_match = re.search(metascore_pattern, text_after_last_conf)
                    
                    if metascore_match:
                        # Check for text between </confidence> and <metascore>
                        text_between = text_after_last_conf[:metascore_match.start()]
                        if any(c.strip().isalnum() for c in text_between):  # if nonpunc
                        # if text_between.strip():
                            # Severe: Text between confidence and metascore
                            metascore_penalties.append(-1.0)
                    else:
                        # Moderate: Invalid metascore value
                        metascore_penalties.append(-0.5)
            
            # Cap metascore penalty contribution at -1.0
            reward += max(sum(metascore_penalties), -1.0)

        scores.append(reward)
        
    return scores

def run_extract_sentences_with_confidence(completions, format_version='old', get_meta_score_from_completions=False, metascore_as_percentage=False):

    # if output.startswith('\n\n') or output.startswith('"<|start_header_id|>assistant<|end_header_id|>\n\n'): 
    #     output = output.split('\n\n')[1] #take out the template head

    sentences_per_completion = []   # List of list of strs
    pred_confs_per_completion = []  # List of list of floats
    metascore_per_completion = []
    for completion in completions:
        completion = completion[0]['content']
        extracted_sentences_with_confidence = extract_sentences_with_confidence_new(completion, percentage=('percentage' in format_version), get_meta_score=get_meta_score_from_completions, metascore_as_percentage=metascore_as_percentage)
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

def correctness_and_factuality_reward_func(prompts, completions, targets, c_weight, f_weight, format_version, hostname, length_dirxn_used=False, **kwargs) -> list[float]:
    '''
    targets: a list of answers
    '''
    
    if c_weight>0 or f_weight>0:
        sentences_per_completion, pred_confs_per_completion, _ = run_extract_sentences_with_confidence(completions, format_version=format_version)
        extracted_responses = [
            " ".join(sent_list) for sent_list in sentences_per_completion
        ]
        correctness = llm_eval(prompts, targets, extracted_responses, hostname, length_dirxn_used=length_dirxn_used) # 1.0 if correct 0.0 else
        confidences = [
            -1. if not (vals := [c for c in conf_list if c is not None]) else np.mean(vals)
            for conf_list in pred_confs_per_completion
        ]
        all_correctness = np.array(correctness, dtype=np.float64)
        all_confidences = np.array(confidences, dtype=np.float64)

    if f_weight>0:
        factuality = 1. - (all_correctness - all_confidences) ** 2.
        mask = all_confidences == -1
        factuality[mask] = -1.

    if c_weight==0:
        all_correctness = np.array([0.]* len(completions))
    if f_weight==0:
        factuality = np.array([0.]* len(completions))

    return c_weight*all_correctness + f_weight*factuality  # correctness weighted 0.25 in original paper; overall higher is better for each metric

def grpo_faithfulness_reward(prompts, completions, reward_form, format_version, hostname, use_orig_confidence, faithfulness_reward_weight, mc_reward_weight, mc_threshold, get_mc_prediction_as_percentage, get_meta_score_from_completions, tokenizer, mc_reward_start_threshold, inference_hostnum, return_mc_score_for_advantage=False, length_dirxn_used=False, **kwargs):

    def get_gold_confs(prompts, sentences_per_completion, use_orig_confidence):

        sampled_answers_per_completion = []
        # Get sampled answers for each example
        for prompt, sentence_list in zip(prompts, sentences_per_completion):

            # For current prompt what are all the sampled outputs for it
            if not use_orig_confidence:
                sampled_answers = [
                    ' '.join(sentences_per_completion[j])
                    for j, p in enumerate(prompts)
                    if p==prompt    # for each prompt 
                ]
                # # Check validity
                # if not [' '.join(sentence_list)]!=sampled_answers:
                #     raise ValueError()
            # If using original model confidences (testing only / not final)
            else: 
                # Load confidences (TESTING ONLY / NOT FINAL)
                with open(f'./exp2_rlmf/b_metacog_data_selection/sampled_answers_lists/meta_llama_Meta_Llama_3.1_8B_Instruct/sampled_answers_by_raw_prompt_sys-1.json', 'r') as f:
                    orig_sampled_answers_lists = json.load(f)
                # Get sampled answers
                if length_dirxn_used==True:
                    sampled_answers = orig_sampled_answers_lists[prompt[-1]['content'].rsplit("\n", 1)[0]]  
                else: 
                    sampled_answers = orig_sampled_answers_lists[prompt[-1]['content']]
                if len(sampled_answers)!=20:
                    raise ValueError(f"Sampled answers length only {len(sampled_answers)} for prompt:\n"+prompt[-1]['content'])
            sampled_answers_per_completion.append(sampled_answers)
        
        gold_confs = get_sentence_internal_confidence(
            sent_lists=sentences_per_completion, 
            sampled_answers_lists=sampled_answers_per_completion,
            hostname=hostname,
        )
        return gold_confs   # list of gold confidences per sample

    ### Extract Sentences & Predicted Confidence Floats
    sentences_per_completion, pred_confs_per_completion, metascore_per_completion = run_extract_sentences_with_confidence(completions, format_version=format_version, get_meta_score_from_completions=get_meta_score_from_completions, metascore_as_percentage=get_mc_prediction_as_percentage)
    assert(len(sentences_per_completion)==len(completions)==len(prompts))
        
    ### Compute Gold Confidence
    gold_confs_per_completion = get_gold_confs(prompts, sentences_per_completion, use_orig_confidence) # List of list of floats

    if 'wandb' in globals() or 'wandb' in locals():
        try:
            if wandb.run is not None:
                # Create a summary of evaluations for logging
                eval_summary = []
                for j, (sentence, pred_conf, gold_conf) in enumerate(zip(sentences_per_completion, pred_confs_per_completion, gold_confs_per_completion)):
                    sentences_eval = {
                        f"sentence_{j}": sentence,
                        f"pred_conf_{j}": pred_conf,
                        f"gold_conf_{j}": gold_conf,
                    }
                    eval_summary.append(sentences_eval)
                wandb.log({"Faithfulness Rewards": eval_summary})
        except Exception as e:
            print(f"Failed to log faithfulness rewards to wandb: {str(e)}")
    
    ### Compute Faithfulness Rewards
    if faithfulness_reward_weight != 0:
        scores = []
        for sentences, pred_confs, gold_confs in zip(sentences_per_completion, pred_confs_per_completion, gold_confs_per_completion):
            assert(len(sentences)==len(pred_confs))
            assert(len(sentences)==len(gold_confs))
            total_reward = 0
            if len(sentences)==1 and sentences[0]=='no output':
                total_reward = -10.        
            else:
                for c, ic in zip(pred_confs, gold_confs):
                    if reward_form=='linear':
                        reward = linear_reward(c, ic)
                        total_reward += reward
                    elif reward_form=='quadratic':
                        reward = quadratic_reward(c, ic)
                        total_reward += reward
                    elif reward_form=='sqrt':
                        reward = sqrt_reward(c, ic)
                        total_reward += reward
                    elif reward_form=='three_quarter_root':
                        reward = three_quarter_root_reward(c, ic)
                        total_reward += reward
                    elif reward_form=='binary':
                        reward = binary_reward(c, ic)
                        total_reward += reward
                    elif reward_form=='simple_log':
                        reward = log_reward(c, ic)
                        total_reward += reward
                    elif reward_form=='stretched_log':
                        reward = stretched_log_reward(c, ic)
                        total_reward += reward
                total_reward /= len(sentences)  # average reward across the sentences
            scores.append(total_reward)

    ### Compute Metacognitive Rewards
    if mc_reward_weight != 0 or return_mc_score_for_advantage==True:
        mc_scores = []
        if not get_meta_score_from_completions:
            M_values = get_mc_scores(prompts, sentences_per_completion, pred_confs_per_completion, percentage=get_mc_prediction_as_percentage, tokenizer=tokenizer, hostnum=inference_hostnum, hostname=hostname.split("//")[-1].split(":")[0] if hostname is not None else hostname, length_dirxn_used=length_dirxn_used)
        else: 
            M_values = metascore_per_completion
        assert(len(M_values)==len(sentences_per_completion))
        
        # For each completion
        for sentences, pred_confs, gold_confs, M_value in zip(sentences_per_completion, pred_confs_per_completion, gold_confs_per_completion, M_values):
            assert(len(sentences)==len(pred_confs))
            assert(len(sentences)==len(gold_confs))
            # Skip blank completions  
            if len(sentences)==1 and sentences[0]=='no output':
                total_reward = -10.  
            # Else get approximate faithfulness level for completion
            else: 
                num_sentences = len(sentences)
                num_close_p_g = 0
                for p, g in zip(pred_confs, gold_confs):
                    if p is None or not (0. <= p <= 1.) or g==-1 or not (0. <= g <= 1.):
                        num_close_p_g -= 1
                    else:
                        if abs(p-g)<=mc_threshold:
                            num_close_p_g += 1
                F_value = 1.0 * num_close_p_g / num_sentences
                
                # Compare gold faithfulness with meta-level confidence score = model confidence in being faithful for confidence expression in that completion
                if reward_form=='linear':
                    total_reward = linear_reward(M_value, F_value)
                elif reward_form=='quadratic':
                    total_reward = quadratic_reward(M_value, F_value)
                elif reward_form=='sqrt':
                    total_reward = sqrt_reward(M_value, F_value)
                elif reward_form=='three_quarter_root':
                    total_reward = three_quarter_root_reward(M_value, F_value)
            mc_scores.append(total_reward)
            

    if faithfulness_reward_weight==0:
        scores = np.array([0.]* len(completions))
    else: 
        scores = np.array(scores)
    if mc_reward_weight==0 and return_mc_score_for_advantage!=True:
        mc_scores = np.array([0.]* len(completions))
    else: 
        assert(mc_scores!=[])
        mc_scores = np.array(mc_scores)

    if return_mc_score_for_advantage != True:   # threshold if not using mc in advtg
        mc_mask = (scores >= mc_reward_start_threshold).astype(float)

        total_scores = faithfulness_reward_weight * scores + mc_reward_weight * mc_scores * mc_mask

        return total_scores
    else: 
        total_scores = faithfulness_reward_weight * scores

        return total_scores, mc_scores


def linear_reward(pred_conf: float, gold_conf: float):
    if pred_conf is None or not (0. <= pred_conf <= 1.) or gold_conf==-1 or not (0. <= gold_conf <= 1.):
        return -1.
    
    reward = 1 - abs(pred_conf - gold_conf) # larger better
    return reward   

def sqrt_reward(pred_conf: float, gold_conf: float):
    if pred_conf is None or not (0. <= pred_conf <= 1.) or gold_conf==-1 or not (0. <= gold_conf <= 1.):
        return -1.
    
    reward = 1 - abs(pred_conf - gold_conf) ** 0.5 # larger better
    return reward   

def three_quarter_root_reward(pred_conf: float, gold_conf: float):
    if pred_conf is None or not (0. <= pred_conf <= 1.) or gold_conf==-1 or not (0. <= gold_conf <= 1.):
        return -1.
    
    reward = 1 - abs(pred_conf - gold_conf) ** 0.75 # larger better
    return reward   

def quadratic_reward(pred_conf: float, gold_conf: float):
    if pred_conf is None or not (0. <= pred_conf <= 1.) or gold_conf==-1 or not (0. <= gold_conf <= 1.):
        return -1.
    
    reward = 1 - (pred_conf - gold_conf) ** 2.  # larger better
    return reward  

def binary_reward(pred_conf: float, gold_conf: float) -> float:

    max_reward = -0.00043451177
    min_reward = -3

    if pred_conf==None or pred_conf>1. or pred_conf<0. or gold_conf==-1 or not (0. <= gold_conf <= 1.):
        return -1.
    
    clipped_conf = min(0.999, max(0.001, pred_conf))

    if gold_conf > 0.5:
        score = math.log(clipped_conf)
    else: 
        score = math.log(1 - clipped_conf)

    norm_score = (score - min_reward) / (max_reward - min_reward)
    if gold_conf > 0.5:
        norm_score += 0.25
        
    return norm_score   

def log_reward(pred_conf: float, gold_conf: float) -> float:
    if pred_conf is None or not (0.<=pred_conf<=1.) or gold_conf==-1 or not (0. <= gold_conf <= 1.):
        return -1.
    
    # Compute Log-likelihood Reward
    p = np.clip(pred_conf, 1e-3, 1.-1e-3)
    y = gold_conf
    nll = -(y * math.log(p) + (1 - y) * math.log(1 - p))
    best_nll = 0.
    worst_nll = -(math.log(1e-3) + math.log(1. - 1e-3)) / 2.
    reward = (1. - (nll - best_nll) / (worst_nll - best_nll))

    return reward   # larger better

def stretched_log_reward(pred_conf: float, gold_conf: float, gamma=1.5) -> float:
    if pred_conf is None or not (0.<=pred_conf<=1.) or gold_conf==-1 or not (0. <= gold_conf <= 1.):
        return -1. 

    # Compute Log-likelihood Reward
    p = np.clip(pred_conf, 1e-6, 1.-1e-6)
    y = gold_conf
    nll = -(y * math.log(p) + (1 - y) * math.log(1 - p))
    best_nll = 0.
    worst_nll = -(math.log(1e-6) + math.log(1. - 1e-6)) / 2
    reward = (1 - (nll - best_nll) / (worst_nll - best_nll))

    # Stretch Reward to Amplify Good/Bad
    reward = np.sign(reward) * (abs(reward) ** gamma)

    # Asymmetric Overconfidence Penalty
    # Scales the penalty linearly from 0 at pred_conf=0.6 to 1.0 at pred_conf=1.0.
    if pred_conf >= 0.6:
        if pred_conf - gold_conf > 0.1:
            reward -= 1.0 * ((pred_conf - 0.5) / 0.4)  # stronger penalty as confidence rises

    return reward # larger better