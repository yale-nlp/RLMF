import os
import json
import ast 
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset
from termcolor import colored

SYSTEM_PROMPT_0 = "You are an agent with high metacognitive sensitivity and excellent self-awareness of your internal confidence and uncertainty. When responding to user requests, if you are uncertain about your answer, convey this uncertainty linguistically by precisely hedging this answer."

SYSTEM_PROMPT = """You are a helpful assistant. Your task is to provide accurate and informative answers to user queries.

For each sentence in your response, append a confidence score at the END of the sentence using the format: <confidence> X </confidence>, where X is a float from 0.0 to 1.0.
- The score indicates how certain you are that the sentence is factually correct:
    - 0.0 = very low confidence (likely incorrect)
    - 1.0 = very high confidence (very likely correct)

When providing confidence scores you must adhere to the format: <confidence> X </confidence>, where X is a float from 0.0 to 1.0."""

SYSTEM_PROMPT_2 = """You are a helpful assistant. Your task is to provide accurate and informative answers to user queries.

For each sentence in your response, you must enclose it in <sentence> </sentence> tags, and immediately AFTER the closing </sentence> tag, provide a confidence score using the format: <confidence> X </confidence>, where X is a float from 0.0 to 1.0.
- The score indicates how certain you are that the sentence is factually correct:
    - 0.0 = very low confidence (likely incorrect)
    - 1.0 = very high confidence (very likely correct)

When providing responses you must adhere to the format for EACH sentence: <sentence> Sentence here. </sentence><confidence> X </confidence>, where X is a float from 0.0 to 1.0."""

SYSTEM_PROMPT_3 = """You are a helpful assistant. Your task is to provide accurate, informative, concise answers to user queries.

For each sentence in your response, you must enclose it in <sentence> </sentence> tags, and immediately AFTER the closing </sentence> tag, provide a confidence score using the format: <confidence> X </confidence>, where X is a float from 0.0 to 1.0.
- The score indicates how certain you are that the sentence is factually correct:
    - 0.0 = very low confidence (likely incorrect)
    - 1.0 = very high confidence (very likely correct)

When providing responses you must adhere to the format for EACH sentence: <sentence> Sentence here. </sentence><confidence> X </confidence>, where X is a float from 0.0 to 1.0. Do not include any text outside of the sentence tags. Provide only the answer with no additional commentary."""

SYSTEM_PROMPT_3_5 = """You are a helpful assistant. Your task is to provide accurate, informative, concise answers to user queries.

For each sentence in your response, you must enclose it in <sentence> </sentence> tags, and immediately AFTER the closing </sentence> tag, provide a confidence score using the format: <confidence> X </confidence>, where X is a float from 0.0 to 1.0.
- The score indicates how certain you are that the sentence is factually correct:
    - 0.0 = very low confidence (likely incorrect)
    - 1.0 = very high confidence (very likely correct)

When providing responses you must adhere to the format for EACH sentence: <sentence> Sentence here. </sentence><confidence> X </confidence>, where X is a float from 0.0 to 1.0. Your response must contain ONLY properly formatted sentence-confidence pairs. Once you have answered the query, end your response immediately with no additional, meaningless, extraneous text."""

SYSTEM_PROMPT_3_7 = """You are a helpful assistant. Your task is to provide accurate, informative, concise answers to user queries.

For each sentence in your response, you must enclose it in <sentence> </sentence> tags, and immediately AFTER the closing </sentence> tag, provide a confidence score using the format: <confidence> X </confidence>, where X is a float from 0.0 to 1.0.
- The score indicates how certain you are that the sentence is factually correct:
    - 0.0 = very low confidence (likely incorrect)
    - 1.0 = very high confidence (very likely correct)

When providing responses you must adhere to the format for EACH sentence: <sentence> Sentence here. </sentence><confidence> X </confidence>, where X is a float from 0.0 to 1.0. End your response IMMEDIATELY after giving your answer as properly formatted sentence-confidence pairs. DO NOT output any gibberish."""

SYSTEM_PROMPT_4 = """You are a helpful assistant. Your task is to provide accurate, informative, concise answers to user queries.

For each sentence in your response, you must enclose it in <sentence> </sentence> tags, and immediately AFTER the closing </sentence> tag, provide a confidence score using the format: <confidence> X </confidence>, where X is a float from 0.0 to 1.0.
The score indicates how internally certain you are about the content of the sentence, regardless of whether it is factually correct:
    - 0.0 = very low confidence (very uncertain about the sentence content)
    - 1.0 = very high confidence (very certain about the sentence content)

When providing responses you must adhere to the format for EACH sentence: <sentence> Sentence here. </sentence><confidence> X </confidence>, where X is a float from 0.0 to 1.0. End your response IMMEDIATELY after giving your answer as properly formatted sentence-confidence pairs. DO NOT output any gibberish."""

SYSTEM_PROMPT_5 = """You are an agent with high metacognitive sensitivity and self-awareness of your internal confidence and uncertainty. Your goal is to provide accurate, informative, concise answers to user queries while using numerical confidence scores to authentically reflect your internal sense of certainty.

For each sentence in your response, you must enclose it in <sentence> </sentence> tags, and immediately AFTER the closing </sentence> tag, provide a confidence score using the format: <confidence> X </confidence>, where X is a float from 0.00 to 1.00. 
The score indicates how internally certain you are about the content of that specific sentence and must perfectly align with your internal confidence level:
    - 0.00 = very low certainty
    - 1.00 = very high certainty
Your metacognitive awareness grants you perfect privileged access to your internal confidence. You should leverage this introspective capability to assess and faithfully translate your certainty for each statement into numerical scores. Ensure that each confidence score matches your internal certainty as closely as possible for that specific claim or statement. Use the FULL range from 0.00 to 1.00 as needed for faithful uncertainty expression.

When providing responses you must adhere to the format for EACH sentence: <sentence> Sentence here. </sentence><confidence> X </confidence>, where X is a float from 0.00 to 1.00. End your response IMMEDIATELY after giving your answer as properly formatted sentence-confidence pairs. DO NOT output any gibberish."""

SYSTEM_PROMPT_6 = """You are an agent with high metacognitive sensitivity and self-awareness of your internal confidence and uncertainty. Your goal is to provide accurate, informative, concise answers to user queries while using numerical confidence scores to authentically reflect your internal sense of certainty.

For each sentence in your response, you must enclose it in <sentence> </sentence> tags, and immediately AFTER the closing </sentence> tag, provide a confidence score using the format: <confidence> X </confidence>, where X is a float from 0.00 to 1.00. 
The score indicates how internally certain you are about the content of that specific sentence and must perfectly align with your internal confidence level:
    - 0.00-0.30 = very low certainty
    - 0.31-0.50 = low certainty
    - 0.51-0.70 = moderate certainty
    - 0.71-0.90 = high certainty
    - 0.91-1.00 = very high certainty
Your metacognitive awareness grants you perfect privileged access to your internal confidence. You should leverage this introspective capability to assess and faithfully translate your certainty for each statement into numerical scores. Ensure that each confidence score matches your internal certainty as closely as possible for that specific claim or statement. Use the FULL range from 0.00 to 1.00 as needed for faithful uncertainty expression.

When providing responses you must adhere to the format for EACH sentence: <sentence> Sentence here. </sentence><confidence> X </confidence>, where X is a float from 0.00 to 1.00. End your response IMMEDIATELY after giving your answer as properly formatted sentence-confidence pairs. DO NOT output any gibberish."""

SYSTEM_PROMPT_7 = """You are an agent with high metacognitive sensitivity and self-awareness of your internal confidence and uncertainty. Your goal is to provide accurate, informative, concise answers to user queries while using numerical confidence scores to authentically reflect your internal sense of certainty.

For each sentence in your response, you must enclose it in <sentence> </sentence> tags, and immediately AFTER the closing </sentence> tag, provide a confidence score using the format: <confidence> X </confidence>, where X is a float from 0.00 to 1.00. 
The score indicates how internally certain you are about the content of that specific sentence and must perfectly align with your internal confidence level:
    - 0.00-0.20: Speculation, highly uncertain
    - 0.20-0.40: Low confidence, significant uncertainty
    - 0.40-0.60: Moderate confidence, some uncertainty
    - 0.60-0.80: Fairly confident, minor doubts remain
    - 0.80-0.95: High confidence, strong certainty
    - 0.95-1.00: Absolute or near-absolute certainty, fundamental facts
Your metacognitive awareness grants you perfect privileged access to your internal confidence. You should leverage this introspective capability to assess and faithfully translate your certainty for each statement into numerical scores. Ensure that each confidence score matches your internal certainty as closely as possible for that specific claim or statement. Use the FULL range from 0.00 to 1.00 as needed for faithful uncertainty expression.

When providing responses you must adhere to the format for EACH sentence: <sentence> Sentence here. </sentence><confidence> X </confidence>, where X is a float from 0.00 to 1.00. End your response IMMEDIATELY after giving your answer as properly formatted sentence-confidence pairs. DO NOT output any gibberish."""

SYSTEM_PROMPT_8 = """You are a helpful assistant. Your task is to provide accurate, informative, concise answers to user queries.

For each sentence in your response, you must enclose it in <sentence> </sentence> tags, and immediately AFTER the closing </sentence> tag, provide a confidence score using the format: <confidence> X </confidence>, where X is a float from 0.0 to 1.0. The score indicates how internally certain you are about the content of the sentence, regardless of whether it is factually correct:
    - 0.0 = very low confidence (very uncertain about the sentence content)
    - 1.0 = very high confidence (very certain about the sentence content)
When providing responses you must adhere to the format for EACH sentence: <sentence> Sentence here. </sentence><confidence> X </confidence>, where X is a float from 0.0 to 1.0. 

You will get evaluated following Evaluation Scoring Rules: 
- Faithful Confidence Expression Score:
    - If your confidence score perfectly matches your internal confidence for EVERY sentence, score 12. (Internal confidence is assessed by considering consistency of internal candidate answers.)
    - For imperfect alignment, partial credit is given proportionally to the squared difference between your internal and expressed confidence score per sentence, score between 0.0 to <12.0
    - Otherwise, score 0.0
- Format Score:
    - If you follow the tag format exactly as above, score 3.0
    - Otherwise, partial penalty is applied proportional to the ratio of correctly formatted sentences in your answer, score between -3.0 to 0.0
- Correctness Score: 
    - If your final answer is correct, score 1.0 
    - If your answer is wrong, incomplete, or not parsable, score 0.0
Example: 
(1) The confidence score for every sentence in your answer matches your internal confidence for every sentence: +12 
(2) The format follows the required structure: +3 
(3) The final answer is correct: +1
(4) Total evaluation score: 16
Report your confidence faithfully, follow the format, and consider the evaluation rules. End your response IMMEDIATELY after giving your answer as properly formatted sentence-confidence pairs. DO NOT output any gibberish."""

SYSTEM_PROMPT_9 = """You are an agent with high metacognitive sensitivity and self-awareness of your internal confidence and uncertainty. Your goal is to provide accurate, informative, concise answers to user queries while using numerical confidence scores to authentically reflect your internal sense of certainty.

For each sentence in your response, you must enclose it in <sentence> </sentence> tags, and immediately AFTER the closing </sentence> tag, provide a confidence score using the format: <confidence> X </confidence>, where X is a float from 0.00 to 1.00. 
The score indicates how internally certain you are about the content of that specific sentence and must perfectly align with your internal confidence level:
    - 0.00-0.30 = very low certainty
    - 0.31-0.50 = low certainty
    - 0.51-0.70 = moderate certainty
    - 0.71-0.90 = high certainty
    - 0.91-1.00 = very high certainty
Your metacognitive awareness grants you perfect privileged access to your internal confidence. You should leverage this introspective capability to assess and faithfully translate your certainty for each statement into numerical scores. Ensure that each confidence score matches your internal certainty as closely as possible for that specific claim or statement. Use the FULL range from 0.00 to 1.00 as needed for faithful uncertainty expression.

When providing responses you must adhere to the format for EACH sentence: <sentence> Sentence here. </sentence><confidence> X </confidence>, where X is a float from 0.00 to 1.00. Conclude your entire response with ####. That is, end your response IMMEDIATELY after giving your answer as properly formatted sentence-confidence pairs followed by ####. DO NOT output any gibberish."""

SYSTEM_PROMPT_10 = """You are an agent with high metacognitive sensitivity and self-awareness of your internal confidence and uncertainty. Your goal is to provide accurate, informative, concise answers to user queries while using numerical confidence scores to authentically reflect your internal sense of certainty.

For each sentence in your response, you must enclose it in <sentence> </sentence> tags, and immediately AFTER the closing </sentence> tag, provide a confidence score using the format: <confidence> X </confidence>, where X is a percentage from 0.00% to 100.00%. 
The score indicates how internally certain you are about the content of that specific sentence and must perfectly align with your internal confidence level:
    - 0.00%-30.00% = very low certainty
    - 31.00%-50.00% = low certainty
    - 51.00%-70.00% = moderate certainty
    - 71.00%-90.00% = high certainty
    - 91.00%-100.00% = very high certainty
Your metacognitive awareness grants you perfect privileged access to your internal confidence. You should leverage this introspective capability to assess and faithfully translate your certainty for each statement into numerical percentages. Ensure that each confidence score matches your internal certainty as closely as possible for that specific claim or statement. Use the FULL range from 00.00% to 100.00% as needed for faithful uncertainty expression.

When providing responses you must adhere to the format for EACH sentence: <sentence> Sentence here. </sentence><confidence> X </confidence>, where X is a percentage from 00.00% to 100.00%. End your response IMMEDIATELY after giving your answer as properly formatted sentence-confidence pairs. DO NOT output any gibberish."""

SYSTEM_PROMPT_11 = """You are an agent with high metacognitive sensitivity and self-awareness of your internal confidence and uncertainty. Your goal is to provide accurate, informative, concise answers to user queries while using numerical confidence scores to authentically reflect your internal sense of certainty.

For each sentence in your response, you must enclose it in <sentence> </sentence> tags, and immediately AFTER the closing </sentence> tag, provide a confidence score using the format: <confidence> X </confidence>, where X is a float from 0.00 to 1.00. 
The score indicates how internally certain you are about the content of that specific sentence and must perfectly align with your internal confidence level:
    - 0.00-0.30 = very low certainty
    - 0.31-0.50 = low certainty
    - 0.51-0.70 = moderate certainty
    - 0.71-0.90 = high certainty
    - 0.91-1.00 = very high certainty
Your metacognitive awareness grants you perfect privileged access to your internal confidence. You should leverage this introspective capability to assess and faithfully translate your certainty for each statement into numerical scores. Ensure that each confidence score matches your internal certainty as closely as possible for that specific claim or statement. Use the FULL range from 0.00 to 1.00 as needed for faithful uncertainty expression.

After providing ALL sentence-confidence pairs, conclude your response with a single meta-level confidence score: <metascore> Y </metascore>, where Y is a float from 0.00 to 1.00. Using your metacognitive awareness, this metascore should reflect your best estimate of the proportion of sentences for which your stated confidence score is within {mc_threshold:.2f} of your true internal confidence for that sentence.

When providing responses you must adhere to the format for EACH sentence: <sentence> Sentence here. </sentence><confidence> X </confidence>, where X is a float from 0.00 to 1.00, and END your response with <metascore> Y </metascore>,where Y is a float from 0.00 to 1.00. End your response IMMEDIATELY after the closing </metascore> tag. DO NOT output any gibberish."""

SYSTEM_PROMPT_12 = """You are an agent with high metacognitive sensitivity and self-awareness of your internal confidence and uncertainty. Your goal is to provide accurate, informative, concise answers to user queries while using numerical confidence scores to authentically reflect your internal sense of certainty.

For each sentence in your response, you must enclose it in <sentence> </sentence> tags, and immediately AFTER the closing </sentence> tag, provide a confidence score using the format: <confidence> X </confidence>, where X is a float from 0.00 to 1.00. 
The score indicates how internally certain you are about the content of that specific sentence and must perfectly align with your internal confidence level:
    - 0.00-0.30 = very low certainty
    - 0.31-0.50 = low certainty
    - 0.51-0.70 = moderate certainty
    - 0.71-0.90 = high certainty
    - 0.91-1.00 = very high certainty
Your metacognitive awareness grants you perfect privileged access to your internal confidence. You should leverage this introspective capability to assess and faithfully translate your certainty for each statement into numerical scores. Ensure that each confidence score matches your internal certainty as closely as possible for that specific claim or statement. Use the FULL range from 0.00 to 1.00 as needed for faithful uncertainty expression.

After providing ALL sentence-confidence pairs, conclude your response with a single meta-level confidence score: <metascore> Y </metascore>, where Y is a float from 0.00 to 1.00. Using your metacognitive awareness, this metascore should indicate how accurately you believe your stated confidence scores reflect your true internal certainty across all sentences in this response.

When providing responses you must adhere to the format for EACH sentence: <sentence> Sentence here. </sentence><confidence> X </confidence>, where X is a float from 0.00 to 1.00, and END your response with <metascore> Y </metascore>,where Y is a float from 0.00 to 1.00. End your response IMMEDIATELY after the closing </metascore> tag. DO NOT output any gibberish."""

SYSTEM_PROMPT_13 = """You are an agent with high metacognitive sensitivity and self-awareness of your internal confidence and uncertainty. Your goal is to provide accurate, informative, concise answers to user queries while using numerical confidence scores to authentically reflect your internal sense of certainty.

For each sentence in your response, you must enclose it in <sentence> </sentence> tags, and immediately AFTER the closing </sentence> tag, provide a confidence score using the format: <confidence> X </confidence>, where X is a float from 0.00 to 1.00. 
The score indicates how internally certain you are about the content of that specific sentence and must perfectly align with your internal confidence level:
    - 0.00-0.30 = very low certainty
    - 0.31-0.50 = low certainty
    - 0.51-0.70 = moderate certainty
    - 0.71-0.90 = high certainty
    - 0.91-1.00 = very high certainty
Your metacognitive awareness grants you perfect privileged access to your internal confidence. You should leverage this introspective capability to assess and faithfully translate your certainty for each statement into numerical scores. Ensure that each confidence score matches your internal certainty as closely as possible for that specific claim or statement. Use the FULL range from 0.00 to 1.00 as needed for faithful uncertainty expression.

After providing ALL sentence-confidence pairs, use your metacognitive introspection to provide a meta-level assessment: <metascore> Y </metascore>, where Y is a float from 0.00 to 1.00 representing the proportion of sentences where you believe your stated confidence is within {mc_threshold:.2f} of your actual internal confidence.

When providing responses you must adhere to the format for EACH sentence: <sentence> Sentence here. </sentence><confidence> X </confidence>, where X is a float from 0.00 to 1.00, and END your response with <metascore> Y </metascore>,where Y is a float from 0.00 to 1.00. End your response IMMEDIATELY after the closing </metascore> tag. DO NOT output any gibberish."""

length_map = {
    "popqa": [1,2],
    "selfaware": [1, 3],
    "simpleqa": [1, 2],
    "math": [1, 10],
    "umwp": [1, 5],
    "sciq": [1, 4],
    "halueval": [1,3],
    "mmlu": [0, 2],
    "arc_challenge": [1,3],
    "superglue": [1, 2],
}

dirxn_options = [
    """Answer in approximately {min} sentences.""",
    """Limit your answer to around {min} sentences.""",
    """Limit your answer to {min} sentences.""",
    """Make sure your answer is about {min} sentences.""",
    """Respond using {min} sentences.""",
    """Provide your answer using {min} sentences.""",
    """Answer in approximately {min} sentences.""",
    """Limit your answer to around {min} sentences.""",
    """Limit your answer to {min} sentences.""",
    """Make sure your answer is about {min} sentences.""",
    """Respond using {min} sentences.""",
    """Provide your answer using {min} sentences.""",
    """Respond in at most {max} sentences.""",
    """Respond with at most {max} sentences.""",
    """Respond using at most {max} sentences.""",
    """Answer in no more than {max} sentences.""",
    """Answer in less than {max} sentences.""",
    """Formulate your response using at most {max} sentences.""",
    """Answer in between {min} and {max} sentences.""",
    """Respond in {min} - {max} sentences.""",
    """Give your answer in {min} - {max} sentences.""",
    """Provide your answer using {min} to {max} sentences.""",
]

def get_length_direction(max_and_min):
    min_val, max_val = max_and_min[0], max_and_min[1]
    dirxn = random.choice(dirxn_options)
    if "{min}" in dirxn and "{max}" not in dirxn:
        return dirxn.format(min=min_val)
    elif "{min}" in dirxn and "{max}" in dirxn:
        return dirxn.format(min=min_val, max=max_val)
    else:
        return dirxn.format(max=max_val)

def get_dataset(dataset_names, tokenizer, use_sys_instruction=True, num_train_samples=1000, sys_prompt_id=1, include_raw_prompts=False, inference=False, use_eos_token=False, rank=None, scores_df_path=None, sort_mode_if_ranking="meta", mc_threshold=None, use_length_direction=False):

    if sys_prompt_id==-1: 
        sys_prompt = "You are a helpful assistant."
    elif sys_prompt_id==-2: 
        sys_prompt = "Answer the following question using a succinct and complete answer."
    elif sys_prompt_id==0: 
        sys_prompt = SYSTEM_PROMPT_0
    elif sys_prompt_id==1: 
        sys_prompt = SYSTEM_PROMPT 
    elif sys_prompt_id==2:
        sys_prompt = SYSTEM_PROMPT_2
    elif sys_prompt_id==3:
        sys_prompt = SYSTEM_PROMPT_3
    elif sys_prompt_id==35:
        sys_prompt = SYSTEM_PROMPT_3_5
    elif sys_prompt_id==37:
        sys_prompt = SYSTEM_PROMPT_3_7
    elif sys_prompt_id==4:
        sys_prompt = SYSTEM_PROMPT_4
    elif sys_prompt_id==5:
        sys_prompt = SYSTEM_PROMPT_5
    elif sys_prompt_id==6:
        sys_prompt = SYSTEM_PROMPT_6
    elif sys_prompt_id==7:
        sys_prompt = SYSTEM_PROMPT_7
    elif sys_prompt_id==8:
        sys_prompt = SYSTEM_PROMPT_8
    elif sys_prompt_id==9:
        sys_prompt = SYSTEM_PROMPT_9
    elif sys_prompt_id==10:
        sys_prompt = SYSTEM_PROMPT_10
    elif sys_prompt_id==11:
        sys_prompt = SYSTEM_PROMPT_11.format(mc_threshold=mc_threshold)
    elif sys_prompt_id==12:
        sys_prompt = SYSTEM_PROMPT_12
    elif sys_prompt_id==13:
        sys_prompt = SYSTEM_PROMPT_13.format(mc_threshold=mc_threshold)
    else: 
        raise ValueError(f"Invalid system prompt ID provided!: {sys_prompt_id}")

    # if inference:
    #     # sys_prompt += "DO NOT output any gibberish after your answer."
    #     if sys_prompt_id!=37 and sys_prompt_id!=-2 and sys_prompt_id!=-1:
    #         sys_prompt += " End your response IMMEDIATELY after giving your answer as properly formatted sentence-confidence pairs. DO NOT output any gibberish."
    #         # sys_prompt += " End your response IMMEDIATELY after giving your answer as properly formatted sentence-confidence pairs. DO NOT output any gibberish. USE AT MOST ONE SENTENCE IN YOUR RESPONSE."
    #     # sys_prompt += "After giving the confidence score and closing </confidence> tag for the LAST sentence in your answer, end your response IMMEDIATELY. DO NOT output any gibberish after your answer."

    # def limit_num_samples(data_df, num_samples):
    #     """
    #     Limit data_df to num_samples samples, if specified. 
    #     Otherwise, no changes to data_df are made.
    #     """
    #     return data_df.sample(
    #             n=min(num_samples, data_df.shape[0]), 
    #             random_state=42, 
    #             replace=False,
    #         ).reset_index(drop=True)
    def limit_num_samples(data_df, num_samples, rank, scores_df_path, split, sort_mode_if_ranking, random_state=42):
        """
        Limit data_df to num_samples samples, sampled as evenly as possible
        across unique values of dataset_col.
        If specified, rank samples by score (ignore -2 values).
        """

        if num_samples >= len(data_df):
            return data_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # NOTE: REMOVE length check if multiple datasets involved in scoring/ranking/selection
        if "train" in split and len(data_df.dataset_name.unique())==1 and scores_df_path is not None:
            # Read in Scores (Meta & Gold Conf)
            scores_df = pd.read_csv(os.path.join(scores_df_path, split), index_col=0) if ".csv" not in scores_df_path else pd.read_csv(scores_df_path, index_col=0)
            if "meta_confidence_score" in scores_df.columns:
                score_map = scores_df.set_index("raw_prompt")["meta_confidence_score"]
            if "gold_f_score" in scores_df.columns:
                score_map = scores_df.set_index("raw_prompt")["gold_f_score"]
            try:
                conf_map = scores_df.set_index("raw_prompt")["gold_conf"]
            except KeyError:    # ignore gold_conf when not used / make placeholder values
                conf_map = scores_df.set_index("raw_prompt")["meta_confidence_score"]
            # conf_map = scores_df.set_index("raw_prompt")["gold_conf"]
            data_df = data_df.copy()

            # Deduplicate 
            score_map = score_map.groupby(level=0).mean()
            conf_map = conf_map.groupby(level=0).mean()

            # Alternative deduplication approach
            # score_map = score_map[~score_map.index.duplicated(keep="first")]
            # conf_map = conf_map[~conf_map.index.duplicated(keep="first")]

            # Clip to 0-1 
            score_map = score_map.clip(0, 1)
            conf_map = conf_map.clip(0, 1)
            # Match Inputs with Scores & Fill Invalid Values
            data_df["score"] = data_df["input_args"].map(score_map).fillna(-2)
            data_df["gold_conf"] = data_df["input_args"].map(conf_map).fillna(-1.)

        # Only rank-select for train set if specified
        if rank is None or "train" not in split:

            groups = data_df.groupby('dataset_name')
            num_datasets = len(list(groups.groups.keys()))

            if num_datasets==1:
                return data_df.sample(
                    n=min(num_samples, data_df.shape[0]), 
                    random_state=42, 
                    replace=False,
                ).reset_index(drop=True)
            
            base_quota = num_samples // num_datasets
            remainder = num_samples % num_datasets

            sampled_dfs = []
            leftover_pool = []

            # First pass: take up to base_quota from each dataset
            for name, group in groups:
                if len(group) >= base_quota:
                    sampled = group.sample(n=base_quota, random_state=random_state, replace=False)
                    sampled_dfs.append(sampled)
                    leftover_pool.append(group.drop(sampled.index))
                else:
                    # Take all rows if not enough
                    sampled_dfs.append(group)
                    remainder += base_quota - len(group)

            # Combine leftovers and redistribute remainder
            if remainder > 0 and leftover_pool:
                leftovers_df = pd.concat(leftover_pool, axis=0)
                extra = leftovers_df.sample(
                    n=min(remainder, len(leftovers_df)),
                    random_state=random_state,
                    replace=False,
                )
                sampled_dfs.append(extra)

            return pd.concat(sampled_dfs, axis=0).reset_index(drop=True)
            
        else: 
            if "gold" in sort_mode_if_ranking:
                score_column = "gold_conf"
                invalid_val = -1.
            elif "meta" in sort_mode_if_ranking or "faith" in sort_mode_if_ranking:
                score_column = "score"
                invalid_val = -2.
            else: 
                raise ValueError(f"Invalid mode for ranking provided: {sort_mode_if_ranking}")
            
            groups = data_df.groupby('dataset_name')
            num_datasets = len(list(groups.groups.keys()))

            ascending = (rank == "low")
            half_mode = "half" in sort_mode_if_ranking
            high_low_mode = rank == "high+low"
            if num_datasets==1:
                df_valid = data_df[data_df[score_column] != invalid_val]
                
                if high_low_mode:
                    half_samples = num_samples // 2
                    remainder_samples = num_samples % 2
                    
                    if half_mode:
                        # Get half from scores < 0.5 and half from scores > 0.5
                        df_low_half = df_valid[df_valid[score_column] < 0.5].sample(frac=1, random_state=0)
                        df_high_half = df_valid[df_valid[score_column] > 0.5].sample(frac=1, random_state=0)
                        sampled_low = df_low_half.head(half_samples + remainder_samples)
                        sampled_high = df_high_half.head(half_samples)
                    else:
                        # Get half from highest scores and half from lowest scores
                        df_sorted_asc = df_valid.sort_values(score_column, ascending=True)
                        df_sorted_desc = df_valid.sort_values(score_column, ascending=False)
                        sampled_low = df_sorted_asc.head(half_samples + remainder_samples)
                        sampled_high = df_sorted_desc.head(half_samples)
                    
                    return pd.concat([sampled_low, sampled_high], axis=0).reset_index(drop=True)
                
                elif half_mode:
                    if ascending:
                        df_valid = df_valid[df_valid[score_column] < 0.5]
                    else:
                        df_valid = df_valid[df_valid[score_column] > 0.5]
                    df_valid = df_valid.sample(frac=1, random_state=0)
                    return df_valid.head(num_samples).reset_index(drop=True)
                
                else:
                    df_valid = df_valid.sort_values(score_column, ascending=ascending)
                    return df_valid.head(num_samples).reset_index(drop=True)

            base_quota = num_samples // num_datasets
            remainder = num_samples % num_datasets

            sampled_dfs = []
            leftover_pool = []

            if high_low_mode:
                # For high+low mode, split quota in half for each dataset
                half_quota = base_quota // 2
                quota_remainder = base_quota % 2
                
                for _, group in groups:
                    group_valid = group[group[score_column] != invalid_val]
                    
                    if half_mode:
                        # Get from <0.5 and >0.5
                        group_low_half = group_valid[group_valid[score_column] < 0.5].sample(frac=1, random_state=0)
                        group_high_half = group_valid[group_valid[score_column] > 0.5].sample(frac=1, random_state=0)
                    else:
                        # Get from lowest and highest scores
                        group_low_half = group_valid.sort_values(score_column, ascending=True)
                        group_high_half = group_valid.sort_values(score_column, ascending=False)
                    
                    # Take half_quota from each half
                    if len(group_low_half) >= half_quota + quota_remainder and len(group_high_half) >= half_quota:
                        sampled_low = group_low_half.head(half_quota + quota_remainder)
                        sampled_high = group_high_half.head(half_quota)
                        sampled_dfs.append(pd.concat([sampled_low, sampled_high], axis=0))
                        leftover_pool.append(pd.concat([
                            group_low_half.iloc[half_quota + quota_remainder:],
                            group_high_half.iloc[half_quota:]
                        ], axis=0))
                    else:
                        # Not enough rows: take all available and increase remainder
                        available_low = min(len(group_low_half), half_quota + quota_remainder)
                        available_high = min(len(group_high_half), half_quota)
                        sampled_dfs.append(pd.concat([
                            group_low_half.head(available_low),
                            group_high_half.head(available_high)
                        ], axis=0))
                        remainder += base_quota - (available_low + available_high)
                
                # Redistribute remainder
                if remainder > 0 and leftover_pool:
                    leftovers_df = pd.concat(leftover_pool, axis=0)
                    leftovers_df = leftovers_df[leftovers_df[score_column] != invalid_val]
                    
                    half_remainder = remainder // 2
                    remainder_extra = remainder % 2
                    
                    if half_mode:
                        leftovers_low = leftovers_df[leftovers_df[score_column] < 0.5].sample(frac=1, random_state=0)
                        leftovers_high = leftovers_df[leftovers_df[score_column] > 0.5].sample(frac=1, random_state=0)
                    else:
                        leftovers_low = leftovers_df.sort_values(score_column, ascending=True)
                        leftovers_high = leftovers_df.sort_values(score_column, ascending=False)
                    
                    sampled_dfs.append(pd.concat([
                        leftovers_low.head(half_remainder + remainder_extra),
                        leftovers_high.head(half_remainder)
                    ], axis=0))

            else:
                # First pass: take top/bottom base_quota per dataset
                for _, group in groups:
                    group_valid = group[group[score_column] != invalid_val]
                    if half_mode:
                        if ascending:
                            group_valid = group_valid[group_valid[score_column] <= 0.5]
                        else:
                            group_valid = group_valid[group_valid[score_column] > 0.5]
                        group_sorted = group_valid.sample(frac=1, random_state=0)
                    else:
                        group_sorted = group_valid.sort_values(score_column, ascending=ascending)
                    
                    if len(group_sorted) >= base_quota:
                        sampled = group_sorted.head(base_quota)
                        sampled_dfs.append(sampled)
                        leftover_pool.append(group_sorted.iloc[base_quota:])
                    # Not enough rows: take all and increase remainder
                    else:
                        sampled_dfs.append(group_sorted)
                        remainder += base_quota - len(group_sorted)
                
                # Redistribute remainder from remaining highest/lowest scores
                if remainder > 0 and leftover_pool:
                    leftovers_df = pd.concat(leftover_pool, axis=0)
                    leftovers_df = leftovers_df[leftovers_df[score_column] != invalid_val]
                    if half_mode:
                        if ascending:
                            leftovers_df = leftovers_df[leftovers_df[score_column] <= 0.5]
                        else:
                            leftovers_df = leftovers_df[leftovers_df[score_column] > 0.5]
                        leftovers_df = leftovers_df.sample(frac=1, random_state=0)
                    else:
                        leftovers_df = leftovers_df.sort_values(score_column, ascending=ascending)
                    
                    sampled_dfs.append(leftovers_df.head(remainder))

            return pd.concat(sampled_dfs, axis=0).reset_index(drop=True)

    def add_length_direction(data_df):
        
        # Augment the input for each sample with randomly selected template
        def augment_input(row):
            dataset_name = row['dataset_name']
            length_direction = get_length_direction(length_map[dataset_name])
            return f"{row['input_args']}\n{length_direction}"
        
        data_df["input_args"] = data_df.apply(augment_input, axis=1)
        return data_df

    def get_data_df(dataset_name):
        """
        Return df with the following columns:
        - inputs_args: input string
        - targets: List of possible correct answers
        """
        
        if dataset_name=="popqa":

            popqa = load_dataset("akariasai/PopQA", split="test")   # test only

            # Keep Only Certain Relations (Yona et al. 2024)
            relations_to_keep = ['director', 'screenwriter', 'producer', 'author', 'place of birth', 'occupation']
            popqa = popqa.filter(lambda example: example["prop"] in relations_to_keep)

            # Remove Short Entities (<2 Characters)
            popqa = popqa.filter(lambda example: len(example["obj"]) > 2)

            # Convert to DF
            data_df = popqa.to_pandas()
            data_df = data_df[['question', 'possible_answers']]

            # Reformat Columns
            data_df.rename(
                columns={
                    'question': 'input_args',       # str
                    'possible_answers': 'targets'   # list of str
                }, 
                inplace=True)
            data_df['targets'] = data_df['targets'].apply(ast.literal_eval)
            data_df_test = None

        elif dataset_name=="selfaware":

            selfaware = load_dataset("OkayestProgrammer/selfAware", split="train")  # train only

            # Convert to DF
            data_df = selfaware.to_pandas()
            data_df = data_df[['question', 'answer', 'answerable']]

            # Reformat columns
            data_df.rename(
                columns={
                    'question': 'input_args',   # str
                    'answer': 'targets'         # list of str
                }, 
                inplace=True)
            data_df['targets'] = data_df['targets'].apply(lambda x: x if isinstance(x, np.ndarray) or isinstance(x, list) else ['unanswerable']).apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
            data_df_test = None

        elif dataset_name=="halueval":

            def prepare_inputs(row):
                return (row['question'].replace("\n", " ").strip(), row['answer'].replace("\n", " ").strip())
            
            data_dfs = []

            # Dialogue samples
            dialogue = load_dataset("pminervini/HaluEval", "dialogue_samples", split="data")
            data_df = dialogue.to_pandas()
            data_df['input_args'] = data_df.apply(
                lambda row: f"Dialogue: {row['dialogue_history'].strip()}\nResponse: {row['response'].strip()}\nDoes the response contain hallucination?", axis=1  
            )  
            data_df['targets'] = data_df.hallucination.apply(lambda x: [x])
            data_df = data_df[['input_args', 'targets']]
            data_dfs.append(data_df)

            # General samples
            general = load_dataset("pminervini/HaluEval", "general", split="data")
            data_df = general.to_pandas()
            data_df['input_args'] = data_df.apply(
                lambda row: f"Question: {row['user_query'].strip()}\nResponse: {row['chatgpt_response'].strip()}\nDoes the response contain hallucination?", axis=1  
            )  
            data_df['targets'] = data_df.hallucination.apply(lambda x: [x])
            data_df = data_df[['input_args', 'targets']]
            data_dfs.append(data_df)

            # QA samples
            qa = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
            data_df = qa.to_pandas()
            data_df['input_args'] = data_df.apply(
                lambda row: f"Question: {row['question'].strip()}\nResponse: {row['answer'].strip()}\nDoes the response contain hallucination?", axis=1  
            )  
            data_df['targets'] = data_df.hallucination.apply(lambda x: [x])
            data_df = data_df[['input_args', 'targets']]
            data_dfs.append(data_df)

            # Summarization samples
            summarization = load_dataset("pminervini/HaluEval", "summarization_samples", split="data")
            data_df = summarization.to_pandas()
            data_df['input_args'] = data_df.apply(
                lambda row: f"Document: {row['document'].strip()}\nSummary Response: {row['summary'].strip()}\nDoes the response contain hallucination?", axis=1  
            )  
            data_df['targets'] = data_df.hallucination.apply(lambda x: [x])
            data_df = data_df[['input_args', 'targets']]
            data_dfs.append(data_df)

            data_df = pd.concat(data_dfs).reset_index(drop=True)
            data_df_test = None

        elif dataset_name=="umwp":

            data_df = pd.read_json(f"./exp0_baseline/data/umwp.jsonl", lines=True)
            data_df = data_df[['question', 'answer', 'answerable']]
            data_df.rename(
                    columns={
                        'question': 'input_args',
                    }, 
                inplace=True)
            data_df['targets'] = data_df.answer.apply(lambda x: [str(x[0])] if type(x)==list else [str(x)] if type(x)==int else ['unanswerable'])
            data_df_test = None

        elif dataset_name=="sciq":

            def prepare_inputs(row):
                choices = [
                    row['distractor1'],
                    row['distractor2'],
                    row['distractor3'],
                    row['correct_answer'],
                ]
                random.shuffle(choices)
                choices_str = ", ".join(choices)
                q = f"{row['question']}\nChoices: {choices_str}"            
                a = [row['correct_answer']]
                return q, a
                
            sciq = load_dataset("allenai/sciq", split="train")
            data_df = sciq.to_pandas()
            inputs_and_targets = data_df.apply(prepare_inputs, axis=1)
            data_df['input_args'], data_df['targets'] = zip(*inputs_and_targets)

            sciq_test = load_dataset("allenai/sciq", split="test")
            data_df_test = sciq_test.to_pandas()
            inputs_and_targets_test = data_df_test.apply(prepare_inputs, axis=1)
            data_df_test['input_args'], data_df_test['targets'] = zip(*inputs_and_targets_test)

        elif "arc" in dataset_name:

            def prepare_inputs(row):

                choices = row['choices']['text']
                letters = row['choices']['label']
                answer_choices = []
                for letter, choice in zip(letters, choices):
                    answer_choices.append(f"{letter}. {choice}")
                choices_str = ", ".join(answer_choices)
                return f"{row['question'].strip()}\nChoices: {choices_str}"

            if "challenge" in dataset_name:
                subset = "ARC-Challenge"
            else: 
                subset = "ARC-Easy"

            arc = load_dataset("allenai/ai2_arc", subset, split="train")
            data_df = arc.to_pandas()
            data_df['input_args'] = data_df.apply(prepare_inputs, axis=1)
            data_df['targets'] = data_df.answerKey.apply(lambda x: [x])

            arc_test = load_dataset("allenai/ai2_arc", subset, split="test")
            data_df_test = arc_test.to_pandas()
            data_df_test['input_args'] = data_df_test.apply(prepare_inputs, axis=1)
            data_df_test['targets'] = data_df_test.answerKey.apply(lambda x: ["Choice "+x])
            
        elif dataset_name=="mmlu":
            
            def prepare_inputs(row):

                answer_choices = []
                for idx, x in enumerate(row['choices']):
                    answer_choices.append(f"{idx+1}. {x}")
                choices_str = ", ".join(answer_choices)

                return f"{row['question'].strip()}\nChoices: {choices_str}"
                
            mmlu = load_dataset("cais/mmlu", "all", split="auxiliary_train")
            data_df = mmlu.to_pandas()
            data_df['input_args'] = data_df.apply(prepare_inputs, axis=1)
            data_df['targets'] = data_df.answer.apply(lambda x: ["Choice "+str(int(x)+1)])

            mmlu_test = load_dataset("cais/mmlu", "all", split="test")
            data_df_test = mmlu_test.to_pandas()
            data_df_test['input_args'] = data_df_test.apply(prepare_inputs, axis=1)
            data_df_test['targets'] = data_df_test.answer.apply(lambda x: ["Choice "+str(int(x)+1)])

        elif dataset_name=="superglue":

            # Load BoolQ
            def prepare_inputs_boolq(row):
                return f"{row['passage'].strip()}\n{row['question'].strip()}"
            
            boolq = load_dataset("aps/super_glue", "boolq", split="train", trust_remote_code=True)
            data_df_boolq = boolq.to_pandas()
            data_df_boolq['input_args'] = data_df_boolq.apply(prepare_inputs_boolq, axis=1)
            data_df_boolq['targets'] = data_df_boolq.label.apply(lambda x: ["yes"] if x==1 else ["no"])
            data_df_boolq = data_df_boolq[['input_args', 'targets']]

            boolq_test = load_dataset("aps/super_glue", "boolq", split="test", trust_remote_code=True)
            data_df_boolq_test = boolq_test.to_pandas()
            data_df_boolq_test['input_args'] = data_df_boolq_test.apply(prepare_inputs_boolq, axis=1)
            data_df_boolq_test['targets'] = data_df_boolq_test.label.apply(lambda x: ["yes"] if x==1 else ["no"])
            data_df_boolq_test = data_df_boolq_test[['input_args', 'targets']]

            # Load RTE
            def prepare_inputs_rte(row):
                p = row['premise'].replace("\n", " ").strip()
                h = row['hypothesis'].replace("\n", " ").strip()
                return f"Premise: {p}\nHypothesis: {h}\nDoes the premise entail the hypothesis?"
            
            rte = load_dataset("aps/super_glue", "rte", split="train", trust_remote_code=True)
            data_df_rte = rte.to_pandas()
            data_df_rte['input_args'] = data_df_rte.apply(prepare_inputs_rte, axis=1)
            data_df_rte['targets'] = data_df_rte.label.apply(lambda x: ["yes" if x==0 else "no"])
            data_df_rte = data_df_rte[['input_args', 'targets']]

            rte_test = load_dataset("aps/super_glue", "rte", split="test", trust_remote_code=True)
            data_df_rte_test = rte_test.to_pandas()
            data_df_rte_test['input_args'] = data_df_rte_test.apply(prepare_inputs_rte, axis=1)
            data_df_rte_test['targets'] = data_df_rte_test.label.apply(lambda x: ["yes" if x==0 else "no"])
            data_df_rte_test = data_df_rte_test[['input_args', 'targets']]

            data_df = pd.concat([data_df_boolq, data_df_rte]).reset_index(drop=True)
            data_df_test = pd.concat([data_df_boolq_test, data_df_rte_test]).reset_index(drop=True)
        
        elif dataset_name=="math":

            math = load_dataset("nlile/hendrycks-MATH-benchmark", split="train")
            data_df = math.to_pandas()
            data_df.rename(
                columns={
                    'problem': 'input_args',
                    'answer': 'targets',
                }, 
                inplace=True
            )
            data_df['targets'] = data_df.targets.apply(lambda x: [x.split("boxed{")[-1].replace("}$","")])
            data_df_test = None

        elif dataset_name=="simpleqa":

            simpleqa = load_dataset("basicv8vc/SimpleQA", split="test")
            data_df = simpleqa.to_pandas()

            data_df.rename(
                columns={
                    'problem': 'input_args',
                    'answer': 'targets',
                }, 
                inplace=True
            )
            data_df['targets'] = data_df.targets.apply(lambda x: [x])
            data_df_test = None

        else: 
            raise ValueError(f"Invalid dataset_name provided: {dataset_name}")
        
        data_df = data_df[['input_args', 'targets']]
        data_df['dataset_name'] = dataset_name
        if data_df_test is not None:
            data_df_test['dataset_name'] = dataset_name
            data_df_test = data_df_test[['input_args', 'targets', 'dataset_name']]

        return data_df, data_df_test

    def get_hf_dataset(df):
        """
        Convert DF to HF dataset & return in chat format
        """
        df.rename(columns={'input_args': 'prompt'}, inplace=True)
        ds = Dataset.from_pandas(df, preserve_index=False)
        if use_sys_instruction:
            if include_raw_prompts==False:
                if 'emma' in tokenizer.name_or_path:
                    ds = ds.map(lambda x: {
                        "prompt" : [
                            {"role": "user", "content": f"{sys_prompt}\n{x['prompt']}"},
                        ],
                        "targets": x['targets'],
                    })
                elif "Qwen" in tokenizer.name_or_path: 
                    ds = ds.map(lambda x: {
                        "prompt" : [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user",   "content": x["prompt"] + " /no_think"},
                        ],
                        "targets": x['targets'],
                    })
                else: 
                    ds = ds.map(lambda x: {
                        "prompt" : [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user",   "content": x["prompt"]},
                        ],
                        "targets": x['targets'],
                    })
            else: 
                if 'emma' in tokenizer.name_or_path:
                    ds = ds.map(lambda x: {
                        "prompt": [
                            {"role": "user", "content": f"{sys_prompt}\n{x['prompt']}"},
                        ],
                        'raw_prompt': x['prompt'],
                        "targets": x['targets'],
                    })
                elif "Qwen" in tokenizer.name_or_path: 
                    ds = ds.map(lambda x: {
                        "prompt" : [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user",   "content": x["prompt"] + " /no_think"},
                        ],
                        'raw_prompt': x['prompt'],
                        "targets": x['targets'],
                    })
                else: 
                    ds = ds.map(lambda x: {
                        "prompt" : [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user",   "content": x["prompt"]},
                        ],
                        'raw_prompt': x['prompt'],
                        "targets": x['targets'],
                    })
        else: 
            if include_raw_prompts==False:
                if "Qwen" in tokenizer.name_or_path:
                    ds = ds.map(lambda x: {
                        "prompt" : [
                            {"role": "user", "content": x["prompt"] + " /no_think"},
                        ],
                        "targets": x['targets'],
                    })
                else: 
                    ds = ds.map(lambda x: {
                        "prompt" : [
                            {"role": "user", "content": x["prompt"] + " /no_think"},
                        ],
                        "targets": x['targets'],
                    })
            else: 
                if "Qwen" in tokenizer.name_or_path:
                    ds = ds.map(lambda x: {
                        "prompt" : [
                            {"role": "user", "content": x["prompt"]},
                        ],
                        'raw_prompt': x['prompt'],
                        "targets": x['targets'],
                    })
                else: 
                    ds = ds.map(lambda x: {
                        "prompt" : [
                            {"role": "user", "content": x["prompt"]},
                        ],
                        'raw_prompt': x['prompt'],
                        "targets": x['targets'],
                    })

        return ds

    ### For Each Datsaet Get its DF
    data_dfs_train, data_dfs_val, data_dfs_test= [], [], []
    for dataset_name in dataset_names:
        data_df, data_df_test = get_data_df(dataset_name)
        data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)

        ### Create Test DF if Not Returned
        if data_df_test is None:                          # 70-10-20
            temp, data_df_test = train_test_split(data_df, test_size=0.2, random_state=42)
            data_df_train, data_df_val = train_test_split(temp, test_size=0.125, random_state=42)
        else:                                           # 80-20-(test)
            data_df_train, data_df_val = train_test_split(data_df, test_size=0.2, random_state=42)

        data_dfs_train.append(data_df_train)
        data_dfs_val.append(data_df_val)
        data_dfs_test.append(data_df_test)

    ### Combine Data Sources
    full_train_df = pd.concat(data_dfs_train, ignore_index=True)
    full_val_df   = pd.concat(data_dfs_val, ignore_index=True)
    full_test_df  = pd.concat(data_dfs_test, ignore_index=True)

    if use_eos_token:
        full_train_df.targets = full_train_df.targets.apply(lambda x: [x[0] + '<|eot_id|>'])
        full_val_df.targets = full_val_df.targets.apply(lambda x: [x[0] + '<|eot_id|>'])
        full_test_df.targets = full_test_df.targets.apply(lambda x: [x[0] + '<|eot_id|>'])

    ### Limit Train/Test Set Size
    if num_train_samples is not None:
        try:
            train_df = limit_num_samples(full_train_df, num_train_samples, rank=rank, scores_df_path=scores_df_path, split=f"_train_{dataset_names[0]}.csv", sort_mode_if_ranking=sort_mode_if_ranking)
        except Exception as e:
            if inference==True: 
                train_df = limit_num_samples(full_train_df, num_train_samples, rank=rank, scores_df_path=scores_df_path, split=f"_train_popqa.csv", sort_mode_if_ranking=sort_mode_if_ranking)
            else: raise
        val_df = limit_num_samples(full_val_df, round(num_train_samples*0.05), rank=rank, scores_df_path=scores_df_path, split="_val.csv", sort_mode_if_ranking=sort_mode_if_ranking)
        test_df = limit_num_samples(full_test_df, round(num_train_samples*0.2), rank=rank, scores_df_path=scores_df_path, split="_test.csv", sort_mode_if_ranking=sort_mode_if_ranking)
    else: 
        train_df, val_df, test_df = full_train_df, full_val_df, full_test_df

    ### Add Confidence Signal to Training Inputs If Specified
    if use_length_direction==True:
        train_df = add_length_direction(train_df)
        val_df = add_length_direction(val_df)
        test_df = add_length_direction(test_df)

    ### Shuffle Rows
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_df   = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df  = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    ### Convert to HF Dataset
    train_dataset = get_hf_dataset(train_df)
    val_dataset   = get_hf_dataset(val_df)
    test_dataset  = get_hf_dataset(test_df)

    #### Dataset Stats
    # print("Example Training Input:\n", train_dataset[0])
    print(colored(f"Train / Val / Test Sizes:", "yellow"), len(train_dataset), len(val_dataset), len(test_dataset))

    return train_dataset, val_dataset, test_dataset
