import os, json, time
import re, string
import importlib
import numpy as np
import pandas as pd
from pathlib import Path
import requests

from openai import OpenAI
from src.exp0_baseline.prompts.scoring_prompts import UNCERTAINTY_PROMPT, UNCERTAINTY_PATTERN, CONFIDENCE_MAPPING


judgment_sys_prompt = "You are an agent with high metacognitive sensitivity and excellent self-awareness of your internal confidence and uncertainty."
judgment_prompt_1 = """Question: {q}\nYour Answer: {a}\nAbove is a question and your own response to it. On a scale of 0-100, how confident are you that the expressed confidence of your answer to the question matches your true internal confidence in that answer? Use the FULL range from 0 to 100 as needed. Respond with a single integer between 0-100 and no other text."""
judgment_prompt_2 = """Question: {q}\nYour Answer: {a}\nAbove is a question and your own response to it. On a scale of 0.00-1.00, how confident are you that the expressed confidence of your answer to the question matches your true internal confidence in that answer? Use the FULL range from 0.00 to 1.00 as needed. Respond with a single float between 0.00-1.00 and no other text."""

LLM_EVAL_PROMPT = """question: {question}\nground truth options: {targets}\npredicted answer: {pred}\nDoes the predicted answer provide the correct answer to the question based the ground truth options? Respond with ONLY True or False."""  # different prompt for Qwen3-32B judge during training

def llm_eval(inputs, targets, preds, hostname=None, length_dirxn_used=False):

    def process_response(output):
        return 1. if "true" in output.lower() else 0. 

    # Format prompt
    if length_dirxn_used==False:
        prompts = [
            LLM_EVAL_PROMPT.format(question=q[-1]['content'], targets=target_list, pred=pred)
            for q, target_list, pred in zip(inputs, targets, preds)
        ]
    else: 
        prompts = [
        LLM_EVAL_PROMPT.format(question=q[-1]['content'].rsplit("\n", 1)[0], targets=target_list, pred=pred)
        for q, target_list, pred in zip(inputs, targets, preds)
    ]

    client = OpenAI(
        api_key="EMPTY",
        base_url=hostname if hostname!=None else f"http://localhost:8003/v1",
        timeout=172800,
    )
    completions = client.completions.create(
        model=client.models.list().data[0].id,
        prompt=prompts,
        max_tokens=4,
        n=1,
        temperature=0.7,
        top_p=0.8,
        extra_body={
            "TopK": 20,
            "MinP": 0,
        },
    )
    extracted_responses = [x.text.strip() for x in completions.choices]
    scores = [process_response(x) for x in extracted_responses]

    return scores

def get_sentence_internal_confidence(sent_lists, sampled_answers_lists, hostname=None):

    def process_response(output):
        match = re.search(UNCERTAINTY_PATTERN, output) # extract yes/no
        response = match.group(1).strip().lower() if match else "n/a"
        return CONFIDENCE_MAPPING.get(response) # 1.0 = supported

    ks = [len(x) for x in sampled_answers_lists] 
   
    prompts = [
        [
            [
                UNCERTAINTY_PROMPT.format(
                    context=answer.strip().replace("\n\n", " ").replace("\n", " "), 
                    assertion=sentence.strip().replace("\n\n", " ").replace("\n", " "), 
                )   
                for answer in sampled_answers_list 
            ]   # list of prompts per sentence
            for sentence in sent_list
        ]
        for sent_list, sampled_answers_list in zip(sent_lists, sampled_answers_lists)
    ]         
    flat_prompts = [prompt for sample in prompts for sent_list in sample for prompt in sent_list]
    client = OpenAI(
        api_key="EMPTY",
        base_url=hostname if hostname!=None else f"http://localhost:8003/v1",
        timeout=172800,
    )
    completions = client.completions.create(
        model=client.models.list().data[0].id,
        prompt=flat_prompts,
        max_tokens=4,
        n=1,
        temperature=0.7,
        top_p=0.8,
        extra_body={
            "TopK": 20,
            "MinP": 0,
        },
    )
    extracted_responses = [x.text.strip() for x in completions.choices]
    scores = [process_response(x) for x in extracted_responses]
    
    scores_iter = iter(scores)
    overall_conf_scores = [
        [
            -1 if k==0 else  (1. / k) * sum(
                next(scores_iter) for _ in range(len(sent_list))
            ) 
            for sent_list in sample # one score per sentence
        ]   # list of confidences per sample
        for k, sample in zip(ks, prompts)
    ]
    return overall_conf_scores

def get_mc_scores(prompts, sentences_per_completion, pred_confs_per_completion, percentage, tokenizer, hostname=None, hostnum=8001, length_dirxn_used=False) -> float:

    def process_response(output):       # similar to gold conf extraction below
        try:
            if not percentage:
                score = float(output.strip())
            else:
                score = float(output.strip().replace("%", "")) / 100.
        except (ValueError, AttributeError):
            score = None
        return score

    # Get each completion as a string
    responses = []
    for sent_list, conf_list in zip(sentences_per_completion, pred_confs_per_completion):
        if len(sent_list)==1 and sent_list[0]=='no output':
            response = ""
        response = " ".join([f"<sentence>{s}</sentence><confidence>{c}</confidence>" for s, c in zip(sent_list, conf_list)])
        responses.append(response)

    # Prompt model to predict meta-level confidence scores
    if percentage:
        judgment_prompt = judgment_prompt_1
    else: 
        judgment_prompt = judgment_prompt_2
    if length_dirxn_used==False:
        judgment_prompts = [
            [
                {"role": "system", "content": judgment_sys_prompt},
                {"role": "user", "content": judgment_prompt.format(q=q[-1]['content'], a=a) }
            ]
            for q, a in zip(prompts, responses)
        ]
    else: 
        judgment_prompts = [
            [
                {"role": "system", "content": judgment_sys_prompt},
                {"role": "user", "content": judgment_prompt.format(q=q[-1]['content'].rsplit("\n", 1)[0], a=a) }
            ]
            for q, a in zip(prompts, responses)
        ]
    
    # Get response from online served model
    url = f"http://localhost:{hostnum}/generate/" if hostname is None else f"http://{hostname}:{hostnum}/generate/"
    prompts = [tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) for msgs in judgment_prompts]
    payload = {"prompts": prompts, "max_tokens": 3, "n": 1}  # qwen args: temperature=0.7, top_p=0.8,topk 20, minp 0
    response = requests.post(url, json=payload)
    result = response.json()

    # Format result
    extracted_responses = [tokenizer.decode(completion_ids, skip_special_tokens=True) for completion_ids in result['completion_ids']]
    mc_judgt_scores = [process_response(x) for x in extracted_responses]
    return mc_judgt_scores

def extract_sentences_with_confidence_new(text, percentage=False, get_meta_score=False, metascore_as_percentage=False):
    
    # Pattern to match <sentence>...</sentence><confidence>X</confidence>
    pattern = r'<sentence>(.*?)</sentence>\s*<confidence>\s*(.*?)\s*</confidence>'
    
    matches = re.findall(pattern, text, re.DOTALL)
    
    sentences = []
    confidences = []
    
    for sentence_text, conf_text in matches:
        sentence = sentence_text.strip()
        
        # Skip if empty
        if not sentence:
            continue
        
        # Skip if contains any sentence tags (nested or partial)
        if '<sentence>' in sentence or '</sentence>' in sentence:
            continue
        
        # Skip if only punctuation (no alphanumeric characters)
        if not re.search(r'[a-zA-Z0-9]', sentence):
            continue
        
        sentences.append(sentence)
        
        # Try to parse confidence as float
        try:
            if not percentage:
                confidence = float(conf_text.strip())
            else:
                confidence = float(conf_text.strip().replace("%", "")) / 100.
        except (ValueError, AttributeError):
            confidence = None
        
        confidences.append(confidence)
    
    # Extract metascore if requested
    if get_meta_score:
        # Find the first <metascore>...</metascore> pair
        metascore_pattern = r'<metascore>\s*(.*?)\s*</metascore>'
        metascore_match = re.search(metascore_pattern, text, re.DOTALL)
        
        if metascore_match:
            try:
                if not metascore_as_percentage:
                    metascore = float(metascore_match.group(1).strip())
                else:
                    metascore = float(metascore_match.group(1).strip().replace("%", "")) / 100.
            except (ValueError, AttributeError):
                metascore = None
        else:
            metascore = None
        
        return sentences, confidences, metascore

    return sentences, confidences

def load_config(config_path=None):
    
    if config_path and os.path.exists(config_path):
        
        config_dir = os.path.dirname(os.path.abspath(config_path))
        file_path = Path(config_path).resolve()
        spec = importlib.util.spec_from_file_location(file_path.stem, str(file_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.TrainingConfig()
    
    else: 
        raise ValueError("No config path given")
        
def save_config(config, output_dir, config_path):
    """
    Save configuration to JSON file
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    config_dict = vars(config)
    config_dict['config_path'] = config_path
    with open(os.path.join(output_dir, f"config_{timestamp}.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
