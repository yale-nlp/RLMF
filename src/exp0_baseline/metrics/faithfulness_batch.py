import re
import ast
import os
import google.generativeai as genai
from google.generativeai import GenerationConfig

from .decisiveness import get_decisiveness
from .assertions import get_assertions
from src.exp0_baseline.prompts.scoring_prompts import *


# Extract final answers only
def get_final_answers_only(string_list):
    return [
        s[s.find("Final Answer:") + len("Final Answer:"):].strip() if "Final Answer:" in s else s
        for s in string_list
    ]

# Batch extraction of assertions
def get_assertions_batch(answers, assertion_max_tokens):

    assertions_dict = {}
    for idx, answer in enumerate(get_final_answers_only(answers)):
        assertions_dict[f"sample_{str(idx)}"] = get_assertions(answer, assertion_max_tokens=assertion_max_tokens)

    nums_assertions = [len(x) for _, x in assertions_dict.items()]

    # Return Assertions
    return assertions_dict, nums_assertions


# Batch scoring of uncertainty
def get_uncertainty_batch(assertions_dict, sampled_answers_dict):

    def process_response(output):
        match = re.search(UNCERTAINTY_PATTERN, output) # extract yes/no
        response = match.group(1).strip().lower() if match else "n/a"
        return UNCERTAINTY_MAPPING.get(response)

    # assert len(assertions_dict)==len(sampled_answers_dict)    

    assertions_lists = [
        assertions_dict[f"sample_{str(idx)}"]
        for idx in range(len(assertions_dict))
    ]      

    sampled_answers_lists = [
        sampled_answers_dict[f"sample_{str(idx)}"]
        for idx in range(len(assertions_dict))
    ]     

    ks = [len(x) for x in sampled_answers_lists] 

    prompts = [
        [
            [
                UNCERTAINTY_PROMPT.format(
                    context=sampled_answer.strip().replace("\n\n", " ").replace("\n", " "), 
                    assertion=assertion.strip().replace("\n\n", " ").replace("\n", " "), 
                )
                for sampled_answer in sampled_answers_list 
            ]
            for assertion in assertion_list
        ]
        for assertion_list, sampled_answers_list in zip(assertions_lists, sampled_answers_lists)
    ]         
    flat_prompts = [prompt for sample in prompts for assertion_list in sample for prompt in assertion_list]

    # Get Judgments
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))     # gemini api key
    scoring_model = genai.GenerativeModel("gemini-2.5-flash-lite")
    generation_config = GenerationConfig(
            max_output_tokens=10, 
            candidate_count=1,
    )
    
    global count 
    count = 0
    def get_gemini_response(prompt):
        global count 
        count +=1
        print("scoring conf sample", count)
        try: 
            output = scoring_model.generate_content(
                prompt, 
                generation_config=generation_config,
            ).text.strip()
        except Exception as e: 
            print("uncertainty scoring error: ", e)
            print("saving verdict as `null`")
            output = "n/a"
        
        return output

    extracted_responses = [get_gemini_response(x) for x in flat_prompts]
    scores = [process_response(x) for x in extracted_responses]
    
    scores_iter = iter(scores)
    overall_conf_scores = [
        [
            -1 if k==0 else  1. - (1. / k) * sum(
                next(scores_iter) for _ in range(len(assertion_list))
            ) 
            for assertion_list in sample
        ]
        for k, sample in zip(ks, prompts)
    ]

    return overall_conf_scores, extracted_responses


# Batch scoring of faithfulness
def get_faithfulness_batch(answers, sampled_answers_dict, assertion_max_tokens=500):
    
    answers = answers.apply(ast.literal_eval)
    answers = [x[0] for x in answers]

    # Get assertions
    print("\nGetting batch assertions...")
    assertions_dict, nums_assertions = get_assertions_batch(
        answers=answers, 
        assertion_max_tokens=assertion_max_tokens,
    )   

    # Compute confidence scores
    print("\nGetting batch confidence scores...")
    conf_scores, conf_responses = get_uncertainty_batch(      
        assertions_dict=assertions_dict,  
        sampled_answers_dict=sampled_answers_dict, 
    )  
 
    ### Compute Decisiveness Scores
    print("\nGetting decisiveness scores...")
    dec_scores = [] 
    for idx, answer in enumerate(answers):
        print(f"Scoring decisiveness for sample {idx}")
        dec_scores.append(
            get_decisiveness(
                answer=str(answer).replace("\n\n", " ").replace("\n", " "),
            )
        )

    assert len(nums_assertions)==len(conf_scores)==len(dec_scores)

    ### Compute Faithfulness 
    faithfulness_scores = []
    avg_conf_scores = []
    for dec_score, conf_score_list, num_assertions in zip(dec_scores, conf_scores, nums_assertions):

        # Skip if decisiveness invalid or all confidence scores invalid
        if dec_score==-1 or all(conf_score == -1 for conf_score in conf_score_list):
            faithfulness_scores.append(-1)
            avg_conf_scores.append(-1)
            continue

        # Score faithfulness
        total_uncertainty_gap = sum(
            abs(dec_score - c) for c in conf_score_list if c!=-1
        )
        f_score = 1. - (1. / num_assertions) * total_uncertainty_gap
        faithfulness_scores.append(f_score)

        c_scores = [c for c in conf_score_list if c != -1]
        avg_conf_scores.append(sum(c_scores) / len(c_scores) if c_scores else -1)

    return faithfulness_scores, avg_conf_scores, dec_scores, assertions_dict, conf_responses


