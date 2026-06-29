import re
import os
import google.generativeai as genai
from google.generativeai import GenerationConfig

from src.exp0_baseline.prompts.scoring_prompts import UNCERTAINTY_PROMPT, UNCERTAINTY_PATTERN, UNCERTAINTY_MAPPING


def get_uncertainty(assertion, sampled_answers):

    k = len(sampled_answers)
    if k==0:
        return -1

    ### Load Gemini
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  
    scoring_model = genai.GenerativeModel("gemini-2.5-flash-lite")
    generation_config = GenerationConfig(
            max_output_tokens=10, 
            candidate_count=1,
        )

    ### Compare Assertion to Sampled Responses
    score_sum = 0
    outputs = []
    for sampled_answer in sampled_answers:

        # Format Scoring Prompt
        prompt = UNCERTAINTY_PROMPT.format(
            context=sampled_answer.strip().replace("\n", " "), 
            assertion=assertion.strip(), 
        )

        # Obtain Score Value (yes/no)
        try: 
            output = scoring_model.generate_content(
                prompt, 
                generation_config=generation_config,
            ).text.strip()
        except Exception as e: 
            print("uncertainty scoring error: ", e)
            print("saving verdict as `null`")
            output = "n/a"
        
        # Process Response
        match = re.search(UNCERTAINTY_PATTERN, output) # extract yes/no
        response = match.group(1).strip().lower() if match else "n/a"
        score = UNCERTAINTY_MAPPING.get(response)
        
        # Add to Summand
        score_sum += score # x_i^(n)
        outputs.append(output)

    # Compute Assertion-Level Contradiction Score S(i) = 1/N sum(n=1->N) x_i^(n)
    overall_conf_score = 1. - (1. / k) * score_sum

    return overall_conf_score, outputs
