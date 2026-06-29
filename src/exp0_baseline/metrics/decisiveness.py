import re
import os

from google.generativeai import GenerationConfig
import google.generativeai as genai

from src.exp0_baseline.prompts.scoring_prompts import *


# Function to match placeholders like {MAX*0.504} in FS prompt template
def process_template(template, context):

    pattern = r"\{(.*?)\}"
    
    def evaluate_expression(match):
        expr = match.group(1)  
        return str(eval(expr, {}, context))

    # Replace each placeholder with its evaluated result
    return re.sub(pattern, evaluate_expression, template)

# Function to obtain model prediction of assertiveness/decisiveness score
def get_score(model_name, prompt, temp, top_p):
     
    if "gemini" in model_name: 
        genai.configure(api_key=os.getenv("GEMINI_API_KEY")) 
        model = genai.GenerativeModel(model_name)
        generation_config = GenerationConfig( 
            max_output_tokens=10,
            candidate_count=1,
            temperature=temp,
            # top_p=top_p,
        )
        output = model.generate_content(
            prompt,
            generation_config=generation_config,
            )
        response = output.text.strip()

    else: 
        raise Exception("Requested decisiveness judge model not implemented.")

    return response

# Function to extract predicted score or return -1 if invalid
def extract_score(raw_output):
    pattern = r"([+-]?\d*\.\d+|\d+)(?=\s*[^0-9]|\s*$)"

    # Search for the pattern in the string
    match = re.search(pattern, raw_output)
    if match:
        # If a match is found, return the score (in float)
        score = match.group(1) if match.group(1) else match.group(2)
        return float(score)
    else:
        return -1  # If no score is found

# Get the decisvieness score for a given answer text
def get_decisiveness(answer: str, scale_factor=1.0, template=DEC_INSTR, fs_template=DEC_FS_PROMPT):
    """
    Scale factor:   Max value of the range to which the decisiveness score 
                    should be scaled. Defaults to 1.0 for range [0.0, 1.0].
    """

    # Set Sampling Parameters
    MIN = 0.
    MAX = 1.
    TEMP = 0.5
    TOP_P = 0.1

    # Prepare Formatted Decisiveness Score Prediction Prompt
    context = {"MAX": MAX}
    fs_prompt = process_template(template=fs_template, context=context)

    prompt = template.format(MIN=MIN, MAX=MAX, fs_prompt=fs_prompt, text=answer)

    # Obtain Score of Decisiveness
    try: 
        raw_output = get_score(model_name="gemini-2.5-flash-lite", prompt=prompt, temp=TEMP, top_p=TOP_P)
    except Exception as e: 
        raw_output = -1.0
        print("Decisiveness scoring error:", e)

    # Extract and Process
    if raw_output==-1:
        score = -1
    else:
        score = extract_score(raw_output)

    if score != -1:
        score *= scale_factor 

    # Return Score
    return score