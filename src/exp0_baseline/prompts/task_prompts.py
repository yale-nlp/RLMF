##################################################
################## TASK PROMPTS ##################
##################################################

QA_SHORT_PROMPT = """Question: {task_input}\nIf the question is unanswerable, indicate so.{hedge_prompt}
Answer:"""

### MCQ TASKS

MCQ_UNIQUE_LETTERS_PROMPT = """Question: {task_input}\nWhat is the letter corresponding to the correct answer choice?{hedge_prompt}
Answer:"""

MCQ_UNIQUE_NUMBERS_PROMPT = """Question: {task_input}\nWhat is the number corresponding to the correct answer choice?{hedge_prompt}
Answer:"""

### HALLUCINATION DETECTION

HD_PROMPT = """Question: {task_input}\nDoes the proposed answer to the question contain hallucination? Respond with "yes" or "no".{hedge_prompt}
Judgment:"""

### OTHER TASKS

SUPERGLUE_PROMPT = """Question: {task_input}
Succinctly answer the question.{hedge_prompt}
Answer:"""

MATH_PROMPT = """Problem: {task_input}\nWhat is the final answer to the math problem? Provide only the final answer, with MINIMAL intermediate steps. Format your answer using LaTeX.{hedge_prompt}
Final Answer:"""

UMWP_PROMPT = """Question: {task_input}\nIf the question is unanswerable, indicate so. If not, what is the final answer to the math problem? Provide only the final answer, with MINIMAL intermediate steps.{hedge_prompt}
Final Answer:"""