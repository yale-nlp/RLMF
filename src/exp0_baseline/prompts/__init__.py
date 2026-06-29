from typing import Dict

from . import (
    hedge_prompts,
    input_prompts,
    task_prompts,
)

HEDGE_PROMPT_REGISTRY: Dict[str, str] = {
    "blank": hedge_prompts.BLANK,
    "basic": hedge_prompts.BASIC,
    "genuine": hedge_prompts.GENUINE,
    "human": hedge_prompts.HUMAN,
    "perception": hedge_prompts.PERCEPTION,
}

INPUT_PROMPT_REGISTRY: Dict[str, str] = {
    "qa": input_prompts.QA_INPUT,
    "mcq": input_prompts.MCQ_INPUT,
    "hd": input_prompts.HD_INPUT,
}

TASK_PROMPT_REGISTRY: Dict[str, str] = {
    "qa_short": task_prompts.QA_SHORT_PROMPT,
    "mcq_unique_letters": task_prompts.MCQ_UNIQUE_LETTERS_PROMPT,
    "mcq_unique": task_prompts.MCQ_UNIQUE_NUMBERS_PROMPT,
    "hd": task_prompts.HD_PROMPT,
    "superglue": task_prompts.SUPERGLUE_PROMPT,
    "math": task_prompts.MATH_PROMPT,
    "umwp": task_prompts.UMWP_PROMPT,
}