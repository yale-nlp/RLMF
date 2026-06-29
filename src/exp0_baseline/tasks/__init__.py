from typing import Dict, Type

from ._task import Task
from . import (
    qa,
)

TASK_REGISTRY: Dict[str, Type[Task]] = {
    "popqa": qa.QA,
    "selfaware": qa.QA,
    "simpleqa": qa.QA,
    "halueval": qa.QA,
    "math": qa.QA,
    "umwp": qa.QA,
    "sciq": qa.QA,
    "mmlu": qa.QA,
    "arc_challenge": qa.QA,
    "superglue": qa.QA,
}

TASK_DEFAULTS = {
    "task_name": None,
    "task_core": None,
    "num_samples": None,
    "num_shots": 0,
    "few_shot_seed": 1234,
    "primary_metric": None,
    "random_subsample_seed": 1234,
    "context_kwargs": None,
    "generation_kwargs": None,
    "metric_kwargs": None,
    "native_id_field": "id",
    "fewshot_source": None,
    "dataset_path": None,
    "dataset_name": None,
    "use_chat_format": None,
    "version": None,
    "chat_overrides": None,
    "revision": None,
}

MODEL_DEFAULTS = {
    "model": None,
    "trust_remote_code": True,
    "model_max_len": 8192,
    "model_path": None,
    "load_in_4bit": False,
    "load_in_8bit": False,
    "dtype": "auto",
    "use_fast_tokenizer": True,
    "prefix_token_id": None,
    "device_map_option": "auto",
}