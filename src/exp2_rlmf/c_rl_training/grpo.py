import os,json
import argparse
import time
import numpy as np
import pandas as pd
from functools import partial
from termcolor import colored
import torch

from src.exp2_rlmf.c_rl_training.rewards import *
from src.exp2_rlmf.utils.utils import load_config, save_config
from src.exp2_rlmf.utils.data_utils import get_dataset


torch.manual_seed(42)

dtypes = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16
}

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", type=str, default=None, help="Path to config file")
    parser.add_argument("--unsloth", default=False, action="store_true")
    parser.add_argument("--use_sys_instruction", default=False, action="store_true", help="Use instruction format for training")
    parser.add_argument("--merge_lora", default=False, action="store_true", help="After training merge LoRA weights into the model and save another copy")
    parser.add_argument("--resume_from_checkpoint", default=False, action="store_true")
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--judge_hostnum", type=int, default=None)
    parser.add_argument("--judge_host", type=str, default=None)
    return parser.parse_args()

def main():

    ### Parse Run Arguments
    args = parse_args()    

    # Load Config
    config = load_config(args.config_path)

    # Initialize WandB
    # wandb.login()
    import wandb
    model_identifier = config.model_name.replace("/", "-").replace("-", "_")
    run_name = config.run_name if config.run_name else model_identifier
    group_name = config.group_name if config.group_name else run_name
    if args.resume_from_checkpoint==True:
        if int(os.environ["LOCAL_RANK"])==0 and args.wandb_run_id: 
            wandb.init(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY"), name=run_name, group=group_name, resume="must", id=args.wandb_run_id)
    else:
        wandb.init(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY"), name=run_name, group=group_name)  # id=run_name -- must be unique
    output_dir = os.path.join(config.output_dir, model_identifier, run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save Config
    if not args.resume_from_checkpoint: 
        save_config(config, output_dir, args.config_path)

    ### Load Model
    dtype = dtypes[config.dtype]
    if args.unsloth:
        from unsloth import FastLanguageModel, PatchFastRL
        PatchFastRL("GRPO", FastLanguageModel)
        import torch
        from vllm import SamplingParams
        from trl import GRPOConfig #, GRPOTrainer
        from rlmf_trainer import RLMFTrainer
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        from safetensors import safe_open

        print(colored("Loading model!", "cyan"))
        orig_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = config.model_name,
            max_seq_length = config.max_seq_length,
            dtype = dtype,
            load_in_4bit = config.load_in_4bit,
            fast_inference = config.fast_inference, # Enable vLLM fast inference
            max_lora_rank = config.lora_rank,
            gpu_memory_utilization = config.gpu_memory_utilization, # Reduce if out of memory
            # attn_implementation="flash_attention_2",
            use_gradient_checkpointing=False, # "unsloth" Reduces memory usage
            # load_in_fp8 = True, # Float8 RL / GRPO!
            # enforce_eager=True if config.fast_inference==True else False,
            # enable_flash_attn=True,
            # chunked_prefill_enabled=False,
            # enable_chunked_prefill=False,
        )
    else:
        import torch
        from vllm import SamplingParams
        from trl import GRPOConfig #, GRPOTrainer
        from rlmf_trainer import RLMFTrainer
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        from safetensors import safe_open
        if 'emma' in config.model_name:
            orig_model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=dtype, attn_implementation='eager') # device_map='auto'
        else:
            orig_model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=dtype) # device_map='auto'
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
    ### Set Pad Token
    if 'lama-3' in config.model_name:
        tokenizer.pad_token_id = 128009
        # tokenizer.pad_token_id = 128001
    else:
        tokenizer.pad_token = tokenizer.eos_token

    ### Load PEFT Version of Model  
    print(colored("Loading PEFT model!", "cyan"))
    if config.lora_rank is not None:
        if args.unsloth:
            model = FastLanguageModel.get_peft_model(
                orig_model,
                r = config.lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                target_modules = config.target_modules if hasattr(config, 'target_modules') else ["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_alpha = config.lora_alpha if hasattr(config, 'lora_alpha') else config.lora_rank,
                lora_dropout = config.lora_dropout if hasattr(config, 'lora_dropout') else 0.05, # Supports any, but = 0 is optimized
                bias = "none",    # Supports any, but = "none" is optimized
                random_state = 42,
                use_gradient_checkpointing=False, #"unsloth"
            )
        else:
            from peft import LoraConfig, get_peft_model, TaskType
        
            # Configure LoRA
            lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha if hasattr(config, 'lora_alpha') else config.lora_rank,
                lora_dropout=config.lora_dropout if hasattr(config, 'lora_dropout') else 0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=config.target_modules if hasattr(config, 'target_modules') else ["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            
            # Apply LoRA to the model
            if hasattr(config, 'lora_path') and config.lora_path is not None:
                print(f"Loading LoRA weights from: {config.lora_path}")
                model = PeftModel.from_pretrained(orig_model, config.lora_path)
            else:
                model = get_peft_model(orig_model, lora_config)
            
        # Count Trainable vs Total Parameters
        model.print_trainable_parameters()

    ### Summarize Training vs Total Parameters 
    print(colored("PEFT Statistics", "green"))
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable Params: {trainable_params:,} ({100 * trainable_params / total_params:.4f}% of total)")
    print(f"Total Params: {total_params:,}")

    ### Load Dataset
    print(colored("Loading dataset!", "cyan"))
    train_dataset, val_dataset, test_dataset = get_dataset(
        dataset_names=config.dataset_names, 
        tokenizer=tokenizer, 
        num_train_samples=config.num_train_samples if config.num_train_samples else None,
        sys_prompt_id=config.sys_prompt_id if config.sys_prompt_id else 1,
        use_eos_token=config.use_eos_token,
        rank=config.data_rank_strategy if hasattr(config, "data_rank_strategy") else None, 
        scores_df_path=config.scores_df_path if hasattr(config, "scores_df_path") else (os.path.join(f"./exp2_rlmf/b_metacog_data_selection/score_dfs", "meta-llama/Meta-Llama-3.1-8B-Instruct".replace("-", "_").replace("/", "_")) if "llama" in config.model_name and "3.1" in config.model_name 
        else (os.path.join(f"./exp2_rlmf/b_metacog_data_selection/score_dfs", "Qwen/Qwen3-8B".replace("-", "_").replace("/", "_")) if "wen3" in config.model_name else os.path.join(f"./exp2_rlmf/b_metacog_data_selection/score_dfs", config.model_name.replace("-", "_").replace("/", "_")))),
        sort_mode_if_ranking=config.data_rank_mode if hasattr(config, "data_rank_mode") else "meta",
        mc_threshold=None if config.sys_prompt_id not in [11,13] else config.mc_threshold,
        use_length_direction=config.use_length_dirxn_during_training if hasattr(config, "use_length_dirxn_during_training") else False,
    )

    ### Prepare Data
    if "Qwen" in config.model_name:
        # print("HERE!!!!!!" * 50)
        tokenized = train_dataset.map(
            lambda x: {"tokens" : tokenizer.apply_chat_template(x["prompt"],
            add_generation_prompt = True, tokenize = True, enable_thinking=False)},
            batched = True,
        )
        print(tokenized[0])
    else: 
        tokenized = train_dataset.map(
            lambda x: {"tokens" : tokenizer.apply_chat_template(x["prompt"],
            add_generation_prompt = True, tokenize = True)},
            batched = True,
        )
    tokenized = tokenized.map(lambda x: {"L" : len(x["tokens"])})
    max_length_tokens = int(np.quantile(tokenized["L"], 0.9))
    print(colored("Training Data Max Length (Tokens):", "yellow"), max_length_tokens)

    ### Filter Data For Samples < 90% Max Length
    # train_dataset = train_dataset.select(np.where(np.array(tokenized["L"]) <= max_length_tokens)[0])
    del tokenized
    max_prompt_length = max_length_tokens + 1 # + 1 just in case!
    max_completion_length = config.max_seq_length - max_prompt_length

    ### Set Up Training Args
    if "Qwen" in config.model_name:
        vllm_sampling_params = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0,
            "seed": 42,
            "stop": [tokenizer.eos_token],
            "include_stop_str_in_output": True,
            # "chat_template_kwargs": {"enable_thinking": False}
        }
    else: 
        vllm_sampling_params = {
            "min_p": 0.1,
            "seed": 42,
            "stop": [tokenizer.eos_token],
            "include_stop_str_in_output": True,
        }
    use_bf16 = True if dtype == torch.bfloat16 and torch.cuda.is_bf16_supported() else False
    use_fp16 = True if dtype == torch.float16 or use_bf16==False else False

    training_args = GRPOConfig(

        ### VLLM Args
        use_vllm = config.use_vllm,
        vllm_mode = config.vllm_mode,
        vllm_gpu_memory_utilization = config.gpu_memory_utilization,
        vllm_tensor_parallel_size = config.vllm_tensor_parallel_size,
        vllm_server_base_url = config.vllm_server_base_url if args.judge_host is None else config.vllm_server_base_url.replace("0.0.0.0", args.judge_host),

        ### Data Args
        max_prompt_length = max_prompt_length,
        max_completion_length = max_completion_length,
        shuffle_dataset = True,

        ### Generation Args
        generation_kwargs=vllm_sampling_params,

        ### GRPO Args
        beta=config.beta,
        reward_weights=config.reward_weights,
        scale_rewards=config.scale_rewards,
        loss_type=config.loss_type,
        mask_truncated_completions=config.mask_truncated_completions,
        num_generations = config.num_generations, # Must evenly divide effective batch size

        ### Logging Args
        report_to=config.report_to,
        run_name=run_name,
        output_dir=output_dir,
        log_completions=True,
        num_completions_to_print=None, # print all
        logging_steps=config.logging_steps,

        ### Training Args
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=False,

        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        # adam_beta1=config.adam_beta1,
        # adam_beta2=config.adam_beta2,
        max_grad_norm=config.max_grad_norm,
        
        # num_train_epochs=config.num_train_epochs if config.num_train_epochs is not None else -1,
        max_steps=config.max_steps,
        lr_scheduler_type=config.lr_scheduler_type,
        optim=config.optim,
        
        bf16=use_bf16,
        fp16=use_fp16,

        ### Evaluation Args
        bf16_full_eval=use_bf16,
        fp16_full_eval=use_fp16,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        eval_accumulation_steps=1,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit if hasattr(config, "save_total_limit") and config.save_total_limit is not None else 20,
    )
    
    ### Set Up Reward Functions
    print(colored("Loading reward functions!", "cyan"))
    strict_format_func = partial(
        format_reward_func_new, 
        sys9=(config.sys_prompt_id==9),
        format_version=config.format_rewards_version, 
        metascore_as_percentage=config.get_mc_prediction_as_percentage if hasattr(config, "get_mc_prediction_as_percentage") else False,
    )
    soft_format_func = partial(
        approximate_format_reward_func_new, 
        sys9=(config.sys_prompt_id==9), 
        sys_11_12_13=(config.sys_prompt_id in [11, 12, 13]),
        format_version=config.format_rewards_version, 
        metascore_as_percentage=config.get_mc_prediction_as_percentage if hasattr(config, "get_mc_prediction_as_percentage") else False,
        length_dirxn_used=config.use_length_dirxn_during_training if hasattr(config, "use_length_dirxn_during_training") else False,
    )
    strict_format_func.__name__ = "format_reward_func_new"
    soft_format_func.__name__ = "approximate_format_reward_func_new"
    
    get_meta_score_from_completions=True if config.sys_prompt_id in [11, 12, 13] else False
    if get_meta_score_from_completions==False:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    else: 
        tokenizer=None
    grpo_faithfulness_reward_func = partial(
        grpo_faithfulness_reward, 
        reward_form=config.reward_form,
        format_version=config.format_rewards_version,
        hostname=None if args.judge_hostnum is None else (f"http://localhost:{args.judge_hostnum}/v1" if args.judge_host is None else f"http://{args.judge_host}:{args.judge_hostnum}/v1"),
        # use_orig_confidence=config.use_orig_confidence if hasattr(config, "use_orig_confidence") else False,
        use_orig_confidence=False,
        faithfulness_reward_weight=config.faithfulness_reward_weight if hasattr(config, "faithfulness_reward_weight") else 1.0,  # default this only
        mc_reward_weight=config.mc_reward_weight if hasattr(config, "mc_reward_weight") else 0.0,                     # default not this
        mc_threshold=config.mc_threshold if hasattr(config, "mc_threshold") else 0.05,
        get_mc_prediction_as_percentage=config.get_mc_prediction_as_percentage if hasattr(config, "get_mc_prediction_as_percentage") else (False if config.sys_prompt_id in [11, 12, 13] else (True if config.sys_prompt_id==14 else False)),
        get_meta_score_from_completions=get_meta_score_from_completions,
        tokenizer=tokenizer,
        mc_reward_start_threshold=config.mc_reward_start_threshold if hasattr(config, "mc_reward_start_threshold") else 0.,
        inference_hostnum=int(config.vllm_server_base_url.split(":")[-1]) if config.vllm_server_base_url is not None else -1,
        return_mc_score_for_advantage=True if hasattr(config, 'advantage_form') else False,
        length_dirxn_used=config.use_length_dirxn_during_training if hasattr(config, "use_length_dirxn_during_training") else False,
    )
    correctness_factuality_reward_func = partial(
        correctness_and_factuality_reward_func, 
        c_weight=config.c_weight, 
        f_weight=config.f_weight,
        format_version=config.format_rewards_version,
        hostname=None if args.judge_hostnum==None else (f"http://localhost:{args.judge_hostnum}/v1" if args.judge_host is None else f"http://{args.judge_host}:{args.judge_hostnum}/v1"),
        length_dirxn_used=config.use_length_dirxn_during_training if hasattr(config, "use_length_dirxn_during_training") else False,
    )
    grpo_faithfulness_reward_func.__name__ = "grpo_faithfulness_reward_func"
    correctness_factuality_reward_func.__name__ = "correctness_factuality_reward_func"
    if len(config.reward_weights)==4:
        reward_funcs = [
            strict_format_func,
            soft_format_func,
            correctness_factuality_reward_func,
            grpo_faithfulness_reward_func, 
        ]
    else:
        raise ValueError("Incorrect number of rewards specified!")

    ### Set Up & Run Trainer
    print(colored("Starting training!", "cyan"))
    # optional: new_dataset = dataset.train_test_split(test_size = 0.01)
    trainer = RLMFTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        advantage_form=config.advantage_form if hasattr(config, 'advantage_form') else None,
        advantage_piecewise=config.advantage_piecewise if hasattr(config, 'advantage_piecewise') else None,
        advantage_k=config.advantage_k if hasattr(config, 'advantage_k') else None,
    )
    start_time = time.time()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint) # Model checkpoints should be saved automatically by the trainer
    end_time = time.time()
    total_time = (end_time - start_time) / 60.

    ### Save Runtime
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_path = os.path.join(output_dir, f"runtime_{timestamp}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Training Time (minutes): {total_time}")
    print(f"Training Time (minutes): {total_time}")
    wandb.finish()

if __name__ == "__main__":
    main()
