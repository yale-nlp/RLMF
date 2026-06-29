from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class TrainingConfig:

    ### Logging Args
    report_to: str = "wandb"
    output_dir: str = "./exp2_rlmf/c_rl_training/__models/"
    run_name: str = "Llama3.1_8B_Ins_popqa_RLMF_MDS_BS64_N2000_LR5e-6"
    group_name: str = None
    logging_steps: int = 1
    
    ### Experiment Args
    dataset_names: List[str] = field(default_factory=lambda: ["popqa"])
    dtype: str = "bfloat16"
    num_train_samples: int = 2000
    sys_prompt_id: int = 6
    use_eos_token: bool = True

    ### Data Selection Args
    data_rank_strategy: str = "high+low"    # our best data ranking strategy; alternative options are "high", "low"
    data_rank_mode: str = "meta"            # use "faith" for active learning baseline
    scores_df_path: str = "./exp2_rlmf/b_metacog_data_selection/score_dfs/meta_llama_Meta_Llama_3.1_8B_Instruct/_train_popqa.csv"   # path to the training data with metacognitive scores

    ### RLMF Args
    advantage_form: str = "mf"   # options: None, 'mf', 'mf_minus_mean', 'mf_minus_mean_over_std'
    advantage_piecewise: bool = False           
    advantage_k: float = 1      # additive k for the metacognitive scaling factor applied to advantages
    mc_threshold: float = 0.10    # pred/gold conf threshold for Z calculation
    faithfulness_reward_weight: float = 1.
    
    ### Model Args
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"    # replace with path to pre-SFT'ed model (must use merged weights)
    max_seq_length: int = 512
    load_in_4bit: bool = False
    fast_inference: bool = False
    gpu_memory_utilization: float = 0.5
    
    ### VLLM Args
    use_vllm: bool = True
    vllm_mode: str = "server"
    vllm_server_base_url: str = "http://0.0.0.0:8001"   # replace as needed
    vllm_tensor_parallel_size: int = 1

    ### PEFT Args
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    lora_rank: int = 64

    ### GRPO Args -- more at https://huggingface.co/docs/trl/main/en/grpo_trainer#trl.GRPOConfig
    beta: float = 0.1 # Parameter controlling the deviation from the reference model. Higher β means less deviation from the reference model.
    reward_form: str = "quadratic"     # GRPO faithfulness reward format; choices: 'linear', 'quadratic', 'binary', 'simple_log', 'stretched_log'
    format_rewards_version: str = "new"     # must be 'new'
    reward_weights: List[float] = field(default_factory=lambda: [3.,3.,1.,12.])    
    c_weight: float = 1.      # GRPO correctness reward scale
    f_weight: float = 1.      # GRPO factual calibration reward scale
    scale_rewards: str = "none" # "group" = rewards are scaled by the standard deviation within each group; "batch" = rewards are scaled by the standard deviation across the entire batch, as recommended in the PPO Lite paper; "none" = DR GRPO to avoid question-level difficulty bias
    loss_type: str = "grpo"
    mask_truncated_completions: bool = False   # disrec'ed by unsloth
    num_generations: int = 32

    ### Training Args -- assume gpu count = 4; globl BS is 4* 2 * 8 = 64
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8 

    learning_rate: float = 5e-6
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0 
    # adam_beta1: float = 0.9
    # adam_beta2: float = 0.99
    max_grad_norm: float = 0.1

    # num_train_epochs: int = 1
    max_steps: int = 1500
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_8bit"

    ### Evaluation Args
    per_device_eval_batch_size: int = 32
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int=20
    