### Run RL(MF) Training of Models to Improve Their Faithful Calibration
# Note: To reproduce our main results, obtained by combining RLMF with metacognitive data selection, it must be that: (1) the model has already undergone pre-SFT per the code in src/exp2_rlmf/a_pre_sft, and (2) metacognitive scores from the model have already been obtained for all training datapoints in consideration per the code in src/exp2_rlmf/b_metacog_data_selection.

## Sample Training Commands for Llama3.1-8B-Instruct (No Pre-SFT) on PopQA

# Step 1: Serve Judge Model for Consistency Judgments for Online Intrinsic Confidence Estimation
# Here, the nonthinking template is obtained from https://github.com/QwenLM/Qwen3/blob/main/docs/source/assets/qwen3_nonthinking.jinja.
export CUDA_VISIBLE_DEVICES=0; vllm serve Qwen/Qwen3-32B-FP8 --tensor-parallel-size 1  --trust-remote-code  --gpu-memory-utilization 0.8  --max-model-len 4096  --port 8000 --chat-template ./exp2_rlmf/qwen3_nonthinking.jinja

# Step 2: Serve Model Being Trained via TRL+VLLM for GRPO Rollout (see details at https://huggingface.co/docs/trl/main/en/grpo_trainer#option-2-server-mode)
# Note: If running RLMF training for the model after pre-SFT is completed, this command must use the path to the resulting merged model weights instead.
export CUDA_VISIBLE_DEVICES=1; trl vllm-serve --model meta-llama/Meta-Llama-3.1-8B-Instruct  --trust-remote-code  --max-model-len 4096 --port 8001 --gpu-memory-utilization 0.9

# Step 3: Launch Training 
# Note: The sample command here uses the example training configuration provided at src/exp2_rlmf/c_rl_training/sample_config.py; however, this config file can be adapted to use alternative arguments, which are described in comments in the file. If resuming training from a checkpoint, add the flag `--resume_from_checkpoint` and, optionally, the ID of the wandb run to continue from with the flag `--wandb_run_id` followed by the run ID.
export CUDA_VISIBLE_DEVICES=2,3,4,5; torchrun --nproc_per_node=4 ./exp2_rlmf/c_rl_training/grpo.py --config_path  ./exp2_rlmf/c_rl_training/sample_config.py --use_sys_instruction  --judge_hostnum 8000 

# Step 4: Simultaneously Evaluate Checkpoints on In-Domain Test Set
# First, serve another judge model to perform consistency judgments for intrinsic confidence estimation during evaluation.
export CUDA_VISIBLE_DEVICES=6; vllm serve Qwen/Qwen3-32B-FP8 --tensor-parallel-size 1  --trust-remote-code  --gpu-memory-utilization 0.8  --max-model-len 4096  --port 8002 --chat-template ./exp2_rlmf/qwen3_nonthinking.jinja
# Then, run the evaluation on another GPU, making sure that the model directory is consistent in name with the `run_name` argument in the training config file.
export CUDA_VISIBLE_DEVICES=7; bash ./exp2_rlmf/d_evaluate/evaluate_checkpoints.sh  8002 ./exp2_rlmf/c_rl_training/__models/meta_llama_Llama_3.1_8B_Instruct/Llama3.1_8B_Ins_popqa_RLMF_MDS_BS64_N2000_LR5e-6 100 1500 100
# Finally, compile in-domain test evaluation results from all checkpoints to determine which is best-performing.
python ./exp2_rlmf/d_evaluate/compile_results.py --test_results_dir ./exp2_rlmf/c_rl_training/__models/meta_llama_Llama_3.1_8B_Instruct/Llama3.1_8B_Ins_popqa_RLMF_MDS_BS64_N2000_LR5e-6 --dataset_name=popqa

# Step 5: Evaluate Best Checkpoint on Out-of-Distribution Tasks
# Note: This example assumes checkpoint 600 was the best.
# First, run the OOD evaluations.
export CUDA_VISIBLE_DEVICES=7; bash ./exp2_rlmf/d_evaluate/evaluate_all_datasets.sh  8002 ./exp2_rlmf/c_rl_training/__models/meta_llama_Llama_3.1_8B_Instruct/Llama3.1_8B_Ins_popqa_RLMF_MDS_BS64_N2000_LR5e-6 600 
# Then, compile all the results into a single CSV for easy visualization.
python ./exp2_rlmf/d_evaluate/compile_scores_from_all_datasets.py ./exp2_rlmf/c_rl_training/__models/meta_llama_Llama_3.1_8B_Instruct/Llama3.1_8B_Ins_popqa_RLMF_MDS_BS64_N2000_LR5e-6/test_results/checkpoint_600

# Step 6: Compute Final Scores and cMFG*
# Note: This example continues the earlier assumption that checkpoint 600 was best; the `test_results` sub-directory is automatically created during Steps 4 and 5 above.
python ./exp2_rlmf/d_evaluate/rescore.py --results_dir=./exp2_rlmf/c_rl_training/__models/meta_llama_Llama_3.1_8B_Instruct/Llama3.1_8B_Ins_popqa_RLMF_MDS_BS64_N2000_LR5e-6/test_results/checkpoint_600

# Step 7: Additional Analysis
# Note: These example commands assume use of the same best checkpoint as above.
# Option A: Analyze average faithful calibration per size-0.1 intrinsic confidence bin
python ./exp2_rlmf/d_evaluate/analyze_conf_bins.py ./exp2_rlmf/c_rl_training/__models/meta_llama_Llama_3.1_8B_Instruct/Llama3.1_8B_Ins_popqa_RLMF_MDS_BS64_N2000_LR5e-6/test_results/checkpoint_600
# Option B: Plot visualizations of model's faithfulness distribution and other associated properties; this can be run per dataset; the example below does so for PopQA results
python ./exp2_rlmf/d_evaluate/analyze_qualitative.py --model_name "my_model_name_for_plot_title" --input_score_json_path=./exp2_rlmf/c_rl_training/__models/meta_llama_Llama_3.1_8B_Instruct/Llama3.1_8B_Ins_popqa_RLMF_MDS_BS64_N2000_LR5e-6/test_results/checkpoint_600/test_scores_popqa.json

