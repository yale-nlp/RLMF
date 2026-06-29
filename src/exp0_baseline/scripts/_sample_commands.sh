### Baseline Faithful Calibration Evaluation (Original Models)

# Get Predictions for All Models
bash ./exp0_baseline/scripts/run_all_datasets_vllm.sh  Qwen/Qwen3-0.6B  basic 
bash ./exp0_baseline/scripts/run_all_datasets_vllm.sh  Qwen/Qwen3-0.6B  blank 
bash ./exp0_baseline/scripts/run_all_datasets_vllm.sh  Qwen/Qwen3-0.6B  genuine 
bash ./exp0_baseline/scripts/run_all_datasets_vllm.sh  Qwen/Qwen3-0.6B  human 
bash ./exp0_baseline/scripts/run_all_datasets_vllm.sh  Qwen/Qwen3-0.6B  perception 

bash ./exp0_baseline/scripts/run_all_datasets_vllm.sh  Qwen/Qwen3-1.7B  basic 
bash ./exp0_baseline/scripts/run_all_datasets_vllm.sh  Qwen/Qwen3-1.7B  blank 
bash ./exp0_baseline/scripts/run_all_datasets_vllm.sh  Qwen/Qwen3-1.7B  genuine 
bash ./exp0_baseline/scripts/run_all_datasets_vllm.sh  Qwen/Qwen3-1.7B  human 
bash ./exp0_baseline/scripts/run_all_datasets_vllm.sh  Qwen/Qwen3-1.7B  perception 

bash ./exp0_baseline/scripts/run_all_datasets_vllm.sh  Qwen/Qwen3-4B-Instruct-2507  basic 
bash ./exp0_baseline/scripts/run_all_datasets_vllm.sh  Qwen/Qwen3-4B-Instruct-2507  blank 
bash ./exp0_baseline/scripts/run_all_datasets_vllm.sh  Qwen/Qwen3-4B-Instruct-2507  genuine 
bash ./exp0_baseline/scripts/run_all_datasets_vllm.sh  Qwen/Qwen3-4B-Instruct-2507  human 
bash ./exp0_baseline/scripts/run_all_datasets_vllm.sh  Qwen/Qwen3-4B-Instruct-2507  perception 

bash ./exp0_baseline/scripts/run_all_datasets_vllm.sh  Qwen/Qwen3-8B  basic 
bash ./exp0_baseline/scripts/run_all_datasets_vllm.sh  Qwen/Qwen3-8B  blank 
bash ./exp0_baseline/scripts/run_all_datasets_vllm.sh  Qwen/Qwen3-8B  genuine 
bash ./exp0_baseline/scripts/run_all_datasets_vllm.sh  Qwen/Qwen3-8B  human 
bash ./exp0_baseline/scripts/run_all_datasets_vllm.sh  Qwen/Qwen3-8B  perception 

# Score Predictions for All Models
# Note: Model names can be changed in src/exp0_baseline/scripts/run_score_vllm.sh
bash ./exp0_baseline/scripts/run_score_vllm.sh  popqa 
bash ./exp0_baseline/scripts/run_score_vllm.sh  simpleqa 
bash ./exp0_baseline/scripts/run_score_vllm.sh  selfaware 
bash ./exp0_baseline/scripts/run_score_vllm.sh  sciq 
bash ./exp0_baseline/scripts/run_score_vllm.sh  math 
bash ./exp0_baseline/scripts/run_score_vllm.sh  umwp 
bash ./exp0_baseline/scripts/run_score_vllm.sh  mmlu 
bash ./exp0_baseline/scripts/run_score_vllm.sh  halueval 
bash ./exp0_baseline/scripts/run_score_vllm.sh  arc_challenge 
bash ./exp0_baseline/scripts/run_score_vllm.sh  superglue 