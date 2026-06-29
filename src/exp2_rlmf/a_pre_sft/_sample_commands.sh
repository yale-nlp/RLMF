### Pre-Fine-Tune Models to Learn Our Custom Output Format

## Sample Commands to Generate Pre-SFT Data for Qwen3-4B

# Note: Requires first serving Qwen3-32B as the judge model for consistency judgments during intrinsic confidence estimation, this can be done via: vllm serve Qwen/Qwen3-32B-FP8 --tensor-parallel-size 1  --trust-remote-code  --gpu-memory-utilization 0.8  --max-model-len 4096  --port 8002 --chat-template ./exp2_rlmf/qwen3_nonthinking.jinja
# Here, the nonthinking template is obtained from https://github.com/QwenLM/Qwen3/blob/main/docs/source/assets/qwen3_nonthinking.jinja

python ./exp2_rlmf/a_pre_sft/get_predictions.py --model_name Qwen/Qwen3-4B-Instruct-2507 --dataset_name popqa  --hostnum 8002 --num_samples=200
python ./exp2_rlmf/a_pre_sft/get_predictions.py --model_name Qwen/Qwen3-4B-Instruct-2507 --dataset_name selfaware  --hostnum 8002 --num_samples=200
python ./exp2_rlmf/a_pre_sft/get_predictions.py --model_name Qwen/Qwen3-4B-Instruct-2507 --dataset_name simpleqa  --hostnum 8002 --num_samples=200
python ./exp2_rlmf/a_pre_sft/get_predictions.py --model_name Qwen/Qwen3-4B-Instruct-2507 --dataset_name umwp  --hostnum 8002 --num_samples=200
python ./exp2_rlmf/a_pre_sft/get_predictions.py --model_name Qwen/Qwen3-4B-Instruct-2507 --dataset_name math  --hostnum 8002 --num_samples=200
python ./exp2_rlmf/a_pre_sft/get_predictions.py --model_name Qwen/Qwen3-4B-Instruct-2507 --dataset_name halueval  --hostnum 8002 --num_samples=200
python ./exp2_rlmf/a_pre_sft/get_predictions.py --model_name Qwen/Qwen3-4B-Instruct-2507 --dataset_name sciq  --hostnum 8002 --num_samples=200
python ./exp2_rlmf/a_pre_sft/get_predictions.py --model_name Qwen/Qwen3-4B-Instruct-2507 --dataset_name arc_challenge  --hostnum 8002 --num_samples=200
python ./exp2_rlmf/a_pre_sft/get_predictions.py --model_name Qwen/Qwen3-4B-Instruct-2507 --dataset_name superglue  --hostnum 8002 --num_samples=200
python ./exp2_rlmf/a_pre_sft/get_predictions.py --model_name Qwen/Qwen3-4B-Instruct-2507 --dataset_name mmlu  --hostnum 8002 --num_samples=200

## Sample Commands to Run Pre-SFT Training for Qwen3-4B
# Directly run the cells in ./exp2_rlmf/a_pre_sft/qwen3_4b.ipynb; this will format and prepare the data, train the model, and save it for use with the code in src/exp2_rlmf/c_rl_training
# Note: You may need to install the `ipykernel` package first.