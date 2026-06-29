### Baseline Faithful Calibration with MetaFaith Prompting (Liu et al., 2025)

# Note: Mode 2 scoring requires first serving Qwen3-32B as the judge model for consistency judgments during intrinsic confidence estimation, this can be done via: vllm serve Qwen/Qwen3-32B-FP8 --tensor-parallel-size 1  --trust-remote-code  --gpu-memory-utilization 0.8  --max-model-len 4096  --port 8002 --chat-template ./exp1_metafaith/qwen3_nonthinking.jinja
# Here, the nonthinking template is obtained from https://github.com/QwenLM/Qwen3/blob/main/docs/source/assets/qwen3_nonthinking.jinja


# Sample Run + Score Commands for Gemini-3-Flash
bash ./exp1_metafaith/run_inference.sh gemini-3-flash-preview 1000
bash ./exp1_metafaith/run_score.sh gemini-3-flash-preview 1000 1
bash ./exp1_metafaith/run_score.sh gemini-3-flash-preview 1000 2 8002
bash ./exp1_metafaith/run_score.sh gemini-3-flash-preview 1000 3

# Sample Run + Score Commands for Qwen3-1.7B
bash ./exp1_metafaith/run_inference.sh Qwen/Qwen3-1.7B 1000
bash ./exp1_metafaith/run_score.sh Qwen/Qwen3-1.7B 1000 1
bash ./exp1_metafaith/run_score.sh Qwen/Qwen3-1.7B 1000 2 8002
bash ./exp1_metafaith/run_score.sh Qwen/Qwen3-1.7B 1000 3

# Sample Result Compilation Commands
python ./exp1_metafaith/compile_metrics.py --dir ./exp1_metafaith/_results/gemini_3_flash_preview
python ./exp1_metafaith/compile_metrics.py --dir ./exp1_metafaith/_results/Qwen_Qwen3_1.7B
