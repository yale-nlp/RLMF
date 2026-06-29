### Score Training Samples Per Our Metacognitive Data Selection Approach

## Step 1: Generate K=20 Sampled Answers Per Prompt
# Sample command for Qwen3-8B on PopQA
python ./exp2_rlmf/b_metacog_data_selection/save_sampled_answers.py  --model_name=Qwen/Qwen3-8B --dataset_name=popqa  --sys_prompt_id=0

## Step 2: Get Per-Sample Metacognitive Scores for Qwen3-8B on PopQA
# Sample commands for Qwen3-8B on PopQA -- split into 4 processes which can be run in parallel for efficiency
python ./exp2_rlmf/b_metacog_data_selection/dynamic_score_samples.py  --model_name Qwen/Qwen3-8B --dataset_name popqa --sys_prompt_id 0 --sampled_answers_path ./exp2_rlmf/b_metacog_data_selection/sampled_answers_lists/Qwen_Qwen3_8B/sampled_answers_by_raw_prompt_popqa_sys0.json --slice 0
python ./exp2_rlmf/b_metacog_data_selection/dynamic_score_samples.py  --model_name Qwen/Qwen3-8B --dataset_name popqa --sys_prompt_id 0 --sampled_answers_path ./exp2_rlmf/b_metacog_data_selection/sampled_answers_lists/Qwen_Qwen3_8B/sampled_answers_by_raw_prompt_popqa_sys0.json --slice 1
python ./exp2_rlmf/b_metacog_data_selection/dynamic_score_samples.py  --model_name Qwen/Qwen3-8B --dataset_name popqa --sys_prompt_id 0 --sampled_answers_path ./exp2_rlmf/b_metacog_data_selection/sampled_answers_lists/Qwen_Qwen3_8B/sampled_answers_by_raw_prompt_popqa_sys0.json --slice 2
python ./exp2_rlmf/b_metacog_data_selection/dynamic_score_samples.py  --model_name Qwen/Qwen3-8B --dataset_name popqa --sys_prompt_id 0 --sampled_answers_path ./exp2_rlmf/b_metacog_data_selection/sampled_answers_lists/Qwen_Qwen3_8B/sampled_answers_by_raw_prompt_popqa_sys0.json --slice 3
# Compile results into a single CSV
python ./exp2_rlmf/b_metacog_data_selection/combine_dfs_for_model.py --model_name Qwen/Qwen3-8B  --dataset_name=popqa --pattern=scores_one_per_prompt --base_path=./exp2_rlmf/b_metacog_data_selection/score_dfs
python ./exp2_rlmf/b_metacog_data_selection/combine_dfs_for_model.py --model_name Qwen/Qwen3-8B  --dataset_name=popqa --output_identifier="_smarter"  --base_path=./exp2_rlmf/b_metacog_data_selection/score_dfs

## Step 3: Get Per-Sample Gold Faithful Calibration Scores for Qwen3-8B on PopQA
# This serves as active learning style baseline to which we compare our metacognitive data selection approach; like above, split into 4 processes which can be run in parallel for efficiency
python ./exp2_rlmf/b_metacog_data_selection/dynamic_score_samples_by_gold_faithfulness.py --model_name=Qwen/Qwen3-8B  --dataset_name=popqa  --sys_prompt_id=0  --sampled_answers_path=./exp2_rlmf/b_metacog_data_selection/sampled_answers_lists/Qwen_Qwen3_8B/sampled_answers_by_raw_prompt_popqa_sys0.json --slice=0
python ./exp2_rlmf/b_metacog_data_selection/dynamic_score_samples_by_gold_faithfulness.py --model_name=Qwen/Qwen3-8B  --dataset_name=popqa  --sys_prompt_id=0  --sampled_answers_path=./exp2_rlmf/b_metacog_data_selection/sampled_answers_lists/Qwen_Qwen3_8B/sampled_answers_by_raw_prompt_popqa_sys0.json --slice=1
python ./exp2_rlmf/b_metacog_data_selection/dynamic_score_samples_by_gold_faithfulness.py --model_name=Qwen/Qwen3-8B  --dataset_name=popqa  --sys_prompt_id=0  --sampled_answers_path=./exp2_rlmf/b_metacog_data_selection/sampled_answers_lists/Qwen_Qwen3_8B/sampled_answers_by_raw_prompt_popqa_sys0.json --slice=2
python ./exp2_rlmf/b_metacog_data_selection/dynamic_score_samples_by_gold_faithfulness.py --model_name=Qwen/Qwen3-8B  --dataset_name=popqa  --sys_prompt_id=0  --sampled_answers_path=./exp2_rlmf/b_metacog_data_selection/sampled_answers_lists/Qwen_Qwen3_8B/sampled_answers_by_raw_prompt_popqa_sys0.json --slice=3
# Compile results into a single CSV
python ./exp2_rlmf/b_metacog_data_selection/combine_dfs_for_model.py --model_name Qwen/Qwen3-8B  --dataset_name=popqa --pattern=faithfulness_scores_for_first_sampled_answer --output_identifier="_f_first_answer" --base_path=./exp2_rlmf/b_metacog_data_selection/score_dfs
python ./exp2_rlmf/b_metacog_data_selection/combine_dfs_for_model.py --model_name Qwen/Qwen3-8B  --dataset_name=popqa --pattern=faithfulness_scores_aggr_over_sampled_answers --output_identifier="_f_aggr"  --base_path=./exp2_rlmf/b_metacog_data_selection/score_dfs

