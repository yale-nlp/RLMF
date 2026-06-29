### Map Faithful Numerical Confidence Expressions to Linguistic (Natural Language) Uncertainty
# Note: The example commands provided here assume the same model and setup as the sample commands provided in src/exp2_rlmf/c_rl_training/_sample_commands.sh and operate over the results of that script.


# Step 1: Run rewriting step for all datasets
# Note: Set the last arg to 0 to skip faithfulness scoring and only complete the numerical -> linguistic confidence transformation
bash ./exp3_rewriting/rewrite.sh ./exp2_rlmf/c_rl_training/__models/meta_llama_Llama_3.1_8B_Instruct/Llama3.1_8B_Ins_popqa_RLMF_MDS_BS64_N2000_LR5e-6/test_results/checkpoint_600  gemini-2.5-flash-lite  all  20  1

# Step 2: Compile results into easily visualizable form
# Note: This command must specify the same rewriting model, rewriting mode, and # of hedges per confidence bin as in the Step 1 command
python ./exp3_rewriting/compile_linguistic_scores.py --dir=./exp2_rlmf/c_rl_training/__models/meta_llama_Llama_3.1_8B_Instruct/Llama3.1_8B_Ins_popqa_RLMF_MDS_BS64_N2000_LR5e-6/test_results/checkpoint_600 --mode all --bin_size 20 --model gemini-2.5-flash-lite
