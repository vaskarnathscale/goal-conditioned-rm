set -x

# Run the evaluation script
python ./examples/eval_rm.py \
    --completions_data_path ./examples/data/base_model_generation\
    --reward_model_path ./ckpt/baseline_reward_model\
    --save_dir ./examples/data/baseline_rm_eval_data  \
    --benchmarks gsm8k math gsm-hard mawps asdiv algebra222 svamp \
    --vanilla 
