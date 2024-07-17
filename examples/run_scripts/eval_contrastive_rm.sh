set -x

# Run the evaluation script
python ./examples/eval_rm.py \
    --completions_data_path ./examples/data/base_model_generation\
    --goal_state_path ./examples/data/representation/contrastive_reward_model_goal_state_embedding.npy\
    --reward_model_path ./ckpt/contrastive_reward_model\
    --save_dir ./examples/data/contrastive_rm_eval_data  \
    --filter_threshold 0.0 \
    --benchmarks gsm8k math gsm-hard mawps asdiv algebra222 svamp 
