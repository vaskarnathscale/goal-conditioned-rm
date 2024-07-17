set -x

python ./examples/metrics.py \
    --model_name "contrastive_reward_model" \
    --eval_results_dir ./examples/data/contrastive_rm_eval_data \
    --filter_threshold 0.0 \
    --benchmarks gsm8k math gsm-hard mawps asdiv algebra222 svamp \