set -x

# Unset AWS_PROFILE if running in NVCR
if [ -n "$NVCR" ]; then
    unset AWS_PROFILE
fi

# Copy the model from S3
# the local path can also be a huggingface model name, and
# the script will download the model from the hub
s3_path=s3://scale-ml/content-understanding-ml/test-rm-save-aws-ngc
aws s3 cp $s3_path ./reward_model --recursive


read -r -d '' training_commands <<EOF
../train_ppo.py \
    --pretrain OpenLLMAI/Llama-2-7b-sft-model-ocra-500k \
    --reward_pretrain ./reward_model \
    --save_path ./ckpt/7b_llama \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --micro_train_batch_size 2 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 4 \
    --rollout_batch_size 1024 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data Open-Orca/OpenOrca \
    --prompt_data_probs 1.0 \
    --normalize_reward \
    --actor_init_on_gpu \
    --adam_offload \
    --flash_attn \
    --gradient_checkpointing \
    --use_wandb $WANDB \
    --wandb_project ngc_test \
    --wandb_group ngc_test \
    --wandb_org gen-ai \
    --s3_save_path test-ppo-save-aws-ngc-2 \
    --max_samples 100
EOF
# note max_samples is set to a very small value, you will need to change this

if [[ ${1} != "slurm" ]]; then
    deepspeed $training_commands
fi
