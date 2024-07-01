set -x 

# Unset AWS_PROFILE if running in NVCR
if [ -n "$NVCR" ]; then
    unset AWS_PROFILE
fi

# Copy the reward model from S3
# the local path can also be a huggingface model name, and
# the script will download the model from the hub
s3_path=s3://scale-ml/content-understanding-ml/test-rm-save-aws-ngc
aws s3 cp $s3_path ./reward_model --recursive

# Copy the dataset from S3
# the local path can also be a huggingface dataset name
# e.g., Anthropic/hh-rlhf
dataset_s3_path=s3://scale-ml/content-understanding-ml/hh-rlhf
aws s3 cp $dataset_s3_path ./hh-rlhf --recursive

# Do training
read -r -d '' training_commands <<EOF
../train_rm.py \
     --save_path ./ckpt/7b_llama \
     --save_steps -1 \
     --logging_steps 1 \
     --eval_steps -1 \
     --train_batch_size 128 \
     --micro_train_batch_size 1 \
     --pretrain ./reward_model \
     --bf16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 3 \
     --learning_rate 9e-6 \
     --dataset ./hh-rlhf \
     --dataset_probs 1.0 \
     --contrastive_loss \
     --contrastive_loss_beta 0.5 \
     --unsim_samples 16 \
     --flash_attn \
     --gradient_checkpointing \
     --use_wandb $WANDB \
     --wandb_project ngc_test \
     --wandb_group ngc_test \
     --wandb_org gen-ai \
     --s3_save_path test-rm-save-aws-ngc-2 \
     --max_samples 100 \
EOF
# note max_samples is set to a very small value, you will need to change this

if [[ ${1} != "slurm" ]]; then
     deepspeed $training_commands
fi
