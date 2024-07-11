# llama7b contrastive RM training
set -x 

dataset_s3_path=s3://your-s3-bucket/hh-rlhf
aws s3 cp $dataset_s3_path ./hh-rlhf --recursive

# Do training
read -r -d '' training_commands <<EOF
./examples/train_rm.py \
     --logging_steps 1 \
     --eval_steps 1000 \
     --train_batch_size 64 \
     --micro_train_batch_size 1 \
     --pretrain meta-llama/Meta-Llama-3-8B-Instruct \
     --bf16 \
     --max_epochs 1 \
     --max_len 4096 \
     --zero_stage 3 \
     --learning_rate $3 \
     --dataset ./hh-rlhf \
     --dataset_probs 1.0 \
     --contrastive_loss \
     --contrastive_loss_beta $1 \
     --unsim_samples 16 \
     --flash_attn \
     --gradient_checkpointing \
     --use_wandb $WANDB \
     --s3_save_path helpful-harmless-llama-3-8b-rm-0511/beta-$1-lr-$3 \
     --seed $2
EOF


if [[ ${1} != "slurm" ]]; then
     deepspeed $training_commands
fi
