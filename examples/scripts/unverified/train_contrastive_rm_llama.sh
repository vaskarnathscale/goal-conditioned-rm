set -x

read -r -d '' training_commands <<EOF
../train_rm.py \
     --save_path ./ckpt/contrastive_codellama7b_v3 \
     --save_steps 2000 \
     --logging_steps 1 \
     --eval_steps 128 \
     --train_batch_size 128 \
     --micro_train_batch_size 16 \
     --pretrain OpenLLMAI/Llama-2-7b-sft-model-ocra-500k \
     --bf16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 2 \
     --learning_rate 9e-6 \
     --l2 0.01 \
     --dataset Anthropic/hh-rlhf \
     --dataset_probs 1.0 \
     --flash_attn \
     --contrastive_loss \
     --contrastive_loss_beta 0.5 \
     --unsim_samples 16 \
     --gradient_checkpointing \
     --use_wandb $WANDB \
     --wandb_project dense_rewards \
     --wandb_group contrastive_rm \
     --wandb_org dylan-slack \
     --s3_save_path test-rm-save-aws-ngc \
     --max_samples 100
EOF


if [[ ${1} != "slurm" ]]; then
    export PATH=$HOME/.local/bin/:$PATH
    deepspeed $training_commands
fi