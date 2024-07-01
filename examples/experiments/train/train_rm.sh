# llama7b contrastive RM training
set -x 

# Unset AWS_PROFILE if running in NVCR
# if [ -n "$NVCR" ]; then
#     unset AWS_PROFILE
# fi

huggingface-cli login --token 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

dataset_s3_path=s3://scale-ml/content-understanding-ml/$4
aws s3 cp $dataset_s3_path ./$4 --recursive

# Do training
read -r -d '' training_commands <<EOF
./examples/train_rm.py \
     --save_path ./ckpt/7b_llama \
     --save_steps -1 \
     --logging_steps 1 \
     --eval_steps ${10} \
     --train_batch_size $6 \
     --micro_train_batch_size 1 \
     --pretrain $5 \
     --bf16 \
     --max_epochs ${11} \
     --max_len ${12} \
     --zero_stage 3 \
     --learning_rate ${13} \
     --dataset ./$4 \
     --dataset_probs 1.0 \
     --contrastive_loss_beta $7 \
     --unsim_samples 16 \
     --flash_attn \
     --gradient_checkpointing \
     --use_wandb $WANDB \
     --wandb_project contrastive_rm \
     --wandb_group ngc_test \
     --wandb_org gen-ai \
     --s3_save_path rms-$1-cs-$2-vhs-$3-ds-$4-ptxs-$8-bs-$6-beta-$7-v-$9-eps-${11}-lr-${13}-ssp-${14}-gsp-${15} \
     --reward_model_strategy $1 \
     --contrastive_strategy $2 \
     --value_head_strategy $3 \
     --source_state_percentile ${14} \
     --goal_state_percentile ${15} 
EOF


if [[ ${1} != "slurm" ]]; then
     deepspeed $training_commands
fi

aws s3 sync ./ckpt/7b_llama s3://scale-ml/content-understanding-ml/rms-$1-cs-$2-vhs-$3-ds-$4-ptxs-$8-bs-$6-beta-$7-v-$9-eps-${11}-lr-${13}-ssp-${14}-gsp-${15}
