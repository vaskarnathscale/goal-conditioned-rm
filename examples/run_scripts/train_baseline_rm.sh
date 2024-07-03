set -x 

# Do training
read -r -d '' training_commands <<EOF
./examples/train_rm.py \
     --save_path ./ckpt/baseline_reward_model \
     --save_steps -1 \
     --logging_steps 1 \
     --eval_steps -1 \
     --train_batch_size 64 \
     --micro_train_batch_size 1 \
     --pretrain nvidia/OpenMath-CodeLlama-7b-Python-hf \
     --bf16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 3 \
     --learning_rate 9e-6 \
     --dataset ./examples/data/paired_openmathinstruct-1-masked-extra-drop \
     --dataset_probs 1.0 \
     --contrastive_loss_beta 0.5 \
     --unsim_samples 16 \
     --flash_attn \
     --gradient_checkpointing \
     --use_wandb <your wandb key> \
     --wandb_project <project name> \
     --wandb_group <group name> \
     --wandb_org <org name> \
     --reward_model_strategy vanilla 
EOF


if [[ ${1} != "slurm" ]]; then
     deepspeed $training_commands
fi
