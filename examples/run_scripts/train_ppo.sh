set -x

# RLHF Prompts: openmath instruct prompts
read -r -d '' training_commands <<EOF
./examples/train_ppo.py \
    --pretrain_data ./sft_dataset \
    --pretrain nvidia/OpenMath-CodeLlama-7b-Python-hf \
    --reward_pretrain ./reward_model \
    --save_path ./ckpt/trained_policy_model \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --num_episodes $5 \
    --micro_train_batch_size 2 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 4 \
    --rollout_batch_size 512 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 3e-6 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.02 \
    --prompt_data $4 \
    --prompt_data_probs 1.0 \
    --normalize_reward \
    --actor_init_on_gpu \
    --adam_offload \
    --flash_attn \
    --gradient_checkpointing \
    --gamma 1.0 \
    --lambd $9 \
    --use_wandb <your wandb key> \
    --wandb_project <project name> \
    --wandb_group <group name> \
    --wandb_org <org name> \
    --temperature $6 \
    --value_head_strategy $7
    --ptx_coef 0.05 \
    --max_samples 100000
EOF


if [[ ${1} != "slurm" ]]; then
    deepspeed $training_commands
fi
