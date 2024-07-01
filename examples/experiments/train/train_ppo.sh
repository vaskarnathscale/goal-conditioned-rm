set -x

# Unset AWS_PROFILE if running in NVCR
if [ -n "$NVCR" ]; then
    unset AWS_PROFILE
fi

# OpenInstruct Paired Dataset
# dataset_s3_path=s3://scale-ml/content-understanding-ml/paired_openmathinstruct-1-masked-extra-drop
# aws s3 cp $dataset_s3_path ./paired_openmathinstruct-1-masked-extra-drop --recursive
dataset_s3_path=s3://scale-ml/content-understanding-ml/$4
aws s3 cp $dataset_s3_path ./$4 --recursive

# Contrastive Reward Model, highest perf as of 3/13
rm_s3_path=s3://scale-ml/content-understanding-ml/$3
# rm_s3_path=s3://scale-ml/content-understanding-ml/baseline_rm_codellama_7b_final
aws s3 cp $rm_s3_path ./reward_model --recursive

# OpneMathInstruct SFT Positives
sft_dataset=s3://scale-ml/content-understanding-ml/openmathinstruct-positives-with-code-executions
aws s3 cp $sft_dataset ./sft_dataset --recursive

# Pretrain data: original openmath instruct solutions
# RLHF Prompts: openmath instruct prompts
read -r -d '' training_commands <<EOF
./examples/train_ppo.py \
    --pretrain_data ./sft_dataset \
    --pretrain nvidia/OpenMath-CodeLlama-7b-Python-hf \
    --reward_pretrain ./reward_model \
    --save_path ./ckpt/7b_llama \
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
    --use_wandb $WANDB \
    --gamma 1.0 \
    --lambd $9 \
    --wandb_project goal_rewards \
    --wandb_group ppo_vanilla_rm_constrative \
    --wandb_org gen-ai \
    --temperature $6 \
    --value_head_strategy $7
    --ptx_coef 0.05 \
    --s3_save_path vanilla-ppo-rm-$3-dataset-$4-episode-$5-t-$6-vhs-$7-v-$8-lambd-$9 \
    --max_samples 100000
EOF


if [[ ${1} != "slurm" ]]; then
    deepspeed $training_commands
fi
