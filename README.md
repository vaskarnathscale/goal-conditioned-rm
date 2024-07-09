# Learning Goal-Conditioned Representations for Language Reward Models

This repository is the official implementation of [Learning Goal-Conditioned Representations for Language Reward Models](https://arxiv.org/abs/2030.12345). 

<div align="center">
	<img src="https://github.com/vaskarnathscale/goal-conditioned-rm/assets/97542499/ee4046ce-ecb2-4265-8fcf-b55dda89df46">
</div>


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Build

To build openrlhf:

```build
conda create -n openrlhf python=3.10
pip install packaging ninja
pip3 install torch
ninja --version
echo $? # output: 0
# install flash-attn: may take some time.
# For network error: you can download specified version from https://github.com/Dao-AILab/flash-attention/releases.
pip install flash-attn==2.4.2
cd examples/build_scripts
./build_openrlhf.sh
```

## Training

In this section, we provide several commands to run the training. We also provide various example scripts under examples/run_scripts. All the runs need to happen from the repository's root directory.

For the baseline reward model training, you can run the script `examples/run_scripts/train_baseline_rm.sh`:

```
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
     --dataset <path to preference ranking dataset> \
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
```

For the goal conditioned reward model training, you can run the script `examples/run_scripts/train_contrastive_rm.sh`:

```
# Do training
read -r -d '' training_commands <<EOF
./examples/train_rm.py \
     --save_path ./ckpt/contrastive_reward_model \
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
     --dataset <path to preference ranking dataset> \
     --dataset_probs 1.0 \
     --contrastive_loss_beta 0.5 \
     --unsim_samples 16 \
     --flash_attn \
     --gradient_checkpointing \
     --use_wandb <your wandb key> \
     --wandb_project <project name> \
     --wandb_group <group name> \
     --wandb_org <org name> \
     --reward_model_strategy contrastive \
     --contrastive_strategy cosine \
     --value_head_strategy linear \
     --source_state_percentile 0 \
     --goal_state_percentile 0
EOF


if [[ ${1} != "slurm" ]]; then
     deepspeed $training_commands
fi
```

## Representations

Once you have a goal-conditioned trained reward model, to get the goal state representations from all the preferred response from the preference ranking dataset, you can run the script `examples/run_scripts/get_representation.sh`:

```
python ./examples/get_representation.py \
    --dataset-path <path to preference ranking dataset> \
    --reward-model-path <path to reward model> \
    --save-dir <path to save dir>
```

This script will provide a numpy file consisting of a tensor of shape (1, hidden_size) representing the mean representation across all the preferred responses. It will be saved in save_dir.

## Evaluation

### Run Evaluations

To evaluate the baseline reward model, you can run the script `examples/run_scripts/eval_basline_rm.sh`:

```
# Run the evaluation script
python ./examples/eval_rm.py \
    --completions_data_path <path to basemodel generations> \
    --reward_model_path <path to reward model>
    --save_dir ./examples/data/rm_eval_data  \
    --filter_threshold 0.0 \
    --benchmarks gsm8k math gsm-hard mawps asdiv algebra222 svamp \
    --vanilla 
```

To evaluate the contrastive reward model, you can run the script `examples/run_scripts/eval_contrastive_rm.sh`:

```
# Run the evaluation script
python ./examples/eval_rm.py \
    --completions_data_path <path to basemodel generations> \
    --goal_state_path <path to goal state representation (following Representations section to get this goal state)>
    --reward_model_path <path to reward model>
    --save_dir ./examples/data/rm_eval_data \
    --filter_threshold 0.0 \
    --benchmarks gsm8k math gsm-hard mawps asdiv algebra222 svamp 
```

These scripts will save all the reward model inference into the folder. There will one sub-directory for each math benchmark. Each benchmark directory will have a `greedy` sub-directory which contains the json file with the full reward model outputs. For the GSM8K and MATH benchmarks that will contain an additional `all` sub-directory, which will contain the eval results on the full 50 sampled generations. 

### Print Metrics

Once the evaluation script is run, you can run the following scripts:

`examples/run_scripts/print_contrastive_metrics.sh` and `examples/run_scripts/print_baseline_metrics.sh`

to print our the metrics of the contrastive reward model performance and the metrics for the baseline reward model performance.
