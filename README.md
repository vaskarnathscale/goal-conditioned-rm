# Learning Goal-Conditioned Representations for Language Reward Models

This repository is the official implementation of [Learning Goal-Conditioned Representations for Language Reward Models](https://arxiv.org/abs/2407.13887). 

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

## Data Setup

To set up the datasets, simply run the script `examples/run_scripts/data_setup.sh`. This will setup two datasets. The first one is the preference ranking dataset, which is a subset of the [OpenMathInstruct](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1) dataset that pairs correct and incorrect solutions in order to construct the preference rankings. The second dataset that is set up is the base model (Open Math codellama-7b) generation dataset provided by [Nemo-Skills](https://github.com/Kipok/NeMo-Skills/tree/main) repository. Both of these datasets and then used to train and evaluate the reward model, respectively. For natural language experiments, no additional setup is required other than setting up HuggingFace when running the scripts.

The citations for the sources of these datasets are the following:

OpenMathInstruct

```
@article{hu2024openrlhf,
  title={OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework},
  author={Jian Hu and Xibin Wu and Weixun Wang and Xianyu and Dehao Zhang and Yu Cao},
  journal={arXiv preprint arXiv:2405.11143},
  year={2024}
}
```

Nemo-Skills

```
@article{toshniwal2024openmath,
  title   = {OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset},
  author  = {Shubham Toshniwal and Ivan Moshkov and Sean Narenthiran and Daria Gitman and Fei Jia and Igor Gitman},
  year    = {2024},
  journal = {arXiv preprint arXiv: Arxiv-2402.10176}
}
```

Helpful-Harmlessness

```
@article{Bai2022TrainingAH,
  title={Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback},
  author={Yuntao Bai and Andy Jones and Kamal Ndousse and Amanda Askell and Anna Chen and Nova Dassarma and Dawn Drain and Stanislav Fort and Deep Ganguli and Tom Henighan and Nicholas Joseph and Saurav Kadavath and John Kernion and Tom Conerly and Sheer El-Showk and Nelson Elhage and Zac Hatfield-Dodds and Danny Hernandez and Tristan Hume and Scott Johnston and Shauna Kravec and Liane Lovitt and Neel Nanda and Catherine Olsson and Dario Amodei and Tom B. Brown and Jack Clark and Sam McCandlish and Christopher Olah and Benjamin Mann and Jared Kaplan},
  journal={ArXiv},
  year={2022},
  volume={abs/2204.05862},
  url={https://api.semanticscholar.org/CorpusID:248118878}
}
```

HelpSteer

```
@article{Wang2023HelpSteerMH,
  title={HelpSteer: Multi-attribute Helpfulness Dataset for SteerLM},
  author={Zhilin Wang and Yi Dong and Jiaqi Zeng and Virginia Adams and Makesh Narsimhan Sreedhar and Daniel Egert and Olivier Delalleau and Jane Polak Scowcroft and Neel Kant and Aidan Swope and Oleksii Kuchaiev},
  journal={ArXiv},
  year={2023},
  volume={abs/2311.09528},
  url={https://api.semanticscholar.org/CorpusID:265220723}
}
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

For the natural language reward model training (on helpful-harmless), you can run the script `examples/run_scripts/train_naturallanguage_rm.sh`

```
# llama7b contrastive RM training
set -x 

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
     --learning_rate 1e-5 \
     --dataset Anthropic/hh-rlhf \
     --dataset_probs 1.0 \
     --contrastive_loss \
     --contrastive_loss_beta 0.5 \
     --unsim_samples 16 \
     --flash_attn \
     --gradient_checkpointing \
     --use_wandb <your wandb key> \
     --s3_save_path <your s3 save path> \
     --seed 0
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

### Guided decoding

For the guided decoding experiments, please see the README in `decoding/`.
