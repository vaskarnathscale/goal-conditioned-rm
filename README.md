
# ScaleAI <> OpenRLHF Fork

Welcome to Scale AI's fork of OpenRLHF! OpenRLHF is a high-performance RLHF framework built on Ray, DeepSpeed and HF Transformers. It offers several key advantages:

- **Simple and easy to use**: OpenRLHF is one of the simplest high-performance RLHF libraries currently available, and compatible with Huggingface models and datasets.
- **High performance**: RLHF training spends 80% of the time on the sample generation stage. Thanks to the ability to use a large inference batch size with Ray and Adam Offload (Pinned Memory), the performance of OpenRLHF with the 13B LLaMA2 model is 4x that of DeepSpeedChat. We also support vLLM generation acceleration to further improve the generation performance.
- **Distributed RLHF**:  OpenRLHF distribute the Actor, Reward, Reference, and Critic models onto separate GPUs using Ray, while placing the Adam optimizer on the CPU. This enables full-scale fine-tuning of 70B+ models with multiple A100 80G GPUs and vLLM (see [architecture](./docs/ray_architecture.png)) and 7B models across multiple 24GB RTX 4090 GPUs.
- **PPO Implementation Tricks**: We integrated the implementation tricks for PPO to improve the training stability, referencing https://arxiv.org/abs/2005.12729 and https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/.


## Scale AI Setup

In this subsection, we describe how to setup containerized training.

### Scale Train Setup

Submitting jobs follows the scale-train workflow. Please follow the Scale Train install [guides](https://train.ml-internal.scale.com/docs/getting_started/).

### NGC Setup

Follow all the instructions in the NGC [setup guide](https://docs.google.com/document/d/16XONI8n4F_DALs7UKMbqFfqAyLvh0I4CkJeODgirubk/edit).

A good litmus test of whether both the scale train setup and ngc setup are working correctly is whether you can run this simple job script.

```shell
cd ~/models; scale-train train --build-manifest-path=research/hugh/scale-train-tests/st_config/build_manifest_ngc.yaml --job-config-path=research/hugh/scale-train-tests/st_config/ngc_test.yaml --user-id=hughzhang --build-config-key=distributed_torch
```

Then navigate to the [jobs page](https://bc.ngc.nvidia.com/jobs) and see that the job has been scheduled successfully.

### OpenRLHF Specific Steps

In addition to these steps, you also _must_:

#### Create a build_values.env file

Navigate to `./st_configs` and create a file `./st_configs/build_values.env`. In this file, place the following variables:

```shell
MODELS_ROOT=...
WANDB=...
```

MODELS_ROOT corresponds to a mount path for a models directory for docker. You can set this to an empty folder if you don't want/have this (e.g., MODELS_ROOT=MODELS_ROOT=/home/dylan/models/ or equivilent). WANDB is your weights and biases API key, for storing training logs.

### Debugging These Steps

In general, unique issues have arisen following these steps. If you are facing issues you are not alone nor unique! Please visit #scale-train in Slack to seek advice and read through previous questions, to see if someone else has faced a similar issue.

### Launching Jobs

In this subsection, we describe how to launch jobs on the NGC cluster.

#### Job Configurations

The training job configurations are stored in `./st_configs/jobs/` folder. For example, a PPO training configuration is given here

```shell
job_type: ngc
job_name: test-ppo-training
image: ${image}
team: gen-ai
product: train
_testing: True

replica_count: 1
ace_instance: dgxa100.80g.8.norm
command:
  - bash
  - example_train_llama_ppo.sh
```

This means we will run `./example_train_llama_ppo.sh` on ngc with job name `test-ppo-training` using 8 A100 GPUS.

#### Training Files

The exact training job and hyperparameters are stored in `./examples/scripts`. For example, an example PPO training recipe with llama is provided in `./examples/scripts/example_train_llama_ppo.sh`. A couple comments on the training configuration files:

- We support models & datasets from huggingface hub and Scale's s3. To use a model or dataset from s3, you'll need to download that model or dataset from s3 locally. To do this, you write an `aws s3 cp ...` command. Examples are in the top of both the PPO and RM training files.
- If you need some custom preprocessing of your dataset, you can generally add this manually in the appropriate dataloader within `./datasets/`.
- You can specify the Deepspeed stage you wish to use via the `--zero_stage` flag (e.g., 1, 2, or 3). 
- For saving the training model to a certain path, you can use `--s3_save_path` flag. This will save the model to the specified s3 path.


#### Launching Via Scale Train

To actually launch the job, first setup the scale train context

```shell
scale-train context
```

Your context should look something like this

```shell
build_config_key: openrlhf_build
build_env: local
build_manifest_path: /home/ubuntu/models/openrlhf/st_configs/build_manifest_ngc.yaml
env_build_values_path: /home/ubuntu/models/openrlhf/st_configs/build_values.env
job_config_path: /home/ubuntu/models/openrlhf/st_configs/jobs/example_train_llama_ppo.yaml
run_env: remote
```

You can make sure the configuration is working locally by running

```shell
scale-train shell
```

Finally, you can launch the job via

```shell
scale-train train
```

## Features

A view of all the features supported within the package.

- Distributed [PPO based on Ray](./examples/scripts/train_ppo_llama_ray.sh). 
- Support full RLHF fine-tuning of models with [over 70 billion parameters](./examples/scripts/train_ppo_llama_ray_70b.sh).
- Support vLLM generation acceleration in RLHF (--vllm_num_engines).
- Support multiple reward models (--reward_pretrain model1,model2...).
- Support [DPO (direct-preference-optimization)/IPO/cDPO](./examples/scripts/train_dpo_llama.sh).
- Support [Kahneman-Tversky optimization (KTO)](./examples/scripts/train_kto_llama.sh).
- Support [Rejection Sampling](./examples/scripts/train_rejection_sampling_llama.sh).
- Support [Conditional SFT](./examples/scripts/train_conditional_llama.sh) (https://arxiv.org/abs/2308.12050).
- Support [Mixtral 8*7b](./examples/test_scripts/train_sft_mixtral_lora.sh) (--aux_loss_coef)
- Support Wandb log (--wandb).
- Support FlashAttention2 (--flash_attn).
- Support QLoRA (--load_in_4bit), LoRA (--lora_rank, --target_modules).

Currently, within the Scale package, we have tested RM and PPO training. Training recipes for the other features are provided in `./examples/scripts/unverified/` and may need additional testing to get working.

**PPO Support Matrix**

| Feature | OpenRLHF | DSChat | CAIChat | TRL | NeMo-Aligner |
| ------------- |:-------------:| :-------------:| :-------------:| :-------------:| :-------------:|
| 70B+ Full Tuning with 16 A100      | ✅ | ❌ | ❌ | ❌ | ✅ (32+ A100s) |
| 7B Full Tuning with 4 RTX4090 | ✅      |    ❌ | ❌ | ❌ | ❌ |
| 34B DPO Full Tuning with 8 A100 | ✅      |    ❌ | ❌ | ❌ | ❌ |  
| PPO Implementation Tricks | ✅      |    ❌ | ❌ | ✅ | ✅ |
| Support QLoRA | ✅      |    ❌ | ❌ | ✅ | ❌ |
| Support Mixtral 8*7b | ✅      |    ❌ | ❌ | ❌ | ❌ |  
| Support Unmerged Actor-Critic | ✅     |   ✅ | ✅ | ❌ | ✅ |
| Support Multiple Reward Models | ✅      |    ❌ | ❌ | ❌ | ❌ |   
| Support Huggingface Models | ✅      |    ✅ | ✅ | ✅ | ❌ (need to convert) |
| Easy-to-use | ✅      |   ✅ | ✅ | ✅ | ❌ |


## Performance
**Common Configuration** 

- Ray: 4 A100 80G for Actor, 2 A100 80G for Critic, 1 A100 80G for RM, and 1 A100 80G for InitPolicy
- DeepSpeed: ZeRO2 with Adam Offload
- Max Sequence Length: 2048 


**Throughput**

| Model | Micro Batch Size (rollout/train) | Throughput | Generation Length |
|-|-|-|-|  
| 7B llama2 | 16/8 | 0.136 samples/gpu/sec | 100-300 |
| 13B llama2 | 8/4 | 0.05 samples/gpu/sec | 200-400 |
| 34B codellama | 2/1 | 0.009 samples/gpu/sec | 300-800 |

samples/gpu/secs = Number of PPO Samples / Number of A100 GPUs / Seconds

**OpenRLHF vs DSChat**

|        | 7B llama2 PPO | 13B llama2 PPO (50k samples) | 
|  ----  | ----  |  ----  |
| OpenRLHF  | - | 17 hours with 8 A100  | 
| DeepSpeedChat  | - | 48 hours with 16 A100  |


## Running Example

Here is how to build and use the package locally.

> [!IMPORTANT]
> You can build openrlhf from **nvidia-docker(recommended)** or from conda envs.

```shell
# Clone the repository: 
git clone https://github.com/openllmai/OpenRLHF.git
```

**install nvidia-docker and OpenRLHF**
  
```bash
cd examples/scripts

# install nvidia-docker (Optional)
./nvidia_docker_install.sh

# build nvidia container with vLLM (Recommended)
./docker_run.sh build

# run nvidia container
./docker_run.sh

# cd in nvidia container
cd /openrlhf/examples/scripts

# build OpenRLHF (i.e, pip install)
./build_openrlhf.sh

# huggingface login 
huggingface-cli login

# wandb login (Optional, also set --wandb True in script)
wandb.login()

```

**Single-node training**

```shell
# Supervised Finetuning
./train_sft_llama.sh

# Reward Model Tuning
./train_rm_llama.sh

# PPO Training
./train_ppo_llama.sh

# DPO
./train_dpo_llama.sh

# KTO
./train_kto_llama.sh

# Rejection Sampling with vLLM
./train_rejection_sampling_llama.sh

# Conditional SFT
./train_conditional_llama.sh

# Continue Pre-training
./train_continue_pretrain_llama.sh
```

**PPO training with Ray**
> [!TIP]
> for >= 13B models on V100/A100/H100.. or 7B models on RTX4090

```bash
# launch the master node of ray in container
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8
# if you want to launch ray on more nodes, use
ray start --address {MASTER-NODE-ADDRESS}:6379  --num-gpus 8

# Ray PPO training, requires 8 GPUs in default config
./train_ppo_llama_ray.sh

# for 70B models
# Launch Ray PPO with vLLM, requires 16 A100s in default config
./train_ppo_llama_ray_70b.sh
```

**Inference and Evaluation**

After completing the training, you can evaluate your model by using the `inference` script:

```bash
# batch generate
# support vLLM acceleration (--eval_task generate_vllm)
python examples/batch_inference.py {args}

# interactive_chat
./interactive_chat_llama.sh { pretrain_model_path }
```

**build openrlhf from conda envs**

If you really don't want to use nvidia-docker, we also provide tutorials for building openrlhf from a conda environment. (We prefer nvidia-docker to avoid errors caused by the environment.)
```shell
# we need conda
conda create -n openrlhf python=3.10
# so, we need install some package manually: when installing torch, you may need to match the corresponding cuda version.
pip install packaging ninja
pip3 install torch
# check ninjia
ninja --version
echo $? # output: 0
# install flash-attn: may take some time.
# For network error: you can download specified version from https://github.com/Dao-AILab/flash-attention/releases.
pip install flash-attn==2.4.2
./build_openrlhf.sh
# enjoy it!
```

## Citation
```
@misc{hu23openrlhf,
   author = {Jian Hu and Xibin Wu and Xianyu and Chen Su and Leon Qiu and Daoning Jiang and Qing Wang and Weixun Wang},
   title = {OpenRLHF: A Ray-based High-performance RLHF framework},
   year={2023},
   publisher = {GitHub},
   journal = {GitHub repository},
   howpublished = {\url{https://github.com/OpenLLMAI/OpenRLHF}}
}
```
