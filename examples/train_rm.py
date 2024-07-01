import argparse
import math
import os
from datetime import datetime

import boto3
from transformers.trainer import get_scheduler

from openrlhf.aws_utils import copy_save_directory_to_s3
from openrlhf.datasets import NEMO_TEMPLATE, RewardDataset
from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.trainer import RewardModelTrainer
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer


def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    if strategy.is_rank_0():
        session = boto3.Session(profile_name="ml-worker")
        s3 = session.client("s3")

    # configure model
    # load huggingface model/config
    model = get_llm_for_sequence_regression(
        args.pretrain,
        "reward",
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        ds_config=strategy.get_ds_train_config(is_actor=False),
        init_value_head=True,
        value_head_strategy=args.value_head_strategy,
    )

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model, "left", strategy)

    strategy.print(model)

    # configure optimizer
    optim = strategy.create_optimizer(
        model, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.l2
    )

    # prepare for data and dataset
    train_data, eval_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=5000000,
        stopping_strategy="all_exhausted",
    )
    train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
    train_dataset = RewardDataset(
        train_data, tokenizer, args.max_len, strategy, input_template=args.input_template, apply_chat_template=args.apply_chat_template
    )
    eval_dataset = RewardDataset(
        eval_data, tokenizer, args.max_len, strategy, input_template=args.input_template, apply_chat_template=args.apply_chat_template
    )

    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        True,
        True,
        train_dataset.collate_fn,
    )
    eval_dataloader = strategy.setup_dataloader(
        eval_dataset, args.micro_train_batch_size, True, False, eval_dataset.collate_fn
    )

    # scheduler
    num_update_steps_per_epoch = (
        len(train_dataloader) * args.max_epochs // strategy.accumulated_gradient
    )
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        "cosine",
        optim,
        num_warmup_steps=math.ceil(max_steps * 0.03),
        num_training_steps=max_steps,
    )

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": args.gradient_checkpointing_use_reentrant
            }
        )

    # strategy prepare
    (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))

    if args.load_checkpoint:
        strategy.print("Load checkpoint: ", args.save_path)

    os.makedirs(args.save_path, exist_ok=True)

    # batch_size here is micro_batch_size * 2
    # we use merged chosen + rejected response forward
    trainer = RewardModelTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        max_epochs=args.max_epochs,
        loss=args.loss,
        contrastive_loss_beta=args.contrastive_loss_beta,
        unsim_samples=args.unsim_samples,
        sim_samples=args.sim_samples,
        reward_model_strategy=args.reward_model_strategy,
        contrastive_strategy=args.contrastive_strategy,
        source_state_percentile=args.source_state_percentile,
        goal_state_percentile=args.goal_state_percentile,
    )

    trainer.fit(args)

    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, tokenizer, args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument(
        "--dataset_probs", type=str, default="1.0", help="sampling probs for datasets"
    )
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_rm")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1000)  # 1000GB
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--micro_train_batch_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=1000000)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--loss", type=str, default="sigmoid")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--compute_fp32_loss", action="store_true", default=False)
    parser.add_argument("--margin_loss", action="store_true", default=False)
    parser.add_argument("--aux_loss_coef", type=float, default=0)
    parser.add_argument("--grad_accum_dtype", type=str, default=None)
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=list, default=None)
    parser.add_argument("--input_template", type=str, default=NEMO_TEMPLATE)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true")

    # contrastive loss params
    parser.add_argument("--contrastive_loss_beta", type=float, default=0.5)
    parser.add_argument("--unsim_samples", type=int, default=16)
    parser.add_argument("--sim_samples", type=int, default=16)
    parser.add_argument("--reward_model_strategy", type=str, required=True)
    parser.add_argument("--contrastive_strategy", type=str, required=True)
    parser.add_argument("--value_head_strategy", type=str, required=True)
    parser.add_argument("--source_state_percentile", type=float, default=None)
    parser.add_argument("--goal_state_percentile", type=float, default=None)

    parser.add_argument("--bos_token", type=str, default=None)
    parser.add_argument("--eos_token", type=str, default=None)
    parser.add_argument("--pad_token", type=str, default=None)
    parser.add_argument("--unk_token", type=str, default=None)

    # custom dataset key name
    parser.add_argument("--prompt_key", type=str, default=None)
    parser.add_argument("--chosen_key", type=str, default=None)
    parser.add_argument("--rejected_key", type=str, default=None)
    parser.add_argument("--apply_chat_template", action="store_true", default=False)

    # wandb pamameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_rm")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="rm_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    args = parser.parse_args()
    train(args)
