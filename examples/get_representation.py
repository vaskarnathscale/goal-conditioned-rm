from datasets import load_from_disk
from openrlhf.models import get_llm_for_sequence_regression
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
import os
import numpy as np


NEMO_TEMPLATE = """\
System:\n\
You're an expert Python programmer and mathematician. \
Help the user to solve this problem using code when necessary. \
Make sure to put the answer (and only answer) inside \\boxed{{}}.\n\nUser:\n{}\n\nAssistant:\n"""

def main(dataset_path, reward_model_path, save_dir):

    ds = load_from_disk(dataset_path)
    reward_model = get_llm_for_sequence_regression(
        reward_model_path, "reward", normalize_reward=False, use_flash_attention_2=True, bf16=True
    ).cuda()

    reward_model_name = reward_model_path.split("/")[-1]
    if len(reward_model_name) == 0:
        reward_model_name = reward_model_path.split("/")[-2]

    reward_model = reward_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path)
    tokenizer.padding_side = "left"

    train_data = ds["train"]
    sequences = []

    for entry in train_data:
        question = entry["question"]
        generated_solution = entry["correct"]
        formatted = NEMO_TEMPLATE.format(question) + generated_solution
        sequences.append(formatted)

    states_aggregated = None
    for i in tqdm(range(0, len(sequences), BS)):
        torch.cuda.empty_cache()
        sequences_batch = sequences[i : i + BS]
        sequences_batch = tokenizer(sequences_batch, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        sequences_batch = {k: v.cuda() for k, v in sequences_batch.items()}

        with torch.no_grad():
            _, _, _, hidden_states = reward_model(**sequences_batch, return_output=True)
            last_hidden_state = hidden_states[:, -1, :].float().cpu()
            if states_aggregated is None:
                states_aggregated = last_hidden_state.numpy().sum(axis=0) / len(sequences)
            else:
                states_aggregated = states_aggregated + (last_hidden_state.numpy().sum(axis=0) / len(sequences))

        if i % 500 == 0:
            np.save(os.path.join(save_dir, f"{reward_model_name}_goal_state_embedding.npy"), states_aggregated)

    np.save(os.path.join(save_dir, f"{reward_model_name}_goal_state_embedding.npy"), states_aggregated)

if __name__ == "__main__":
    BS=32
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--reward-model-path", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    args = parser.parse_args()
    main(args.dataset_path, args.reward_model_path, args.save_dir)
