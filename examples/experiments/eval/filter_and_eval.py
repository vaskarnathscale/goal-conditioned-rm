import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer


def get_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


def get_generated_data(data_dir):
    print("getting generated data")
    files = os.listdir(data_dir)
    jsonl_files = sorted(
        [file for file in files if file.endswith(".jsonl") and file.startswith("output-rs")]
    )
    generated_data = []
    for file in tqdm(jsonl_files):
        file_path = os.path.join(data_dir, file)
        with open(file_path, "r") as f:
            lines = f.readlines()
            generated_data.append([json.loads(line) for line in lines])
    return generated_data


def load_embeddings(embedding_data_dir):
    print("loading embeddings")
    files = os.listdir(embedding_data_dir)
    embedding_files = sorted([file for file in files if file.endswith(".pt")])
    ret = []
    for file in tqdm(embedding_files):
        file_path = os.path.join(embedding_data_dir, file)
        embedding = torch.load(file_path)
        ret.append(embedding)
    return ret


def filter(embeddings, goal_state_embedding, filter_threshold, tokenizer, embeddings_dir):
    print("filtering")
    # calculate cosine similarity
    num_generations = len(generated_data)
    eval_size = len(generated_data[0])
    filter_mask = torch.zeros((num_generations, eval_size))
    tokens_saved_list = []
    total_tokens_list = []

    for eval_idx in tqdm(range(eval_size)):
        tokens_saved = 0
        total_tokens = 0
        for gen_idx in range(num_generations):
            gen_sol_input_ids = tokenizer.encode(
                generated_data[gen_idx][eval_idx]["generated_solution"], return_tensors="pt"
            )
            if os.path.exists(os.path.join(embeddings_dir, f"similarity_{eval_idx}_{gen_idx}.pt")):
                similarity = torch.load(
                    os.path.join(embeddings_dir, f"similarity_{eval_idx}_{gen_idx}.pt")
                )
            else:
                embedding = embeddings[gen_idx][eval_idx].squeeze()
                gen_sol_embedding = embedding[-gen_sol_input_ids.shape[1] :]
                similarity = torch.nn.functional.cosine_similarity(
                    gen_sol_embedding, goal_state_embedding.repeat(gen_sol_embedding.shape[0], 1), 1
                )
                torch.save(
                    similarity, os.path.join(embeddings_dir, f"similarity_{eval_idx}_{gen_idx}.pt")
                )

            similarity_above_threshold_flag = similarity > filter_threshold
            # find the left most index where the similarity is below the threshold
            indices = torch.sort(torch.where(similarity_above_threshold_flag == 0)[0])[0]
            if len(indices) > 0:
                tokens_saved += gen_sol_input_ids.size(1) - indices[0].item()
            total_tokens += gen_sol_input_ids.size(1)
            filter_mask[gen_idx][eval_idx] = torch.all(
                similarity_above_threshold_flag, dim=0
            ).item()
        tokens_saved_list.append(tokens_saved)
        total_tokens_list.append(total_tokens)

    return filter_mask, tokens_saved_list, total_tokens_list


def run_eval(
    filter_mask,
    generated_data,
    filter_threshold,
    metric_out_path,
    tokens_saved_list,
    total_tokens_list,
):
    print("running eval")

    num_generations = len(generated_data)
    eval_size = len(generated_data[0])
    correct = 0
    wrong = 0
    total = 0
    final_tokens_saved_list = []

    for eval_idx in tqdm(range(eval_size)):
        is_correct_filtered = 0
        is_correct_total = 0
        not_correct_answers_filtered = []
        not_correct_answers_total = []
        for gen_idx in range(num_generations):
            entry = generated_data[gen_idx][eval_idx]
            is_correct = entry["is_correct"]
            predicted_answer = entry["predicted_answer"]
            if filter_mask[gen_idx][eval_idx] == 1:
                if is_correct:
                    is_correct_filtered += 1
                else:
                    not_correct_answers_filtered.append(predicted_answer)

            if is_correct:
                is_correct_total += 1
            else:
                not_correct_answers_total.append(predicted_answer)

        # majority vote
        if len(not_correct_answers_filtered) > 0 or is_correct_filtered > 0:
            majority_vote_correct = majority_vote(is_correct_filtered, not_correct_answers_filtered)
            final_tokens_saved_list.append(tokens_saved_list[eval_idx])
        else:
            # set filter mask to 1 for this eval_idx row
            filter_mask[:, eval_idx] = 1
            majority_vote_correct = majority_vote(is_correct_total, not_correct_answers_total)
            final_tokens_saved_list.append(0)

        correct += majority_vote_correct
        wrong += 1 - majority_vote_correct

        total += 1

    metric = {
        "num_entries": total,
        "correct_answer": correct / total,
        "wrong_answer": wrong / total,
        "no_answer": (total - (correct + wrong)) / total,
        "filter_threshold": filter_threshold,
        "examples_filtered": (filter_mask.shape[0] - filter_mask.sum(dim=0)).tolist(),
        "tokens_saved": final_tokens_saved_list,
        "total_tokens": total_tokens_list,
    }

    print(metric)
    json.dump(metric, open(metric_out_path, "w"))


def majority_vote(is_correct_count, not_correct_list):
    # get the most common element in not_correct_answers
    if is_correct_count > len(not_correct_list):
        return 1
    most_common = max(set(not_correct_list), key=not_correct_list.count)
    return is_correct_count >= not_correct_list.count(most_common)


if __name__ == "__main__":

    # Example usage
    # python /home/ubuntu/workspace/models/openrlhf/examples/experiments/vaskar/reward_model_filter/filter_and_eval.py
    # --model_path /home/ubuntu/workspace/cache/dense_rewards/contrastive_codellama7b_v3_highest_perf_3_13
    # --embedding_data_dir /home/ubuntu/workspace/cache/dense_rewards/data/gsm8k-emb
    # --generated_data_dir /home/ubuntu/workspace/cache/dense_rewards/data/gsm8k
    # --sim_data_dir /home/ubuntu/workspace/cache/dense_rewards/data/gsm8k-sim
    # --goal_state_embedding /home/ubuntu/workspace/cache/dense_rewards/goal_state_embedding.npy
    # --filter_threshold 0
    # --metric_out_path /home/ubuntu/workspace/cache/dense_rewards/data/metrics.json

    parser = argparse.ArgumentParser(description="Filter and eval script for reward model filter")
    parser.add_argument("--model_path", type=str, help="The model string to use for generation")
    parser.add_argument("--embedding_data_dir", type=str, help="The file to read the embeddings")
    parser.add_argument("--generated_data_dir", type=str, help="The file to read the generations")
    parser.add_argument(
        "--sim_data_dir", type=str, help="The file to read the similarity score if available"
    )
    parser.add_argument("--goal_state_embedding", type=str, help="The goal state embedding")
    parser.add_argument("--filter_threshold", type=float, help="The filter threshold")
    parser.add_argument("--metric_out_path", type=str, help="The file to write the metric to")
    parser.add_argument(
        "--start_with_saved_similarity",
        help="Whether to start with saved similarity or not",
        action="store_true",
    )

    args = parser.parse_args()

    tokenizer = get_tokenizer(args.model_path)
    generated_data = get_generated_data(args.generated_data_dir)

    if not args.start_with_saved_similarity:
        embeddings = load_embeddings(args.embedding_data_dir)
    else:
        embeddings = None
        print("Starting with saved similarity. No need to load embeddings.")

    goal_state_embedding = np.load(args.goal_state_embedding)
    goal_state_embedding = torch.tensor(goal_state_embedding)
    goal_state_embedding_mean = torch.mean(goal_state_embedding, dim=0)
    filter_mask, tokens_saved_list, total_tokens_list = filter(
        embeddings, goal_state_embedding_mean, args.filter_threshold, tokenizer, args.sim_data_dir
    )
    run_eval(
        filter_mask,
        generated_data,
        args.filter_threshold,
        args.metric_out_path,
        tokens_saved_list,
        total_tokens_list,
    )
