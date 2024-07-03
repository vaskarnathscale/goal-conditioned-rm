import os
import numpy as np
import torch
from transformers import AutoTokenizer
from sklearn.metrics import roc_auc_score
import glob
import json
from tqdm import tqdm
from scipy import stats
from openrlhf.models import get_llm_for_sequence_regression
import argparse

def get_data(data_path):    
    jsonfiles = glob.glob(data_path)
    data = []
    for file in jsonfiles:
        with open(file, "r") as f:
            for line in f:
                line = json.loads(line)
                data.append(line)
    return data

def get_roc_auc(data):
    scores = []
    labels = []
    for d in data:
        if 'reward' not in d:
            continue
        scores.append(d['reward'])
        labels.append(d['is_correct'])
    return roc_auc_score(labels, scores)

def print_roc_auc(model_name, benchmarks, data_path):
    print(f"Model: {model_name}")
    for benchmark in benchmarks:
        path = f"{data_path}/{model_name}/{benchmark}/greedy/*.jsonl"
        data = get_data(path)
        roc_auc = get_roc_auc(data)
        print(f"{benchmark}: {roc_auc}")

def get_accuracy(data_path, dataset_len, lower_bound_threshold, generation_percentile=0):
    data = get_data(data_path)
    num_samples = len(data) // dataset_len
    filtered_per_question = []

    majority_vote_correct = 0
    tokens_saved_per_question = [0 for _ in range(dataset_len)]
    tokens_per_question = [0 for _ in range(dataset_len)]
    proportion_correct_filtered = []
    proportion_correct = []

    for q in tqdm(range(dataset_len)):
        num_filtered = 0
        num_filtered_correct = 0
        num_correct = 0
        filtered_answers_count = {}
        answers_count = {}

        for i in range(q, len(data), dataset_len):
            if i > 0:
                assert data[i]['question'] != data[i - 1]['question']
            entry = data[i]

            cosine_sim = np.array(entry['cosine_sim'])
            cosine_sim = cosine_sim[max(5,int(len(cosine_sim) * generation_percentile)):]
            tokens_per_question[q] += len(cosine_sim)
 
            if np.min(cosine_sim) >= lower_bound_threshold:
                if entry['is_correct']:
                    num_filtered_correct += 1
                else:
                    if entry['predicted_answer'] not in filtered_answers_count:
                        filtered_answers_count[entry['predicted_answer']] = 0
                    filtered_answers_count[entry['predicted_answer']] += 1
            else:
                # find earliest index where cosine sim is below threshold
                idx = np.where(cosine_sim < lower_bound_threshold)[0][0]
                tokens_saved_per_question[q] += len(cosine_sim) - idx
                num_filtered += 1
            
            if entry['is_correct']:
                num_correct += 1
            else:
                if entry['predicted_answer'] not in answers_count:
                    answers_count[entry['predicted_answer']] = 0
                answers_count[entry['predicted_answer']] += 1
        filtered_per_question.append(num_filtered)

        proportion_correct.append(num_correct / (num_correct + sum(answers_count.values())))

        if num_filtered == num_samples:
            tokens_saved_per_question[q] = 0
            filtered_per_question[q] = 0
            if len(answers_count) == 0:
                proportion_correct_filtered.append(1)
                majority_vote_correct += 1
            else:
                proportion_correct_filtered.append(num_correct / (sum(answers_count.values()) + num_correct))
                max_answer_count = max(answers_count.values())
                majority_vote_correct += num_correct >= max_answer_count
        else:
            if len(filtered_answers_count) == 0:
                proportion_correct_filtered.append(1)
                majority_vote_correct += 1
            else:
                proportion_correct_filtered.append(num_filtered_correct / (sum(filtered_answers_count.values()) + num_filtered_correct))
                max_filtered_answer_count = max(filtered_answers_count.values())
                majority_vote_correct += num_filtered_correct >= max_filtered_answer_count
        
    return majority_vote_correct / dataset_len, filtered_per_question, tokens_saved_per_question, tokens_per_question, proportion_correct_filtered, proportion_correct

def prepare_model(args):
    if 'vhs-mlp' in args.reward_model_path:
        value_head_strategy = "mlp"
    else:
        value_head_strategy = "linear"

    reward_model = get_llm_for_sequence_regression(
        args.reward_model_path, "reward", normalize_reward=False, use_flash_attention_2=False, bf16=True, value_head_strategy=value_head_strategy
    ).cuda()
    reward_model = reward_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path)

    return reward_model, tokenizer

def prepare_data(args, benchmark, full_eval=False):

    # Load results
    if full_eval:
        data_path = f"{args.completions_data_path}/{benchmark}/*.jsonl"
    else:
        data_path = f"{args.completions_data_path}/{benchmark}/*-greedy.jsonl"

    jsonl_files = glob.glob(data_path)
    print(f'Processing {len(jsonl_files)} files')
    random_seed=42

    results = []

    for file in jsonl_files:
        with open(file, "r") as f:
            for line in f:
                line = json.loads(line)
                line["random_seed"] = random_seed
                results.append(line)

    if not os.path.exists(args.reward_model_path):
        os.makedirs(args.reward_model_path)
        os.system(f"aws s3 cp {args.rm_s3_path} {args.reward_model_path} --recursive")

    # Load the goal state
    states = np.load(args.goal_state_path)
    if len(states.shape) == 1:
        goal_state = torch.tensor(states)
    else:
        goal_state = torch.tensor(np.mean(states, axis=0))
    goal_state = goal_state.cuda()

    rm_model_name = args.reward_model_path.split("/")[-1]

    if full_eval:
        os.makedirs(f"{args.save_dir}/{rm_model_name}/{benchmark}/all", exist_ok=True)
        output_path = f"{args.save_dir}/{rm_model_name}/{benchmark}/all/output-metrics-dense.jsonl"
    else:
        os.makedirs(f"{args.save_dir}/{rm_model_name}/{benchmark}/greedy", exist_ok=True)
        output_path = f"{args.save_dir}/{rm_model_name}/{benchmark}/greedy/output-metrics-dense.jsonl"

    return goal_state, results, output_path

def eval(reward_model, tokenizer, goal_state, results, save_path, is_vanilla):
    
    sequences = []
    num_actions = []
    for result in results:
        question = result["question"]
        generated_solution = result["generated_solution"]
        formatted = NEMO_TEMPLATE.format(question) + generated_solution
        sequences.append(formatted)
        num_actions.append(len(tokenizer(generated_solution, add_special_tokens=False)["input_ids"]))

    for i in tqdm(range(0, len(sequences), BS)):
        torch.cuda.empty_cache()
        sequences_batch = sequences[i : i + BS]
        sequences_batch = tokenizer(sequences_batch, return_tensors="pt", padding=True, truncation=True)
        attention_mask = sequences_batch["attention_mask"]
        eos_indices = (
                        attention_mask.size(1)
                        - 1
                        - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
                    )
        sequences_batch = {k: v.cuda() for k, v in sequences_batch.items()}
        with torch.no_grad():
            rewards, _, partial_rewards, hidden_states = reward_model(**sequences_batch, return_output=True)
            if not is_vanilla:
                cosine_sims = torch.nn.functional.cosine_similarity(hidden_states, goal_state[None, None, :], dim=2)
            for j in range(i, i + BS):
                if j >= len(sequences):
                    break
                results[j]["reward"] = rewards[j - i].item()
                
                if not is_vanilla:
                    results[j]["partial_rewards"] = partial_rewards[j-i, :eos_indices[j-i] + 1].float().detach().cpu().numpy().tolist()
                    results[j]["cosine_sim"] = (
                        cosine_sims[j - i][(-1 * num_actions[j]) :].detach().cpu().numpy().tolist()
                    )
                results[j]["num_actions"] = num_actions[j - i]
            
            if i % 3_000 == 0:
                # save jsonl
                with open(save_path, "w") as f:
                    for result in results:
                        f.write(json.dumps(result) + "\n")

    with open(save_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


def print_eval_results(benchmarks, model_name, data_path, filter_threshold=0.0):
    print_roc_auc(model_name, benchmarks, data_path)

    # filtering experiments are only done on the 50 sample completions of gsm8k and math
    if full_eval:
        if 'gsm8k' in benchmarks:
            _print_filter_results(model_name, 'gsm8k', 1319, data_path, filter_threshold)
        if 'math' in benchmarks:
            _print_filter_results(model_name, 'math', 5000, data_path, filter_threshold)

def _print_filter_results(model_name, benchmark, dataset_len, data_path, filter_threshold=0.0, generation_percentile=0.0):
    dataset_path = f"{data_path}/{model_name}/{benchmark}/all/*.jsonl"
    accuracy, filtered_per_question, tokens_saved_per_question, tokens_per_question, proportion_correct_filtered, proportion_correct = get_accuracy(dataset_path, dataset_len, filter_threshold, generation_percentile)
    print(f"{benchmark} accuracy: {accuracy}")
    print(f"Average number of generations filtered: {np.mean(filtered_per_question)}")
    print(f"Total number of generations filtered: {np.sum(filtered_per_question)}")
    print(f"Average number of tokens saved: {np.mean(tokens_saved_per_question)}")
    print(f"Total number of tokens saved: {np.sum(tokens_saved_per_question)}")
    print(f"Average number of tokens: {np.mean(tokens_per_question)}")
    print(f"Total number of tokens: {np.sum(tokens_per_question)}")
    print(f"Average proportion correct filtered: {np.mean(proportion_correct_filtered)}")
    print(f"Average proportion correct: {np.mean(proportion_correct)}")
    print(f"Ttest p value: {stats.ttest_ind(proportion_correct_filtered, proportion_correct)}" )


if __name__ == "__main__":
    NEMO_TEMPLATE = """\
  System:\n\
  You're an expert Python programmer and mathematician. \
  Help the user to solve this problem using code when necessary. \
  Make sure to put the answer (and only answer) inside \\boxed{{}}.\n\nUser:\n{}\n\nAssistant:\n"""
    BS = 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--completions_data_path", type=str)
    parser.add_argument("--goal_state_path", type=str)
    parser.add_argument("--reward_model_path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--filter_threshold", type=float, default=0.0)
    parser.add_argument(
        "--benchmarks", nargs="+", type=str, help="Benchmarks to evaluate.", default=["gsm8k", "math", "gsm-hard", "mawps", "asdiv", "algebra222", "svamp"]
    )
    parser.add_argument(
        "--vanilla", action='store_true', default=False, help="Whether or not using vanilla reward model whice cannot be used to compute Q-value."
    )
    
    args = parser.parse_args()

    reward_model, tokenizer = prepare_model(args)

    for benchmark in args.benchmarks:

        goal_state, results, output_path = prepare_data(args, benchmark)

        if benchmark == 'gsm8k' or benchmark == 'math':
            goal_state, results, output_path = prepare_data(args, benchmark.lower())
            eval(reward_model, tokenizer, goal_state, results, output_path, args.vanilla)
        eval(reward_model, tokenizer, goal_state, results, output_path, args.vanilla)
    
    print_eval_results(args.benchmarks, args.reward_model_path.split("/")[-1], args.save_dir, args.filter_threshold)
