import glob
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import roc_auc_score
import glob
import json
import numpy as np
from tqdm import tqdm
import numpy as np
from scipy import stats

from openrlhf.models import get_llm_for_sequence_regression
import argparse
from sklearn.metrics import roc_auc_score

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

def print_roc_auc(model_name, benchmarks):
    print(f"Model: {model_name}")
    for benchmark in benchmarks:
        data_path = f"/mnt/efs/vaskarnath/workspace/cache/dense-rewards/math-data/{model_name}-results-new/{benchmark}/*full-False.jsonl"
        data = get_data(data_path)
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

def llm_code_replace(text: str) -> str:
    if "<llm-code-output>" not in text or "</llm-code-output>" not in text:
        return text
    last_code_start = text.rfind("<llm-code-output>")
    modified_text = text[:last_code_start] # mask out everything after + "<llm-code-output>\n[MASK]\n</llm-code-output>" + text[last_code_end:]
    return modified_text

def prepare(args, benchmark):

    # Positive goal state representations
    if not os.path.exists(args.goal_state_local_path):
        os.system(f"aws s3 cp {args.goal_state_path} {args.goal_state_local_path}")

    # Load results
    if args.full_eval:
        data_path = f"/mnt/efs/vaskarnath/workspace/cache/dense-rewards/math-data/math_eval_generations/{benchmark}/*.jsonl"
    else:
        data_path = f"/mnt/efs/vaskarnath/workspace/cache/dense-rewards/math-data/math_eval_generations/{benchmark}/*-greedy.jsonl"
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

    if not os.path.exists(args.rm_local_path):
        os.makedirs(args.rm_local_path)
        os.system(f"aws s3 cp {args.rm_s3_path} {args.rm_local_path} --recursive")

    if 'vhs-mlp' in args.rm_local_path:
        value_head_strategy = "mlp"
    else:
        value_head_strategy = "linear"

    print(f"Value head strategy: {value_head_strategy}")

    reward_model = get_llm_for_sequence_regression(
        args.rm_local_path, "reward", normalize_reward=False, use_flash_attention_2=False, bf16=True, value_head_strategy=value_head_strategy
    ).cuda()
    reward_model = reward_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.rm_local_path)

    # Load the goal state
    states = np.load(args.goal_state_local_path)
    if len(states.shape) == 1:
        goal_state = torch.tensor(states)
    else:
        goal_state = torch.tensor(np.mean(states, axis=0))
    goal_state = goal_state.cuda()

    rm_model_name = args.rm_local_path.split("/")[-1]
    os.makedirs(f"/mnt/efs/vaskarnath/workspace/cache/dense-rewards/math-data/{rm_model_name}-results-new/{benchmark}", exist_ok=True)
    output_path = f"/mnt/efs/vaskarnath/workspace/cache/dense-rewards/math-data/{rm_model_name}-results-new/{benchmark}/output-metrics-dense-mask-{MASK}-full-{args.full_eval}.jsonl"

    return reward_model, tokenizer, goal_state, results, output_path

# Model
def eval(reward_model, tokenizer, goal_state, results, is_vanilla):
    
    sequences = []
    num_actions = []
    for result in results:
        question = result["question"]
        generated_solution = result["generated_solution"]
        formatted = NEMO_TEMPLATE.format(question) + generated_solution
        if MASK:
            formatted = llm_code_replace(formatted)
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
                cosine_sims = torch.nn.functional.cosine_similarity(
            hidden_states, goal_state[None, None, :], dim=2
        )
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
                with open(output_path, "w") as f:
                    for result in results:
                        f.write(json.dumps(result) + "\n")


    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


def print_eval_results(benchmarks, model_name):
    print_roc_auc(model_name, benchmarks)
    if 'GSM8K' in benchmarks:
        _print_filter_results(model_name, 'gsm8k', 1319)
    if 'MATH' in benchmarks:
        _print_filter_results(model_name, 'math', 5000)

def _print_filter_results(model_name, benchmark, dataset_len):
    dataset_path = f"/mnt/efs/vaskarnath/workspace/cache/dense-rewards/math-data/{model_name}-results/{benchmark}/*full-True.jsonl"
    accuracy, filtered_per_question, tokens_saved_per_question, tokens_per_question, proportion_correct_filtered, proportion_correct = get_accuracy(dataset_path, dataset_len, 0, tokenizer)
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
    parser.add_argument("--goal_state_path", type=str, default="s3://scale-ml/content-understanding-ml/pos_reps_goal.npy")
    parser.add_argument("--goal_state_local_path", type=str, default="/mnt/efs/vaskarnath/workspace/cache/dense-rewards/representations/goal_state_embedding.npy")
    parser.add_argument("--rm_s3_path", type=str, default="s3://scale-ml/content-understanding-ml/contrastive-llama-7b-rm-average-goal-state")
    parser.add_argument("--rm_local_path", type=str)
    parser.add_argument(
        "--benchmarks", nargs="+", type=str, help="Benchmarks to evaluate.", default=["GSM8K", "MATH", "GSM-Hard", "mawps", "Asdiv", "algebra222", "svamp"]
    )

    parser.add_argument(
        "--full_eval", action='store_true', default=False
    )

    parser.add_argument(
        "--mask", action='store_true', default=False
    )

    parser.add_argument(
        "--vanilla", action='store_true', default=False
    )
    
    args = parser.parse_args()
    MASK = args.mask

    for benchmark in args.benchmarks:
        reward_model, tokenizer, goal_state, results, output_path = prepare(args, benchmark.lower())
        eval(reward_model, tokenizer, goal_state, results, args.vanilla)
    
    print_eval_results(args.benchmarks, args.rm_local_path.split("/")[-1])
