import numpy as np
from sklearn.metrics import roc_auc_score
import glob
import json
from tqdm import tqdm
from scipy import stats
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

def get_partial_roc_auc(data, percentile):
    scores = []
    labels = []
    for d in data:
        if 'reward' not in d:
            continue
        partial_rewards = d['partial_rewards']
        cosine_sim = d['cosine_sim']
        partial_rewards = partial_rewards[-len(cosine_sim):]
        scores.append(partial_rewards[max(int(len(partial_rewards)*percentile) - 1, 0)])
        labels.append(d['is_correct'])
    return roc_auc_score(labels, scores)

def print_roc_auc(model_name, benchmarks, data_path):
    print(f"Model: {model_name}")
    for benchmark in benchmarks:
        path = f"{data_path}/{model_name}/{benchmark}/greedy/*.jsonl"
        data = get_data(path)
        roc_auc = get_roc_auc(data)
        print(f"{benchmark}: {roc_auc}")

def get_accuracy(data_path, dataset_len, lower_bound_threshold):
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
            # remove first few tokens as they provide some noisy signal
            cosine_sim = cosine_sim[5:]
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

def print_eval_results(benchmarks, model_name, data_path, filter_threshold, is_vanilla, percentiles):
    print_roc_auc(model_name, benchmarks, data_path)

    for benchmark in benchmarks:
        for percentile in percentiles:
            path = f"{data_path}/{model_name}/{benchmark}/greedy/*.jsonl"
            data = get_data(path)
            roc_auc = get_partial_roc_auc(data, percentile)
            print(f"{benchmark} @ percentile {percentile}: {roc_auc}")

    if not is_vanilla:
        if 'gsm8k' in benchmarks:
            _print_filter_results(model_name, 'gsm8k', 1319, data_path, filter_threshold)
        if 'math' in benchmarks:
            _print_filter_results(model_name, 'math', 5000, data_path, filter_threshold)

def _print_filter_results(model_name, benchmark, dataset_len, data_path, filter_threshold=0.0, generation_percentile=0.0):
    dataset_path = f"{data_path}/{model_name}/{benchmark}/all/*.jsonl"
    accuracy, filtered_per_question, tokens_saved_per_question, tokens_per_question, proportion_correct_filtered, proportion_correct = get_accuracy(dataset_path, dataset_len, filter_threshold)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--eval_results_dir", type=str)
    parser.add_argument("--filter_threshold", type=float, default=0.0)
    parser.add_argument(
        "--benchmarks", nargs="+", type=str, help="Benchmarks to evaluate.", default=["gsm8k", "math", "gsm-hard", "mawps", "asdiv", "algebra222", "svamp"]
    )
    parser.add_argument(
        "--percentiles", nargs="+", type=float, help="Percentiles to evaluate.", default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    parser.add_argument(
        "--vanilla", action='store_true', default=False, help="Whether or not using vanilla reward model whice cannot be used to compute Q-value."
    )
    
    args = parser.parse_args()    
    print_eval_results(args.benchmarks, args.model_name, args.eval_results_dir, args.filter_threshold, args.vanilla, args.percentiles)
