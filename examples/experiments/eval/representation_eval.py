import glob
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from openrlhf.models import get_llm_for_sequence_regression

NEMO_TEMPLATE = """\
  System:\n\
  You're an expert Python programmer and mathematician. \
  Help the user to solve this problem using code when necessary. \
  Make sure to put the answer (and only answer) inside \\boxed{{}}.\n\nUser:\n{}\n\nAssistant:\n"""
BS = 32

# Load results
jsonl_files = glob.glob("./math/output-rs*.jsonl")
print("There are", len(jsonl_files), "jsonl files")
results = []
for _, file in enumerate(jsonl_files):
    random_seed = file.split("-")[-1].split(".")[0]
    with open(file, "r") as f:
        for line in f:
            line = json.loads(line)
            line["random_seed"] = random_seed
            results.append(line)

rm_s3_path = "s3://scale-ml/content-understanding-ml/contrastive_codellama7b_v3_highest_perf_3_13"
if not os.path.exists("./reward_model"):
    os.makedirs("./reward_model")
    os.system(f"aws s3 cp {rm_s3_path} ./reward_model --recursive")

# Positive goal state representations
goal_state_path = "s3://scale-ml/content-understanding-ml/pos_reps_goal.npy"
goal_state_local_path = "./pos_reps_goal.npy"
if not os.path.exists(goal_state_local_path):
    os.system(f"aws s3 cp {goal_state_path} {goal_state_local_path}")

# Model
reward_model = get_llm_for_sequence_regression(
    "./reward_model", "reward", normalize_reward=True, use_flash_attention_2=True, bf16=True
).cuda()
reward_model = reward_model.eval()
tokenizer = AutoTokenizer.from_pretrained("./reward_model")

# Load the goal state
states = np.load(goal_state_local_path)
goal_state = torch.tensor(np.mean(states, axis=0))
goal_state = goal_state.cuda()

sequences = []
num_actions = []
for result in results:
    question = result["question"]
    generated_solution = result["generated_solution"]
    formatted = NEMO_TEMPLATE.format(question) + generated_solution
    sequences.append(formatted)
    num_actions.append(len(tokenizer(generated_solution, add_special_tokens=False)["input_ids"]))

for i in tqdm(range(0, len(sequences), BS)):
    sequences_batch = sequences[i : i + BS]
    sequences_batch = tokenizer(sequences_batch, return_tensors="pt", padding=True, truncation=True)
    sequences_batch = {k: v.cuda() for k, v in sequences_batch.items()}
    with torch.no_grad():
        rewards, _, _, hidden_states = reward_model(**sequences_batch, return_output=True)
    cosine_sims = torch.nn.functional.cosine_similarity(
        hidden_states, goal_state[None, None, :], dim=2
    )

    for j in range(i, i + BS):
        if j >= len(sequences):
            break
        results[j]["reward"] = rewards[j - i].item()
        results[j]["cosine_sim"] = (
            cosine_sims[j - i][(-1 * num_actions[j]) :].detach().cpu().numpy().tolist()
        )
        results[j]["num_actions"] = num_actions[j - i]

    if i % 2_000 == 0:
        # save jsonl
        with open("output-representation-eval-math.jsonl", "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")


with open("output-representation-eval-math.jsonl", "w") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")
