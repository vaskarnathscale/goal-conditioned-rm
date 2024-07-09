
FILENAME = "gpt4_eval_results_05_17.json"
MODEL_INDEX = 3

import json
from collections import defaultdict

scores_store = defaultdict(int)
totals_store = defaultdict(int)
for line in open(FILENAME):
    data = json.loads(line)
    if data["our_model_i"] == MODEL_INDEX:
        for score_type in data["scores"]:
            if data["scores"][score_type] in ["0", "1"]:
                scores_store[score_type] += int(data["scores"][score_type])
                totals_store[score_type] += 1

print("Scores for model", MODEL_INDEX)
for score_type in scores_store:
    print(score_type, scores_store[score_type] / totals_store[score_type])