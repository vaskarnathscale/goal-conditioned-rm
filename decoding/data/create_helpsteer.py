import datasets
import json

dataset = datasets.load_dataset("nvidia/HelpSteer")

outfile = open("helpsteer.jsonl", "w")

prompts = set()
for item in dataset["validation"]["prompt"]:
    prompts.add(item)

for item in prompts:
    item = {"question": item}
    outfile.write(json.dumps(item) + "\n")
