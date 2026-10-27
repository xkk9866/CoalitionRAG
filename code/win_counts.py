import json
from pathlib import Path

base = {}
with Path("results/action_retrieve_predictions.jsonl").open(encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            row = json.loads(line)
            base[row["id"]] = row
ace = []
with Path("results/policy_retrieve_predictions.jsonl").open(encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            ace.append(json.loads(line))

def count(pred):
    return sum(1 for row in ace if pred(row, base[row["id"]]))

n = len(ace)
print("n", n)
print("ace>dense", count(lambda a, b: a["ace"] > b["dense"]))
print("ace==dense", count(lambda a, b: a["ace"] == b["dense"]))
print("ace<dense", count(lambda a, b: a["ace"] < b["dense"]))
print("ace>ce", count(lambda a, b: a["ace"] > b["ce"]))
print("ace==ce", count(lambda a, b: a["ace"] == b["ce"]))
print("ace<ce", count(lambda a, b: a["ace"] < b["ce"]))
print("ace==1 and dense<1", count(lambda a, b: a["ace"] == 1 and b["dense"] < 1))
print("ace<1 and dense==1", count(lambda a, b: a["ace"] < 1 and b["dense"] == 1))
