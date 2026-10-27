import json
from collections import defaultdict
from pathlib import Path

def by_type(path, keys):
    buckets = {key: defaultdict(list) for key in keys}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            for key in keys:
                buckets[key][row["type"]].append(row[key])
    for key, groups in buckets.items():
        for name, values in sorted(groups.items()):
            print(f"{key:8} {name:12} {sum(values)/len(values):.4f} n {len(values)}")

by_type("results/action_retrieve_predictions.jsonl", ["dense", "term", "ce"])
by_type("results/policy_retrieve_predictions.jsonl", ["ace"])
