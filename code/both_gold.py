import json
from pathlib import Path

def both(path, key):
    n = 0
    b = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            n += 1
            b += int(row[key] == 1.0)
    print(path, key, b, n, round(b / n, 4))

both("results/action_retrieve_predictions.jsonl", "dense")
both("results/action_retrieve_predictions.jsonl", "ce")
both("results/action_retrieve_predictions.jsonl", "term")
both("results/policy_retrieve_predictions.jsonl", "ace")
