import json
from pathlib import Path

root = Path("results")
fair = {}
with (root / "fair_retrieve_predictions.jsonl").open(encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        fair[row["id"]] = row["ace_ids"]
old = {}
with (root / "policy_retrieve_predictions.jsonl").open(encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        old[row["id"]] = row["rowids"]
same = 0
missing = 0
for qid, ids in fair.items():
    if qid not in old:
        missing += 1
        continue
    if ids == old[qid]:
        same += 1
print("fair", len(fair), "old", len(old), "same", same, "missing", missing)
