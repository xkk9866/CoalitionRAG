import json
from pathlib import Path

rows = []
for line in Path("results/policy_retrieve_predictions.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        rows.append(json.loads(line))
hits = [row for row in rows if row["type"] == "bridge" and row["ace"] == 1.0 and row["outside_hit"] >= 1]
print("bridge both and outside", len(hits))
for row in hits[:8]:
    print(row["id"], row["outside"], row["outside_hit"], " | ".join(row["titles"]))
