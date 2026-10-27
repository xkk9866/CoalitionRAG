import json
from collections import defaultdict
from pathlib import Path

path = Path("results/policy_retrieve_predictions.jsonl")
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
by = defaultdict(list)
outside = 0
hit = 0
both = 0
for row in rows:
    by[row["type"]].append(row["ace"])
    outside += row["outside"]
    hit += row["outside_hit"]
    both += int(row["ace"] == 1.0)
ace = sum(row["ace"] for row in rows) / len(rows)
print(f"n {len(rows)} ace {ace:.4f} both {both} outside {outside} hit {hit}")
for name, values in sorted(by.items()):
    print(f"{name} n {len(values)} ace {sum(values) / len(values):.4f}")
