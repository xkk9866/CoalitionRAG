import json
from pathlib import Path

rows = [json.loads(line) for line in Path("results/policy_retrieve_predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
n4 = sum(1 for row in rows if len(row["rowids"]) == 4)
n3 = sum(1 for row in rows if len(row["rowids"]) == 3)
n2 = sum(1 for row in rows if len(row["rowids"]) <= 2)
print("n", len(rows), "eq4", n4, "eq3", n3, "le2", n2, "mean", sum(len(row["rowids"]) for row in rows) / len(rows))
