import json
from pathlib import Path

rows = [json.loads(line) for line in Path("results/nextq_cal_predictions.jsonl").open(encoding="utf-8")]
better = [row for row in rows if row["nextq"] > row["ace"]]
worse = [row for row in rows if row["nextq"] < row["ace"]]
print("better", len(better), "worse", len(worse), "same", len(rows) - len(better) - len(worse))
for row in better[:8]:
    print("UP", json.dumps({"ace": row["ace"], "next": row["nextq"], "q": row["question"], "rewrite": row["nextq_text"]}, ensure_ascii=False))
print("---")
for row in worse[:8]:
    print("DOWN", json.dumps({"ace": row["ace"], "next": row["nextq"], "q": row["question"], "rewrite": row["nextq_text"]}, ensure_ascii=False))
