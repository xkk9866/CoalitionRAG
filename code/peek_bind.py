import json
from pathlib import Path

rows = [json.loads(line) for line in Path("results/bind_cal_predictions.jsonl").open(encoding="utf-8")]
differ = [row for row in rows if row["bind_ids"] != row["ace_ids"]]
better = [row for row in differ if row["bind"] > row["ace"]]
worse = [row for row in differ if row["bind"] < row["ace"]]
print("differ", len(differ), "better", len(better), "worse", len(worse))
from collections import Counter

print(Counter(row["bound"] for row in differ).most_common(30))
