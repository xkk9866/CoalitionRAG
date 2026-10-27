import json
from pathlib import Path

rows = [json.loads(line) for line in Path("results/define_cal_predictions.jsonl").open(encoding="utf-8")]
differ = [row for row in rows if row["bind_ids"] != row["ace_ids"]]
better = [row for row in differ if row["bind"] > row["ace"]]
worse = [row for row in differ if row["bind"] < row["ace"]]
print("n", len(rows), "differ", len(differ), "better", len(better), "worse", len(worse))
print("mean_delta", sum(row["bind"] - row["ace"] for row in rows) / len(rows))
