import json
from pathlib import Path

import run_open as ro

want = {
    "5a8c7595554299585d9e36b6",
    "5a722b8655429971e9dc9329",
    "5adf37a95542995ec70e8f97",
    "5adddccd5542997dc7907069",
}
saved = {}
for line in Path("results/policy_retrieve_predictions.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if row["id"] in want:
        saved[row["id"]] = row
rows = ro.load_hotpot()
for row in rows:
    if row["id"] not in want:
        continue
    got = saved[row["id"]]
    print("---")
    print(row["id"])
    print(row["question"])
    print("gold", row["gold_titles"])
    print("picked", " | ".join(got["titles"]))
