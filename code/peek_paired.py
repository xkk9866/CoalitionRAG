import json
from pathlib import Path
import numpy as np

preds = [json.loads(line) for line in (Path("results/cover_predictions.jsonl")).open(encoding="utf-8")]
main = [r for r in preds if r["split"] == "main"]
for ds in ["hotpot", "2wiki", "musique"]:
    cover = {r["id"]: r for r in main if r["dataset"] == ds and r["method"] == "cover"}
    single = {r["id"]: r for r in main if r["dataset"] == ds and r["method"] == "single"}
    diffs = []
    for i, c in cover.items():
        if c["agents"] == ["dense"]:
            continue
        diffs.append(c["f1"] - single[i]["f1"])
    arr = np.array(diffs)
    se = arr.std(ddof=1) / np.sqrt(len(arr))
    print(ds, "n", len(arr), "mean", round(arr.mean(), 3), "se", round(se, 3))
