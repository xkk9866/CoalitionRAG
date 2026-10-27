"""F1 by question type for the main split. Types come from the cached datasets."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset

rows = [
    json.loads(line)
    for line in (Path(__file__).resolve().parent / "results" / "full_predictions.jsonl").open(encoding="utf-8")
]
main = [r for r in rows if r["split"] == "main"]

# Rebuild the same stratified ids.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiment import load_examples

# load_examples needs the dataset loaders inside the module. Call it.
id_type = {}
for name, n_cal, n_test in [("hotpot", 200, 300), ("2wiki", 200, 300), ("musique", 200, 300)]:
    cal, test = load_examples(name, n_cal, n_test, seed=7)
    # load_examples may not keep type. Inspect keys.
    print(name, "keys", sorted(test[0].keys()), "n", len(test))
    if "qtype" in test[0]:
        for ex in test:
            id_type[(name, ex["id"])] = ex["qtype"]
    elif "type" in test[0]:
        for ex in test:
            id_type[(name, ex["id"])] = ex["type"]

print("typed", len(id_type))
methods = ["single", "router", "fixed", "debate", "coalition"]
for ds in ["hotpot", "2wiki", "musique"]:
    types = sorted({id_type[(ds, r["id"])] for r in main if r["dataset"] == ds and (ds, r["id"]) in id_type})
    print("==", ds, types)
    for method in methods:
        sub = [r for r in main if r["dataset"] == ds and r["method"] == method]
        print(" ", method)
        for t in types:
            vals = [r["f1"] for r in sub if id_type.get((ds, r["id"])) == t]
            if not vals:
                continue
            print(f"    {t:20} n={len(vals):3d} F1={np.mean(vals):.3f}")
