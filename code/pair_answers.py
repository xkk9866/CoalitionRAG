import json
from pathlib import Path

import numpy as np

root = Path("results")


def load(path, key="f1"):
    out = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                out[row["id"]] = float(row[key])
    return out


def boot(left, right):
    diff = np.asarray(left) - np.asarray(right)
    se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(diff), size=(10000, len(diff)))
    means = diff[draws].mean(axis=1)
    return {
        "n": int(len(diff)),
        "delta": float(diff.mean()),
        "se": se,
        "boot_low": float(np.quantile(means, 0.025)),
        "boot_high": float(np.quantile(means, 0.975)),
    }


ace = load(root / "policy_answer_3b_predictions.jsonl")
flat = load(root / "fair_answer_flat_3b_test300_predictions.jsonl")
lex = load(root / "fair_answer_lexlink_3b_test300_predictions.jsonl")
ids = [i for i in ace if i in flat and i in lex]
print("n", len(ids))
print("ace_vs_flat", boot([ace[i] for i in ids], [flat[i] for i in ids]))
print("ace_vs_lex", boot([ace[i] for i in ids], [lex[i] for i in ids]))
print("lex_vs_flat", boot([lex[i] for i in ids], [flat[i] for i in ids]))
print("means", float(np.mean([ace[i] for i in ids])), float(np.mean([lex[i] for i in ids])), float(np.mean([flat[i] for i in ids])))
