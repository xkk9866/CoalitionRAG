import json
import math
import random
from pathlib import Path

root = Path("results")


def load(path):
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def paired(left, right, key="f1"):
    ids = sorted(set(left) & set(right))
    delta = [left[i][key] - right[i][key] for i in ids]
    n = len(delta)
    mean = sum(delta) / n
    var = sum((x - mean) ** 2 for x in delta) / (n - 1)
    se = math.sqrt(var / n)
    rng = random.Random(0)
    boots = []
    for _ in range(10000):
        sample = [delta[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    return {
        "n": n,
        "delta": mean,
        "se": se,
        "boot_low": boots[250],
        "boot_high": boots[9749],
    }


ace = load(root / "policy_answer_3b_predictions.jsonl")
flat = load(root / "fair_answer_flat_3b_test300_predictions.jsonl")
lex = load(root / "fair_answer_lexlink_3b_test300_predictions.jsonl")
print("ace_vs_flat", json.dumps(paired(ace, flat)))
print("ace_vs_lex", json.dumps(paired(ace, lex)))
