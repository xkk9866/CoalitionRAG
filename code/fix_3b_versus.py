import json
from pathlib import Path

import numpy as np

root = Path("results")


def load_method(path, method):
    out = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("method") == method:
                out[row["id"]] = float(row["f1"])
    return out


def boot(left, right):
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
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


base = root / "open3b_predictions.jsonl"
dense = load_method(base, "dense")
links = load_method(base, "links")
for name in ("flat", "lexlink"):
    path = root / f"fair_answer_{name}_3b_test300_summary.json"
    pred = root / f"fair_answer_{name}_3b_test300_predictions.jsonl"
    rows = [json.loads(line) for line in pred.open(encoding="utf-8") if line.strip()]
    ids = [row["id"] for row in rows if row["id"] in dense]
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["versus_dense_f1"] = boot([row["f1"] for row in rows if row["id"] in dense], [dense[i] for i in ids])
    summary["versus_links_f1"] = boot([row["f1"] for row in rows if row["id"] in links], [links[i] for i in ids])
    summary["baseline_file"] = "open3b_predictions.jsonl"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(name, summary["f1"], summary["versus_dense_f1"]["delta"], summary["versus_links_f1"]["delta"])
