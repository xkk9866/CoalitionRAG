"""R@2 / R@5 / AR@5 / EM / F1 of the HippoRAG 2 and seed runs, printed as json."""
import json, sys
import numpy as np
from common import RES, jload, load_bench, recall_at

out = {}
for tag in sys.argv[1:]:
    for ds in ("musique", "2wikimultihopqa", "hotpotqa"):
        try:
            rows = jload(RES / "retrieval" / f"{tag}_{ds}.json")
        except FileNotFoundError:
            continue
        docs, qs = load_bench(ds)
        r2 = [recall_at((x.get("top2") or x["rank"][:2]), q["gold"]) for x, q in zip(rows, qs)]
        r5 = [recall_at((x.get("top5") or x["rank"][:5]), q["gold"]) for x, q in zip(rows, qs)]
        d = {"R@2": 100 * np.mean(r2), "R@5": 100 * np.mean(r5), "AR@5": 100 * np.mean([float(v == 1) for v in r5])}
        for rd in ("qwen25-7b-awq", "qwen3-8b-awq"):
            try:
                s = jload(RES / "qa" / f"{rd}_{tag}_{ds}.json")["summary"]
                d[rd] = {"EM": 100 * s["EM"], "F1": 100 * s["F1"]}
            except FileNotFoundError:
                pass
        out[f"{tag}/{ds}"] = d
print(json.dumps(out, indent=1))
