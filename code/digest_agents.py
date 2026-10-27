import json
from collections import Counter, defaultdict
from pathlib import Path

path = Path(__file__).resolve().parent / "results" / "full_predictions.jsonl"
rows = [json.loads(line) for line in path.open(encoding="utf-8")]
print("rows", len(rows))

def rate(subset, name):
    if not subset:
        return 0.0
    return sum(name in r["agents"] for r in subset) / len(subset)

for ds in ["hotpot", "2wiki", "musique"]:
    print("==", ds)
    for method in ["coalition", "no_credit", "no_cost", "no_profile", "no_rev", "router", "debate"]:
        sub = [r for r in rows if r["dataset"] == ds and r["method"] == method and r["split"] in ("main", "ablation")]
        if not sub:
            continue
        counts = Counter()
        for r in sub:
            for a in r["agents"]:
                counts[a] += 1
        n = len(sub)
        shown = {k: round(v / n, 3) for k, v in sorted(counts.items())}
        print(f"  {method:12} n={n} size={sum(len(r['agents']) for r in sub)/n:.2f} {shown}")

print("== robust coalition agents")
for nweak in [0, 1, 2, 4]:
    sub = [r for r in rows if r["split"] == f"robust{nweak}" and r["method"] == "coalition"]
    counts = Counter()
    for r in sub:
        for a in r["agents"]:
            counts[a] += 1
    n = len(sub)
    shown = {k: round(v / n, 3) for k, v in sorted(counts.items())}
    print(nweak, shown)
