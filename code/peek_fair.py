import json
from pathlib import Path

arms = ["flat", "overlap", "surface", "title", "ace", "lexlink", "qtitle"]
path = Path("results/fair_2wiki_predictions.jsonl")
total = {name: 0.0 for name in arms}
n = 0
for line in path.open(encoding="utf-8"):
    if not line.strip():
        continue
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    n += 1
    for name in arms:
        total[name] += row[name]
print("n", n)
for name in arms:
    print(name, round(total[name] / n, 4))
