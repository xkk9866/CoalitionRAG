import json
from pathlib import Path

data = json.loads(Path("results/open_retrieve_summary.json").read_text(encoding="utf-8"))
for name, row in data["methods"].items():
    delta = row.get("versus_dense_support", {})
    print(
        f"{name:12} {row['support']:.4f}  d {delta.get('delta', 0):+.4f}  se {delta.get('se', 0):.4f}  "
        f"out {row.get('outside_recovered_pages')}"
    )
