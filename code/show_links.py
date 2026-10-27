import json
from pathlib import Path

import open_retrieve as ore

target = "5adf37a95542995ec70e8f97"
row = None
for line in Path("results/policy_retrieve_predictions.jsonl").read_text(encoding="utf-8").splitlines():
    if target in line:
        row = json.loads(line)
        break
db = ore.WikiDB()
try:
    pages = db.fetch(row["rowids"])
    for rowid in row["rowids"]:
        page = pages[rowid]
        names = []
        for link in page.links:
            key = link.get("t")
            if key in db.titles:
                names.append(f"{link.get('a')} -> {key}")
        print(page.title)
        print(" text:", page.text[:280].replace("\n", " "))
        print(" links:", " | ".join(names[:12]))
        print()
finally:
    db.close()
