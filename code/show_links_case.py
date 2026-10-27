import json

import open_retrieve as ore
import run_open as ro

qid = "5ab3e45655429976abd1bcd4"
saved = None
with (ro.RESULT_DIR / "fair_retrieve_predictions.jsonl").open(encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        if row["id"] == qid:
            saved = row
            break
print("outside", saved["outside"], "ace_hit", saved["ace_hit"], "flat_hit", saved["flat_hit"], "qtitle_hit", saved["qtitle_hit"])
db = ore.WikiDB()
try:
    pages = db.fetch(saved["ace_ids"])
    for rowid in saved["ace_ids"]:
        page = pages[rowid]
        print("PAGE", page.title)
        print(page.text)
        print("LINKS", [link.get("a") + " -> " + link.get("t", "") for link in page.links[:12]])
        print()
finally:
    db.close()
