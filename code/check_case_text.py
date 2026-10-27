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
db = ore.WikiDB()
try:
    pages = db.fetch(saved["flat_ids"] + saved["qtitle_ids"] + saved["ace_ids"])
    for arm in ("flat", "qtitle", "ace"):
        print("ARM", arm)
        for rowid in saved[arm + "_ids"]:
            text = pages[rowid].text or ""
            print(pages[rowid].title, "NORTH" if "North Atlantic" in text else "no-name", "1988" if "1988" in text else "no-year")
finally:
    db.close()
