"""Print the pages chosen for one stored question."""

from __future__ import annotations

import json
import sys

import open_retrieve as ore
import run_open as ro

qid = sys.argv[1]
saved = None
with (ro.RESULT_DIR / "fair_retrieve_predictions.jsonl").open(encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] == qid:
            saved = row
            break
if saved is None:
    raise SystemExit("missing id")
hotpot = {row["id"]: row for row in ro.load_hotpot()}
src = hotpot[qid]
print("Q:", src["question"])
print("GOLD:", src["gold_titles"])
print("ANSWER:", src["answers"][0])
db = ore.WikiDB()
try:
    for arm in ("ace", "flat", "qtitle", "lexlink", "title"):
        pages = db.fetch(saved[arm + "_ids"])
        print("---", arm, saved[arm])
        for rowid in saved[arm + "_ids"]:
            page = pages.get(rowid)
            if page is None:
                continue
            print(page.title, "|", (page.text or "")[:240].replace("\n", " "))
finally:
    db.close()
