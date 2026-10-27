"""Does a second BM25 search with the restated question contain the gold page the link ranker missed?"""

from __future__ import annotations

import json

import open_retrieve as ore
import run_open as ro

records = [json.loads(line) for line in (ro.RESULT_DIR / "nextq_cal_predictions.jsonl").open(encoding="utf-8")]
rows = {row["id"]: row for row in ro.load_hotpot()}
db = ore.WikiDB()
miss = 0
hit10 = 0
hit4 = 0
examples = 0
try:
    for record in records:
        if record["ace"] >= 1:
            continue
        row = rows[record["id"]]
        opened = db.fetch(record["ace_ids"])
        opened_keys = {page.key for page in opened.values()}
        missing = [key for key in row["gold_keys"] if key not in opened_keys]
        found = db.search(record["nextq_text"], 10)
        pages = db.fetch(found)
        keys = [pages[rowid].key for rowid in found if rowid in pages]
        for key in missing:
            miss += 1
            if key in keys:
                hit10 += 1
            if key in keys[:4]:
                hit4 += 1
            if examples < 8 and key in keys:
                examples += 1
                print(json.dumps({"q": record["question"], "next": record["nextq_text"], "missing": key, "rank": keys.index(key) + 1}, ensure_ascii=False))
finally:
    db.close()
print(json.dumps({"miss_pages": miss, "in_rewrite_bm25_10": hit10, "in_rewrite_bm25_4": hit4}, indent=2))
