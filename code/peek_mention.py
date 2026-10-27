import json

import open_retrieve as ore
import run_mention

rows = {row["id"]: row for row in run_mention.cal_rows()}
records = [json.loads(line) for line in (run_mention.ro.RESULT_DIR / "mention_cal_predictions.jsonl").open(encoding="utf-8")]
db = ore.WikiDB()
shown = 0
try:
    for record in records:
        if record["ace"] - record["named"] < 0.5:
            continue
        row = rows[record["id"]]
        pages = db.fetch(record["ace_ids"] + record["named_ids"])
        named = run_mention.named_pages(row["question"], db.titles)
        named_pages = db.fetch(named)
        print(json.dumps({
            "q": row["question"],
            "gold": row["gold_titles"],
            "spans": [named_pages[i].title for i in named if i in named_pages],
            "ace": [pages[i].title for i in record["ace_ids"] if i in pages],
            "named": [pages[i].title for i in record["named_ids"] if i in pages],
            "ace_s": record["ace"],
            "named_s": record["named"],
        }, ensure_ascii=False))
        shown += 1
        if shown >= 6:
            break
finally:
    db.close()
