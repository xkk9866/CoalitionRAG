"""Where dossier-level anchor search disagrees with greedy anchors."""

import json

import open_retrieve as ore
import run_reach

rows = {row["id"]: row for row in run_reach.cal_rows()}
records = [
    json.loads(line)
    for line in (ore.RESULT_DIR if hasattr(ore, "RESULT_DIR") else run_reach.ro.RESULT_DIR).joinpath(
        "reach_cal_predictions.jsonl"
    ).open(encoding="utf-8")
]
db = ore.WikiDB()
try:
    better = worse = same = 0
    examples = []
    for record in records:
        row = rows[record["id"]]
        delta = record["reach8"] - record["ace"]
        if delta > 0:
            better += 1
        elif delta < 0:
            worse += 1
        else:
            same += 1
        if len(examples) < 12 and delta != 0:
            wanted = set(record["ace_ids"] + record["reach8_ids"])
            pages = db.fetch(list(wanted))
            examples.append({
                "delta": delta,
                "type": record["type"],
                "ace": record["ace"],
                "reach8": record["reach8"],
                "flat": record["flat"],
                "q": row["question"],
                "gold": row["gold_titles"],
                "ace_titles": [pages[i].title for i in record["ace_ids"] if i in pages],
                "reach_titles": [pages[i].title for i in record["reach8_ids"] if i in pages],
            })
    print(json.dumps({"better": better, "worse": worse, "same": same, "n": len(records)}))
    for item in examples:
        print(json.dumps(item, ensure_ascii=False))
finally:
    db.close()
