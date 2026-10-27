"""Classify gold pages missed by the frozen full-question controller on the calibration 200."""

from __future__ import annotations

import json

import open_retrieve as ore
import run_fair

records = [json.loads(line) for line in (run_fair.ro.RESULT_DIR / "chain_cal_predictions.jsonl").open(encoding="utf-8")]
rows = {row["id"]: row for row in run_fair.hotpot_rows("cal")}
db = ore.WikiDB()
counts = {
    "questions": 0,
    "miss_pages": 0,
    "in_anchor_links": 0,
    "in_opened_hop2": 0,
    "in_bm25_not_opened": 0,
    "linked_from_bm25": 0,
    "unreachable": 0,
}
try:
    for record in records:
        if record["ace"] >= 1:
            continue
        counts["questions"] += 1
        row = rows[record["id"]]
        gold = set(row["gold_keys"])
        opened = db.fetch(record["ace_ids"])
        opened_keys = {page.key for page in opened.values()}
        missing = [key for key in gold if key not in opened_keys]
        anchor_ids = record["ace_ids"][:2]
        anchors = [opened[rowid] for rowid in anchor_ids if rowid in opened]
        hop1 = {}
        blocked = set(anchor_ids)
        for page in anchors:
            for rowid, _phrase in run_fair.link_offers(page, db.titles, blocked):
                hop1[rowid] = page.rowid
                blocked.add(rowid)
        hop1_pages = db.fetch(list(hop1))
        hop2 = set()
        opened_links = [opened[rowid] for rowid in record["ace_ids"][2:] if rowid in opened]
        blocked2 = set(record["ace_ids"])
        for page in opened_links:
            for rowid, _phrase in run_fair.link_offers(page, db.titles, blocked2):
                hop2.add(rowid)
                blocked2.add(rowid)
        hop2_pages = db.fetch(list(hop2))
        bm25_ids = db.search(row["question"], ore.FIRST_HOP_K)
        bm25 = set(bm25_ids)
        observed = db.fetch(bm25_ids)
        observed_links = set()
        blocked_obs = set(bm25_ids)
        for rowid in bm25_ids:
            page = observed.get(rowid)
            if page is None:
                continue
            for link_id, _phrase in run_fair.link_offers(page, db.titles, blocked_obs):
                observed_links.add(link_id)
                blocked_obs.add(link_id)
        for key in missing:
            counts["miss_pages"] += 1
            rowid = db.titles.get(key)
            if rowid in hop1:
                counts["in_anchor_links"] += 1
            elif rowid in hop2_pages:
                counts["in_opened_hop2"] += 1
            elif rowid in bm25:
                counts["in_bm25_not_opened"] += 1
            elif rowid in observed_links:
                counts["linked_from_bm25"] += 1
            else:
                counts["unreachable"] += 1
                if counts["unreachable"] <= 8:
                    print(json.dumps({
                        "q": row["question"],
                        "gold": row["gold_titles"] if "gold_titles" in row else row["gold_keys"],
                        "missing": key,
                        "opened": [opened[i].title for i in record["ace_ids"] if i in opened],
                    }, ensure_ascii=False), flush=True)
finally:
    db.close()
print(json.dumps(counts, indent=2))
