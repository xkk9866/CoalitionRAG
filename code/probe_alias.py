"""How often a bound phrase is unique, and whether a missed gold page states a question phrase."""

from __future__ import annotations

import json

import open_retrieve as ore
import run_fair

rows = {row["id"]: row for row in run_fair.hotpot_rows("cal")}
records = [json.loads(line) for line in (run_fair.ro.RESULT_DIR / "chain_cal_predictions.jsonl").open(encoding="utf-8")]
db = ore.WikiDB()


def phrase_count(span: str) -> int:
    toks = [tok for tok in ore.title_tokens(span) if tok.isalnum()]
    if len(toks) < 2:
        return -1
    query = '"' + " ".join(toks) + '"'
    try:
        return int(db.con.execute("SELECT count(*) AS n FROM pages WHERE pages MATCH ?", (query,)).fetchone()["n"])
    except Exception:
        return -1


unique = 0
checked = 0
contain = 0
rare = 0
try:
    for record in records:
        if record["ace"] >= 1:
            continue
        row = rows[record["id"]]
        opened = db.fetch(record["ace_ids"])
        opened_keys = {page.key for page in opened.values()}
        missing = [key for key in row["gold_keys"] if key not in opened_keys]
        q_tokens = ore.title_tokens(row["question"])
        for key in missing:
            rowid = db.titles.get(key)
            if rowid is None:
                continue
            page = db.fetch([rowid]).get(rowid)
            if page is None:
                continue
            blob = " " + " ".join(ore.title_tokens(page.title + " " + page.text)) + " "
            checked += 1
            best = ""
            for n in range(min(6, len(q_tokens)), 1, -1):
                for i in range(len(q_tokens) - n + 1):
                    span = " ".join(q_tokens[i : i + n])
                    if f" {span} " in blob and len(span) > len(best):
                        best = span
                        break
                if best:
                    break
            if not best:
                continue
            contain += 1
            count = phrase_count(best)
            if 0 < count <= 3:
                rare += 1
                if rare <= 12:
                    print(json.dumps({"q": row["question"], "missing": key, "span": best, "n": count}, ensure_ascii=False))
finally:
    db.close()
print(json.dumps({"miss_pages_in_index": checked, "question_phrase_in_page": contain, "phrase_in_at_most_3_pages": rare}, indent=2))
