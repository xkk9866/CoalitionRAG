"""Do capitalized title spans name gold pages the greedy dossier missed?"""

import json
import re

import open_retrieve as ore
import run_chain
import run_fair

WORD = re.compile(r"[0-9A-Za-z\u00C0-\u024F]+(?:'[0-9A-Za-z\u00C0-\u024F]+)?")


def surface_tokens(text: str):
    return [(m.group(0), m.group(0)[:1].isupper()) for m in WORD.finditer(text)]


def kept_spans(question: str, titles: dict[str, int]):
    tokens = surface_tokens(question)
    lower = [tok.lower() for tok, _cap in tokens]
    mentions = ore.maximal_title_matches(lower, titles)
    kept = []
    for mention in mentions:
        caps = [tokens[i][1] for i in range(mention.start, mention.end)]
        if sum(caps) * 2 > len(caps):
            kept.append(mention)
    kept.sort(key=lambda item: (-(item.end - item.start), item.start))
    return mentions, kept


def main() -> None:
    rows = {row["id"]: row for row in run_fair.hotpot_rows("cal")}
    records = [json.loads(line) for line in (run_fair.ro.RESULT_DIR / "chain_cal_predictions.jsonl").open(encoding="utf-8")]
    db = ore.WikiDB()
    gold_hit = gold_cap = gold_pages = 0
    missed = missed_cap = 0
    questions_cap_helps = 0
    try:
        for record in records:
            row = rows[record["id"]]
            gold = set(row["gold_keys"])
            mentions, kept = kept_spans(row["question"], db.titles)
            mention_keys = {db_key(db, m.rowid) for m in mentions}
            cap_keys = {db_key(db, m.rowid) for m in kept}
            opened = {db_key(db, rowid) for rowid in record["ace_ids"]}
            gold_pages += len(gold)
            gold_hit += len(gold & mention_keys)
            gold_cap += len(gold & cap_keys)
            missing = gold - opened
            missed += len(missing)
            missed_cap += len(missing & cap_keys)
            if missing & cap_keys:
                questions_cap_helps += 1
                if questions_cap_helps <= 8:
                    pages = db.fetch([m.rowid for m in kept])
                    print(json.dumps({
                        "q": row["question"],
                        "gold": row["gold_titles"],
                        "cap": [pages[m.rowid].title for m in kept if m.rowid in pages],
                        "ace": record["ace"],
                    }, ensure_ascii=False))
    finally:
        db.close()
    print(json.dumps({
        "n": len(records),
        "gold_pages": gold_pages,
        "gold_in_any_span": gold_hit,
        "gold_in_cap_span": gold_cap,
        "missed_by_ace": missed,
        "missed_but_cap_span": missed_cap,
        "questions_where_cap_names_a_miss": questions_cap_helps,
    }))


def db_key(db, rowid: int) -> str:
    page = db.fetch([rowid]).get(rowid)
    return "" if page is None else page.key


if __name__ == "__main__":
    main()
