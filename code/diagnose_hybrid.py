"""Insert capitalized question titles ahead of the greedy dossier. Calibration only."""

import json
import re

import open_retrieve as ore
import run_action as act
import run_fair

WORD = re.compile(r"[0-9A-Za-z\u00C0-\u024F]+(?:'[0-9A-Za-z\u00C0-\u024F]+)?")


def cap_anchors(question: str, titles: dict[str, int]) -> list[int]:
    tokens = [(m.group(0), m.group(0)[:1].isupper()) for m in WORD.finditer(question)]
    mentions = ore.maximal_title_matches([tok.lower() for tok, _ in tokens], titles)
    kept = []
    for mention in mentions:
        caps = [tokens[i][1] for i in range(mention.start, mention.end)]
        if caps and sum(caps) * 2 > len(caps):
            kept.append(mention)
    kept.sort(key=lambda item: (-(item.end - item.start), item.start))
    out, seen = [], set()
    for mention in kept:
        if mention.rowid in seen:
            continue
        seen.add(mention.rowid)
        out.append(mention.rowid)
        if len(out) >= 2:
            break
    return out


def main() -> None:
    rows = {row["id"]: row for row in run_fair.hotpot_rows("cal")}
    records = [json.loads(line) for line in (run_fair.ro.RESULT_DIR / "chain_cal_predictions.jsonl").open(encoding="utf-8")]
    db = ore.WikiDB()
    hybrid = ace = 0.0
    better = worse = 0
    try:
        for record in records:
            row = rows[record["id"]]
            gold = set(row["gold_keys"])
            anchors = cap_anchors(row["question"], db.titles)
            order = anchors + [rowid for rowid in record["ace_ids"] if rowid not in set(anchors)]
            pages = db.fetch(order[:4])
            keys = {pages[rowid].key for rowid in order[:4] if rowid in pages}
            score = act.support(keys, gold)
            hybrid += score
            ace += record["ace"]
            if score > record["ace"]:
                better += 1
            elif score < record["ace"]:
                worse += 1
    finally:
        db.close()
    n = len(records)
    print(json.dumps({"n": n, "ace": ace / n, "hybrid": hybrid / n, "better": better, "worse": worse}))


if __name__ == "__main__":
    main()
