"""Oracle ceiling of capitalized-title anchors plus their hyperlinks."""

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
        if sum(caps) * 2 > len(caps):
            kept.append(mention)
    kept.sort(key=lambda item: (-(item.end - item.start), item.start))
    out = []
    seen = set()
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
    oracle_sum = ace_sum = 0.0
    changed = better = worse = 0
    try:
        for record in records:
            row = rows[record["id"]]
            gold = set(row["gold_keys"])
            anchors = [rowid for rowid in cap_anchors(row["question"], db.titles)]
            pages = db.fetch(anchors)
            links = run_fair.link_ids(anchors, pages, db.titles) if hasattr(run_fair, "link_ids") else []
            if not links:
                offers = []
                seen = set(anchors)
                for rowid in anchors:
                    page = pages.get(rowid)
                    if page is None:
                        continue
                    for link_id, _phrase in run_fair.link_offers(page, db.titles, seen):
                        seen.add(link_id)
                        offers.append(link_id)
                links = offers
            pool = db.fetch(anchors + links)
            # Oracle: anchors are fixed, the two link slots may be any legal links, including gold ones.
            gold_links = [rowid for rowid in links if rowid in pool and pool[rowid].key in gold and rowid not in set(anchors)]
            other = [rowid for rowid in links if rowid not in set(gold_links)]
            chosen = act.pack_ids(anchors, gold_links + other, pool)
            keys = {pool[rowid].key for rowid in chosen if rowid in pool}
            support = act.support(keys, gold)
            oracle_sum += support
            ace_sum += record["ace"]
            if abs(support - record["ace"]) > 1e-9:
                changed += 1
            if support > record["ace"]:
                better += 1
            elif support < record["ace"]:
                worse += 1
    finally:
        db.close()
    n = len(records)
    print(json.dumps({
        "n": n,
        "ace": ace_sum / n,
        "oracle_cap_links": oracle_sum / n,
        "changed": changed,
        "better": better,
        "worse": worse,
    }))


if __name__ == "__main__":
    main()
