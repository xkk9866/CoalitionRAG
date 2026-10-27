"""Mention-anchored chain.

The question names the page where reading starts. That page stays in the
dossier even when a neighbor in the BM25 list scores higher. A page the
question does not name can be opened only as a hyperlink of a named page.
The cross-encoder ranks those links and does not choose the anchor.

Calibration uses 200 questions that no earlier controller used. The chain is
adopted only when it beats greedy cross-encoder anchors by 0.01. Otherwise
the greedy controller stays, and the test split is not scored.
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("OPEN_CUDA", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import open_retrieve as ore
import run_action as act
import run_bind
import run_fair
import run_open as ro

RESULT = ro.RESULT_DIR / "mention_hparams.json"
MIN_GAIN = 0.01
ARMS = ("flat", "ace", "named", "namedfill")
CAL_SEED = 41
PRIOR_CAL_SEEDS = (23, 29, 31, 37)


def named_pages(question: str, titles: dict[str, int], k: int = 2) -> list[int]:
    mentions = ore.maximal_title_matches(ore.title_tokens(question), titles)
    mentions.sort(key=lambda item: (-(item.end - item.start), item.start))
    out = []
    seen = set()
    for mention in mentions:
        if mention.rowid in seen:
            continue
        seen.add(mention.rowid)
        out.append(mention.rowid)
        if len(out) >= k:
            break
    return out


def link_ids(anchors: list[int], pages: dict[int, ore.Page], titles: dict[str, int]) -> list[int]:
    out = []
    seen = set(anchors)
    for rowid in anchors:
        page = pages.get(rowid)
        if page is None:
            continue
        for link_id, _phrase in run_fair.link_offers(page, titles, seen):
            seen.add(link_id)
            out.append(link_id)
    return out


def evaluate(rows: list[dict], db: ore.WikiDB, reranker) -> list[dict]:
    records = []
    for start in range(0, len(rows), 40):
        batch = rows[start : start + 40]
        t0 = time.perf_counter()
        bm25_lists = db.search_many([row["question"] for row in batch], ore.FIRST_HOP_K, workers=8)
        named = {row["id"]: named_pages(row["question"], db.titles) for row in batch}
        pages = db.fetch([rowid for ids in bm25_lists for rowid in ids] + [rowid for ids in named.values() for rowid in ids])
        pairs, owners = [], []
        seen_pairs = set()
        for row, ids in zip(batch, bm25_lists):
            for rowid in list(ids) + named[row["id"]]:
                if rowid not in pages or (row["id"], rowid) in seen_pairs:
                    continue
                seen_pairs.add((row["id"], rowid))
                page = pages[rowid]
                pairs.append((row["question"], page.title + ". " + page.text))
                owners.append((row["id"], rowid))
        scores: dict[str, dict[int, float]] = {}
        for (qid, rowid), value in zip(owners, run_fair.batched_scores(reranker, pairs)):
            scores.setdefault(qid, {})[rowid] = value

        needed = []
        for row, ids in zip(batch, bm25_lists):
            qid = row["id"]
            ace_anchor = run_fair.top_scored(ids, scores.get(qid, {}), ore.ANCHOR_K, set())
            fill = run_fair.top_scored(ids, scores.get(qid, {}), ore.ANCHOR_K, set(named[qid]))
            for anchor in (ace_anchor, named[qid], named[qid] + fill):
                for rowid in anchor:
                    page = pages.get(rowid)
                    if page is None:
                        continue
                    for link_id, _phrase in run_fair.link_offers(page, db.titles, set(anchor)):
                        if link_id not in scores.get(qid, {}) and link_id not in pages:
                            needed.append(link_id)
        if needed:
            pages.update(db.fetch(needed))
        pairs, owners = [], []
        for row, ids in zip(batch, bm25_lists):
            qid = row["id"]
            ace_anchor = run_fair.top_scored(ids, scores.get(qid, {}), ore.ANCHOR_K, set())
            fill = run_fair.top_scored(ids, scores.get(qid, {}), ore.ANCHOR_K, set(named[qid]))
            wanted = []
            seen = set(scores.get(qid, {}))
            for anchor in (ace_anchor, named[qid], named[qid] + fill):
                for link_id in link_ids(anchor, pages, db.titles):
                    if link_id not in seen and link_id in pages:
                        seen.add(link_id)
                        wanted.append(link_id)
            for rowid in wanted:
                page = pages[rowid]
                pairs.append((row["question"], page.title + ". " + page.text))
                owners.append((qid, rowid))
        for (qid, rowid), value in zip(owners, run_fair.batched_scores(reranker, pairs)):
            scores.setdefault(qid, {})[rowid] = value

        for row, ids in zip(batch, bm25_lists):
            qid = row["id"]
            gold = set(row["gold_keys"])
            sc = scores.get(qid, {})
            ace_anchor = run_fair.top_scored(ids, sc, ore.ANCHOR_K, set())
            mention = [rowid for rowid in named[qid] if rowid in pages]
            fill = run_fair.top_scored(ids, sc, ore.ANCHOR_K, set(mention))
            fill_anchor = (mention + fill)[: ore.ANCHOR_K]
            packed = {
                "flat": run_fair.top_scored(ids, sc, ore.BUDGET, set()),
                "ace": act.pack_ids(ace_anchor, run_fair.top_scored(link_ids(ace_anchor, pages, db.titles), sc, ore.BUDGET, set(ace_anchor)), pages),
                "named": act.pack_ids(mention or ace_anchor, run_fair.top_scored(link_ids(mention or ace_anchor, pages, db.titles), sc, ore.BUDGET, set(mention or ace_anchor)), pages),
                "namedfill": act.pack_ids(fill_anchor, run_fair.top_scored(link_ids(fill_anchor, pages, db.titles), sc, ore.BUDGET, set(fill_anchor)), pages),
            }
            pool = {pages[rowid].key for rowid in ids if rowid in pages}
            outside = [key for key in gold if key not in pool]
            record = {"id": qid, "type": row["type"], "outside": len(outside), "n_named": len(mention)}
            for name, chosen in packed.items():
                keys = {pages[rowid].key for rowid in chosen if rowid in pages}
                record[name] = act.support(keys, gold)
                record[name + "_ids"] = chosen
            records.append(record)
        print(f"mention {start + len(batch)} / {len(rows)} sec {time.perf_counter() - t0:.1f}", flush=True)
    return records


def prior_ids(rows: list[dict]) -> set[str]:
    blocked: set[str] = set()
    for seed in PRIOR_CAL_SEEDS:
        idx = run_bind.fresh_indices(rows, seed=seed, extra_ids=blocked)
        blocked.update(rows[i]["id"] for i in idx)
    return blocked


def cal_rows() -> list[dict]:
    rows = ro.load_hotpot()
    idx = run_bind.fresh_indices(rows, seed=CAL_SEED, extra_ids=prior_ids(rows))
    return [rows[i] for i in idx]


def choose(scores: dict[str, float]) -> str:
    best = max(("named", "namedfill"), key=lambda name: scores[name])
    if scores[best] >= scores["ace"] + MIN_GAIN:
        return best
    return "ace"


def evaluate_lex(rows: list[dict], db: ore.WikiDB) -> list[dict]:
    """Same chain, with BM25 as the only ranker. No cross-encoder is loaded."""
    records = []
    for start in range(0, len(rows), 40):
        batch = rows[start : start + 40]
        t0 = time.perf_counter()
        bm25_lists = db.search_many([row["question"] for row in batch], ore.FIRST_HOP_K, workers=8)
        named = {row["id"]: named_pages(row["question"], db.titles) for row in batch}
        pages = db.fetch([rowid for ids in bm25_lists for rowid in ids] + [rowid for ids in named.values() for rowid in ids])
        wanted = []
        for row, ids in zip(batch, bm25_lists):
            mention = [rowid for rowid in named[row["id"]] if rowid in pages]
            lexical_anchor = list(ids[: ore.ANCHOR_K])
            fill = [rowid for rowid in ids if rowid not in set(mention)][: ore.ANCHOR_K]
            for anchor in (lexical_anchor, mention, (mention + fill)[: ore.ANCHOR_K]):
                wanted.extend(link_ids(anchor, pages, db.titles))
        missing = [rowid for rowid in dict.fromkeys(wanted) if rowid not in pages]
        if missing:
            pages.update(db.fetch(missing))
        for row, ids in zip(batch, bm25_lists):
            qid = row["id"]
            gold = set(row["gold_keys"])
            mention = [rowid for rowid in named[qid] if rowid in pages]
            lexical_anchor = [rowid for rowid in ids[: ore.ANCHOR_K] if rowid in pages]
            fill = [rowid for rowid in ids if rowid not in set(mention)]
            fill_anchor = (mention + fill)[: ore.ANCHOR_K]
            rank_ids = list(dict.fromkeys(
                link_ids(lexical_anchor, pages, db.titles)
                + link_ids(mention or lexical_anchor, pages, db.titles)
                + link_ids(fill_anchor, pages, db.titles)
                + list(ids)
            ))
            raw = db.bm25_scores(row["question"], rank_ids)
            sc = {rowid: -raw.get(rowid, 1e6) for rowid in rank_ids}
            packed = {
                "flat": list(ids[: ore.BUDGET]),
                "lexanchor": act.pack_ids(lexical_anchor, run_fair.top_scored(link_ids(lexical_anchor, pages, db.titles), sc, ore.BUDGET, set(lexical_anchor)), pages),
                "named": act.pack_ids(mention or lexical_anchor, run_fair.top_scored(link_ids(mention or lexical_anchor, pages, db.titles), sc, ore.BUDGET, set(mention or lexical_anchor)), pages),
                "namedfill": act.pack_ids(fill_anchor, run_fair.top_scored(link_ids(fill_anchor, pages, db.titles), sc, ore.BUDGET, set(fill_anchor)), pages),
            }
            record = {"id": qid, "type": row["type"], "n_named": len(mention)}
            for name, chosen in packed.items():
                keys = {pages[rowid].key for rowid in chosen if rowid in pages}
                record[name] = act.support(keys, gold)
            records.append(record)
        print(f"mention-lex {start + len(batch)} / {len(rows)} sec {time.perf_counter() - t0:.1f}", flush=True)
    return records


def run_lex() -> None:
    chosen = cal_rows()
    db = ore.WikiDB()
    try:
        records = evaluate_lex(chosen, db)
    finally:
        db.close()
    arms = ("flat", "lexanchor", "named", "namedfill")
    scores = {name: act.mean(records, name) for name in arms}
    by_type = {}
    for kind in sorted({row["type"] for row in records}):
        group = [row for row in records if row["type"] == kind]
        by_type[kind] = {name: act.mean(group, name) for name in arms}
    best = max(("named", "namedfill"), key=lambda name: scores[name])
    policy = best if scores[best] >= scores["lexanchor"] + MIN_GAIN else "lexanchor"
    payload = {
        "n": len(records),
        "seed": CAL_SEED,
        "ranker": "bm25",
        "scores": scores,
        "by_type": by_type,
        "named_questions": sum(1 for row in records if row["n_named"]),
        "policy": policy,
        "frozen_before_test": False,
        "note": "Lexical probe only. It does not replace the cross-encoder comparison.",
    }
    (ro.RESULT_DIR / "mention_lex_cal.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


def run_cal() -> None:
    chosen = cal_rows()
    db = ore.WikiDB()
    reranker = run_fair.load_reranker()
    try:
        records = evaluate(chosen, db, reranker)
    finally:
        db.close()
    with (ro.RESULT_DIR / "mention_cal_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")
    scores = {name: act.mean(records, name) for name in ARMS}
    by_type = {}
    for kind in sorted({row["type"] for row in records}):
        group = [row for row in records if row["type"] == kind]
        by_type[kind] = {name: act.mean(group, name) for name in ARMS}
    payload = {
        "n": len(records),
        "seed": CAL_SEED,
        "min_gain": MIN_GAIN,
        "scores": scores,
        "by_type": by_type,
        "named_questions": sum(1 for row in records if row["n_named"]),
        "policy": choose(scores),
        "frozen_before_test": True,
        "rule": "Adopt the mention chain only if it beats greedy anchors by 0.01.",
    }
    RESULT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["cal", "lex"], required=True)
    args = parser.parse_args()
    if args.stage == "lex":
        run_lex()
        return
    run_cal()


if __name__ == "__main__":
    main()
