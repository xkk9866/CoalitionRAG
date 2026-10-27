"""Unresolved-clause ranking of hyperlinks.

The anchor is still the two highest cross-encoder pages in the BM25 top 100.
The next page must be a hyperlink on that anchor. It is chosen with the
question tokens that do not already occur in the anchor titles: the clause
the opened pages have not named. Calibration may adopt that clause only when
it beats scoring the full question by 0.01. The test split is scored once.
"""

from __future__ import annotations

import json
import os
import re

os.environ.setdefault("OPEN_CUDA", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import time

import open_retrieve as ore
import run_action as act
import run_fair
import run_open as ro

RESULT = ro.RESULT_DIR / "chain_hparams.json"
MIN_GAIN = 0.01
ARMS = ("flat", "ace", "gap_local", "gap_sent", "gap_title", "gap_abs")
POLICIES = ("gap_local", "gap_sent", "gap_title", "gap_abs", "ace")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sentence_with(page: ore.Page, phrase: str) -> str:
    text = page.text or ""
    needle = (phrase or "").lower()
    for part in SENT_SPLIT.split(text):
        if needle and needle in part.lower():
            return part.strip()
    return run_fair.source_span(page, phrase)


def residual(question: str, anchors: list[ore.Page]) -> str:
    ban = set()
    for page in anchors:
        ban.update(ore.title_tokens(page.title))
    kept = [tok for tok in ore.title_tokens(question) if tok not in ban]
    if len(kept) < 4:
        return question
    return " ".join(kept)


def local_links(offers: list[tuple[int, str, str]], scores: dict[int, float], k: int) -> list[int]:
    """Keep hyperlinks from the sentences the unresolved clause ranks first."""
    groups: dict[str, list[int]] = {}
    order: list[str] = []
    for rowid, _phrase, sent in offers:
        if sent not in groups:
            groups[sent] = []
            order.append(sent)
        groups[sent].append(rowid)
    ranked = sorted(order, key=lambda sent: scores.get(groups[sent][0], -1e9), reverse=True)
    out: list[int] = []
    seen: set[int] = set()
    for sent in ranked:
        for rowid in groups[sent]:
            if rowid in seen:
                continue
            seen.add(rowid)
            out.append(rowid)
            if len(out) >= k:
                return out
    return out


def choose(scores: dict[str, float]) -> str:
    """Cheapest preview that beats the full question by 0.01 and stays with the best clause."""
    best = max(scores[name] for name in POLICIES)
    for name in POLICIES:
        if scores[name] >= best - MIN_GAIN and scores[name] >= scores["ace"] + MIN_GAIN:
            return name
    return "ace"


def evaluate(rows: list[dict], db: ore.WikiDB, reranker) -> list[dict]:
    records = []
    for start in range(0, len(rows), 40):
        batch = rows[start : start + 40]
        t0 = time.perf_counter()
        print(f"chain batch {start}:{start + len(batch)} / {len(rows)}", flush=True)
        bm25_lists = db.search_many([row["question"] for row in batch], ore.FIRST_HOP_K, workers=8)
        pages = db.fetch([rowid for ids in bm25_lists for rowid in ids])
        flat_pairs = []
        flat_owners = []
        for row, ids in zip(batch, bm25_lists):
            for rowid in ids:
                page = pages.get(rowid)
                if page is None:
                    continue
                flat_pairs.append((row["question"], page.title + ". " + page.text))
                flat_owners.append((row["id"], rowid))
        flat_values = run_fair.batched_scores(reranker, flat_pairs)
        flat_scores: dict[str, dict[int, float]] = {}
        for (qid, rowid), value in zip(flat_owners, flat_values):
            flat_scores.setdefault(qid, {})[rowid] = value

        offers: dict[str, list[tuple[int, str, str]]] = {}
        anchors: dict[str, list[int]] = {}
        gaps: dict[str, str] = {}
        for row, ids in zip(batch, bm25_lists):
            anchor = run_fair.top_scored(ids, flat_scores.get(row["id"], {}), ore.ANCHOR_K, set())
            anchors[row["id"]] = anchor
            anchor_pages = [pages[rowid] for rowid in anchor if rowid in pages]
            gaps[row["id"]] = residual(row["question"], anchor_pages)
            blocked = set(anchor)
            found = []
            for page in anchor_pages:
                for rowid, phrase in run_fair.link_offers(page, db.titles, blocked):
                    found.append((rowid, phrase, sentence_with(page, phrase)))
                blocked.update(item[0] for item in found)
            offers[row["id"]] = found

        sent_pairs = []
        sent_owners = []
        for row in batch:
            gap = gaps[row["id"]]
            for rowid, _phrase, sent in offers[row["id"]]:
                sent_pairs.append((gap, sent))
                sent_owners.append((row["id"], rowid))
        sent_values = run_fair.batched_scores(reranker, sent_pairs)
        sent_scores: dict[str, dict[int, float]] = {}
        for (qid, rowid), value in zip(sent_owners, sent_values):
            sent_scores.setdefault(qid, {})[rowid] = value

        link_ids = [rowid for row in batch for rowid, _p, _s in offers[row["id"]]]
        missing = [rowid for rowid in dict.fromkeys(link_ids) if rowid not in pages]
        if missing:
            pages.update(db.fetch(missing))
        title_pairs = []
        full_pairs = []
        gap_full_pairs = []
        owners = []
        for row in batch:
            gap = gaps[row["id"]]
            for rowid, _phrase, _sent in offers[row["id"]]:
                page = pages.get(rowid)
                if page is None:
                    continue
                body = page.title + ". " + page.text
                title_pairs.append((gap, page.title))
                full_pairs.append((row["question"], body))
                gap_full_pairs.append((gap, body))
                owners.append((row["id"], rowid))
        title_values = run_fair.batched_scores(reranker, title_pairs)
        full_values = run_fair.batched_scores(reranker, full_pairs)
        gap_full_values = run_fair.batched_scores(reranker, gap_full_pairs)
        title_scores: dict[str, dict[int, float]] = {}
        full_scores: dict[str, dict[int, float]] = {}
        gap_scores: dict[str, dict[int, float]] = {}
        for (qid, rowid), title_value, full_value, gap_value in zip(owners, title_values, full_values, gap_full_values):
            title_scores.setdefault(qid, {})[rowid] = title_value
            full_scores.setdefault(qid, {})[rowid] = full_value
            gap_scores.setdefault(qid, {})[rowid] = gap_value

        for row, ids in zip(batch, bm25_lists):
            gold = set(row["gold_keys"])
            anchor = anchors[row["id"]]
            offer_ids = [rowid for rowid, _p, _s in offers[row["id"]]]
            packed = {
                "flat": run_fair.top_scored(ids, flat_scores.get(row["id"], {}), ore.BUDGET, set()),
                "ace": act.pack_ids(anchor, run_fair.top_scored(offer_ids, full_scores.get(row["id"], {}), ore.BUDGET, set(anchor)), pages),
                "gap_local": act.pack_ids(anchor, local_links(offers[row["id"]], sent_scores.get(row["id"], {}), ore.BUDGET), pages),
                "gap_sent": act.pack_ids(anchor, run_fair.top_scored(offer_ids, sent_scores.get(row["id"], {}), ore.BUDGET, set(anchor)), pages),
                "gap_abs": act.pack_ids(anchor, run_fair.top_scored(offer_ids, gap_scores.get(row["id"], {}), ore.BUDGET, set(anchor)), pages),
                "gap_title": act.pack_ids(anchor, run_fair.top_scored(offer_ids, title_scores.get(row["id"], {}), ore.BUDGET, set(anchor)), pages),
            }
            pool = {pages[rowid].key for rowid in ids if rowid in pages}
            outside = [key for key in gold if key not in pool]
            record = {
                "id": row["id"],
                "type": row["type"],
                "outside": len(outside),
                "gap": gaps[row["id"]],
                "gap_changed": int(gaps[row["id"]] != " ".join(ore.title_tokens(row["question"]))),
            }
            for name, chosen in packed.items():
                keys = {pages[rowid].key for rowid in chosen if rowid in pages}
                record[name] = act.support(keys, gold)
                record[name + "_hit"] = sum(1 for key in outside if key in keys)
                record[name + "_ids"] = chosen
            records.append(record)
        print(f"chain scored {start + len(batch)} sec {time.perf_counter() - t0:.1f}", flush=True)
    return records


def hotpot_rows(split: str) -> list[dict]:
    return run_fair.hotpot_rows(split)


def run_cal(limit: int) -> None:
    chosen = hotpot_rows("cal")
    if limit:
        chosen = chosen[:limit]
    db = ore.WikiDB()
    reranker = run_fair.load_reranker()
    try:
        records = evaluate(chosen, db, reranker)
    finally:
        db.close()
        del reranker
    scores = {name: act.mean(records, name) for name in ARMS}
    picked = choose(scores) if not limit else "unfrozen"
    payload = {
        "n": len(chosen),
        "limited": bool(limit),
        "min_gain": MIN_GAIN,
        "scores": scores,
        "policy": picked,
        "frozen_before_test": not bool(limit),
        "rule": (
            "Score links by the question tokens absent from the anchor titles. "
            "Among the opened sentence, the target title, and the target abstract, "
            "keep the cheapest preview that is within 0.01 of the best and beats "
            "scoring the full question by 0.01. Otherwise keep the full question."
        ),
        "gap_changed": sum(1 for row in records if row["gap_changed"]),
    }
    cal_path = ro.RESULT_DIR / ("chain_cal_debug.jsonl" if limit else "chain_cal_predictions.jsonl")
    with cal_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    path = ro.RESULT_DIR / "chain_cal_debug.json" if limit else RESULT
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


def run_test() -> None:
    if not RESULT.exists():
        raise SystemExit("chain_hparams.json is missing")
    hparams = json.loads(RESULT.read_text(encoding="utf-8"))
    if hparams.get("policy") not in POLICIES or not hparams.get("frozen_before_test"):
        raise SystemExit(f"refusing to score an unfrozen policy: {hparams.get('policy')}")
    chosen = hotpot_rows("test")
    pred_path = ro.RESULT_DIR / "chain_retrieve_predictions.jsonl"
    done = set()
    if pred_path.exists():
        with pred_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    done.add(json.loads(line)["id"])
    pending = [row for row in chosen if row["id"] not in done]
    print(f"chain test pending {len(pending)} done {len(done)} policy {hparams['policy']}", flush=True)
    db = ore.WikiDB()
    reranker = run_fair.load_reranker()
    try:
        with pred_path.open("a" if done else "w", encoding="utf-8") as handle:
            for start in range(0, len(pending), 40):
                batch = pending[start : start + 40]
                for row in evaluate(batch, db, reranker):
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
    finally:
        db.close()
        del reranker
    saved = [json.loads(line) for line in pred_path.open(encoding="utf-8") if line.strip()]
    summary = {
        "n": len(saved),
        "policy": hparams["policy"],
        "scores": {name: act.mean(saved, name) for name in ARMS},
        "frozen_policy": hparams["policy"],
    }
    (ro.RESULT_DIR / "chain_retrieve_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["cal", "test"], required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.stage == "cal":
        run_cal(args.limit)
        return
    if args.limit:
        raise SystemExit("the frozen rule is scored once, without a limit")
    run_test()


if __name__ == "__main__":
    main()
