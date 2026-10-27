"""Controls that separate the link set from the cross-encoder.

The observation is still the BM25 top 100. Flat ranking spends the
cross-encoder on those 100 abstracts and keeps four, with no hyperlinks.
Cheaper link rules may open a hyperlink without reading its abstract.
Calibration prefers the cheapest rule that stays within 0.01 of the best
rule and beats flat ranking by 0.01. The test split is scored once.
"""

from __future__ import annotations

import json
import os
import time

os.environ.setdefault("OPEN_CUDA", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse

import open_retrieve as ore
import run_action as act
import run_open as ro

RESULT = ro.RESULT_DIR / "fair_hparams.json"
MIN_GAIN = 0.01
ARMS = ("flat", "overlap", "surface", "title", "ace", "lexlink", "qtitle")
POLICIES = ("overlap", "surface", "title", "ace", "flat")


def batched_scores(reranker, pairs: list[tuple[str, str]]) -> list[float]:
    if not pairs:
        return []
    values = reranker.predict(pairs, batch_size=128, show_progress_bar=False)
    return [float(value) for value in values]


def link_offers(page: ore.Page, titles: dict[str, int], blocked: set[int]) -> list[tuple[int, str]]:
    out = []
    seen = set(blocked)
    for mention in ore.page_mentions(page, titles, "links"):
        if mention.rowid in seen:
            continue
        seen.add(mention.rowid)
        out.append((mention.rowid, mention.phrase))
    return out


def source_span(page: ore.Page, phrase: str) -> str:
    text = page.text or ""
    low = text.lower()
    needle = phrase.lower()
    at = low.find(needle) if needle else -1
    if at < 0:
        return phrase
    start = max(0, at - 80)
    end = min(len(text), at + len(phrase) + 80)
    return text[start:end].strip()


def top_scored(ids: list[int], scores: dict[int, float], k: int, blocked: set[int]) -> list[int]:
    order = sorted(ids, key=lambda rowid: scores.get(rowid, -1e9), reverse=True)
    out = []
    seen = set(blocked)
    for rowid in order:
        if rowid in seen:
            continue
        seen.add(rowid)
        out.append(rowid)
        if len(out) >= k:
            break
    return out


def question_titles(question: str, titles: dict[str, int]) -> list[int]:
    found = ore.maximal_title_matches(ore.title_tokens(question), titles)
    out = []
    seen = set()
    for mention in found:
        if mention.rowid in seen:
            continue
        seen.add(mention.rowid)
        out.append(mention.rowid)
    return out


def choose(scores: dict[str, float]) -> str:
    """Prefer the rule that reads the least of an unopened page."""
    best = max(scores[name] for name in ("overlap", "surface", "title", "ace"))
    if scores["overlap"] >= best - MIN_GAIN and scores["overlap"] >= scores["flat"] + MIN_GAIN:
        return "overlap"
    if scores["surface"] >= best - MIN_GAIN and scores["surface"] >= scores["flat"] + MIN_GAIN:
        return "surface"
    if scores["title"] >= best - MIN_GAIN and scores["title"] >= scores["ace"] - MIN_GAIN and scores["title"] >= scores["flat"] + MIN_GAIN:
        return "title"
    if scores["ace"] >= scores["flat"] + MIN_GAIN:
        return "ace"
    return "flat"


def evaluate(rows: list[dict], db: ore.WikiDB, reranker) -> list[dict]:
    records = []
    for start in range(0, len(rows), 40):
        batch = rows[start : start + 40]
        t0 = time.perf_counter()
        print(f"fair batch {start}:{start + len(batch)} / {len(rows)}", flush=True)
        bm25_lists = db.search_many([row["question"] for row in batch], ore.FIRST_HOP_K, workers=8)
        print("fair bm25 done", flush=True)
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
        flat_values = batched_scores(reranker, flat_pairs)
        flat_scores: dict[str, dict[int, float]] = {}
        for (qid, rowid), value in zip(flat_owners, flat_values):
            flat_scores.setdefault(qid, {})[rowid] = value

        offers: dict[str, list[tuple[int, str]]] = {}
        anchors: dict[str, list[int]] = {}
        for row, ids in zip(batch, bm25_lists):
            anchor = top_scored(ids, flat_scores.get(row["id"], {}), ore.ANCHOR_K, set())
            anchors[row["id"]] = anchor
            blocked = set(anchor)
            found: list[tuple[int, str]] = []
            for rowid in anchor:
                page = pages.get(rowid)
                if page is None:
                    continue
                found.extend(link_offers(page, db.titles, blocked))
                blocked.update(item[0] for item in found)
            offers[row["id"]] = found

        surface_pairs = []
        surface_owners = []
        for row in batch:
            anchor_pages = [pages[rowid] for rowid in anchors[row["id"]] if rowid in pages]
            for rowid, phrase in offers[row["id"]]:
                host = next((page for page in anchor_pages if phrase and phrase.lower() in (page.text or "").lower()), None)
                snippet = source_span(host, phrase) if host is not None else phrase
                surface_pairs.append((row["question"], snippet))
                surface_owners.append((row["id"], rowid))
        surface_values = batched_scores(reranker, surface_pairs)
        surface_scores: dict[str, dict[int, float]] = {}
        for (qid, rowid), value in zip(surface_owners, surface_values):
            surface_scores.setdefault(qid, {})[rowid] = value

        link_ids = [rowid for row in batch for rowid, _phrase in offers[row["id"]]]
        missing = [rowid for rowid in dict.fromkeys(link_ids) if rowid not in pages]
        if missing:
            pages.update(db.fetch(missing))
        title_pairs = []
        full_pairs = []
        link_owners = []
        for row in batch:
            for rowid, _phrase in offers[row["id"]]:
                page = pages.get(rowid)
                if page is None:
                    continue
                title_pairs.append((row["question"], page.title))
                full_pairs.append((row["question"], page.title + ". " + page.text))
                link_owners.append((row["id"], rowid))
        title_values = batched_scores(reranker, title_pairs)
        full_values = batched_scores(reranker, full_pairs)
        title_scores: dict[str, dict[int, float]] = {}
        full_scores: dict[str, dict[int, float]] = {}
        for (qid, rowid), title_value, full_value in zip(link_owners, title_values, full_values):
            title_scores.setdefault(qid, {})[rowid] = title_value
            full_scores.setdefault(qid, {})[rowid] = full_value
        q_wanted = [rowid for row in batch for rowid in question_titles(row["question"], db.titles)]
        q_missing = [rowid for rowid in dict.fromkeys(q_wanted) if rowid not in pages]
        if q_missing:
            pages.update(db.fetch(q_missing))

        for row, ids in zip(batch, bm25_lists):
            gold = set(row["gold_keys"])
            anchor = anchors[row["id"]]
            offer_ids = [rowid for rowid, _phrase in offers[row["id"]]]
            phrases = {rowid: phrase for rowid, phrase in offers[row["id"]]}
            overlap_scores = {
                rowid: float(len(set(ore.content_tokens(phrases.get(rowid, ""))) & set(ore.content_tokens(row["question"]))))
                for rowid in offer_ids
            }
            bm25_link = db.bm25_scores(row["question"], offer_ids)
            # FTS5 bm25 is negative for stronger hits. Flip it so larger is better.
            lexical_scores = {rowid: -bm25_link.get(rowid, 1e6) for rowid in offer_ids}
            flat_ids = top_scored(ids, flat_scores.get(row["id"], {}), ore.BUDGET, set())
            overlap_ids = top_scored(offer_ids, overlap_scores, ore.BUDGET, set(anchor))
            surface_ids = top_scored(offer_ids, surface_scores.get(row["id"], {}), ore.BUDGET, set(anchor))
            title_ids = top_scored(offer_ids, title_scores.get(row["id"], {}), ore.BUDGET, set(anchor))
            ace_ids = top_scored(offer_ids, full_scores.get(row["id"], {}), ore.BUDGET, set(anchor))
            lex_ids = top_scored(offer_ids, lexical_scores, ore.BUDGET, set(anchor))
            q_ids = question_titles(row["question"], db.titles)
            q_pack = act.pack_ids(q_ids, [rowid for rowid in ids if rowid not in set(q_ids)], pages)
            packed = {
                "flat": flat_ids,
                "overlap": act.pack_ids(anchor, overlap_ids, pages),
                "surface": act.pack_ids(anchor, surface_ids, pages),
                "title": act.pack_ids(anchor, title_ids, pages),
                "ace": act.pack_ids(anchor, ace_ids, pages),
                "lexlink": act.pack_ids(anchor, lex_ids, pages),
                "qtitle": q_pack,
            }
            pool = {pages[rowid].key for rowid in ids if rowid in pages}
            outside = [key for key in gold if key not in pool]
            record = {
                "id": row["id"],
                "type": row["type"],
                "outside": len(outside),
                "ce_abstracts": len(flat_scores.get(row["id"], {})) + len(full_scores.get(row["id"], {})),
                "ce_surface": len(surface_scores.get(row["id"], {})),
                "ce_titles": len(title_scores.get(row["id"], {})),
            }
            for name, chosen in packed.items():
                keys = {pages[rowid].key for rowid in chosen if rowid in pages}
                record[name] = act.support(keys, gold)
                record[name + "_hit"] = sum(1 for key in outside if key in keys)
                record[name + "_ids"] = chosen
            records.append(record)
        print(f"fair scored {start + len(batch)} sec {time.perf_counter() - t0:.1f}", flush=True)
    return records


def summarize(records: list[dict], policy: str) -> dict:
    scores = {name: act.mean(records, name) for name in ARMS}
    return {
        "n": len(records),
        "policy": policy,
        "scores": scores,
        "outside": sum(row["outside"] for row in records),
        "outside_hit": {name: sum(row[name + "_hit"] for row in records) for name in ARMS},
        "both": {name: sum(1 for row in records if row[name] == 1.0) for name in ARMS},
        "ce_abstracts": sum(row["ce_abstracts"] for row in records),
        "ce_surface": sum(row["ce_surface"] for row in records),
        "ce_titles": sum(row["ce_titles"] for row in records),
    }


def hotpot_rows(split: str) -> list[dict]:
    rows = ro.load_hotpot()
    cal_idx, _, _ = ro.split_indices(rows)
    blocked = {rows[i]["id"] for i in cal_idx}
    if split == "cal":
        return [rows[i] for i in cal_idx]
    return [row for row in rows if row["id"] not in blocked]


def load_reranker():
    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda:0", max_length=256)
    return reranker


def run_cal(limit: int) -> None:
    chosen = hotpot_rows("cal")
    if limit:
        chosen = chosen[:limit]
    db = ore.WikiDB()
    reranker = load_reranker()
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
            "Prefer overlap, then surface text, then the target title, then the target abstract, "
            "when that rule is within 0.01 of the best link rule and beats flat cross-encoder ranking by 0.01."
        ),
    }
    path = ro.RESULT_DIR / "fair_cal_debug.json" if limit else RESULT
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


def run_test() -> None:
    if not RESULT.exists():
        raise SystemExit("fair_hparams.json is missing")
    hparams = json.loads(RESULT.read_text(encoding="utf-8"))
    if hparams.get("policy") not in POLICIES or not hparams.get("frozen_before_test"):
        raise SystemExit(f"refusing to score an unfrozen policy: {hparams.get('policy')}")
    chosen = hotpot_rows("test")
    pred_path = ro.RESULT_DIR / "fair_retrieve_predictions.jsonl"
    done = set()
    if pred_path.exists():
        with pred_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    done.add(json.loads(line)["id"])
    pending = [row for row in chosen if row["id"] not in done]
    print(f"fair test pending {len(pending)} done {len(done)} policy {hparams['policy']}", flush=True)
    db = ore.WikiDB()
    reranker = load_reranker()
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
    saved = []
    with pred_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                saved.append(json.loads(line))
    summary = summarize(saved, hparams["policy"])
    summary["frozen_policy"] = hparams["policy"]
    (ro.RESULT_DIR / "fair_retrieve_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
