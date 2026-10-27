"""Rank admissible hyperlink actions. The ranker is frozen on calibration.

An action is admissible only when the current page contains a hyperlink that
resolves to a corpus document. Calibration chooses the ranker. The test split
is scored once, after that file exists.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("OPEN_CUDA", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
from pathlib import Path

import open_retrieve as ore
import run_open as ro

RESULT = ro.RESULT_DIR / "action_hparams.json"
MIN_GAIN = 0.01
HOP2_KEEP = 3


def support(keys: set[str], gold: set[str]) -> float:
    if not gold:
        return 0.0
    return len(keys & gold) / len(gold)


def link_rowids(page: ore.Page, titles: dict[str, int], blocked: set[int]) -> list[int]:
    out = []
    seen = set(blocked)
    for mention in ore.page_mentions(page, titles, "links"):
        if mention.rowid in seen:
            continue
        seen.add(mention.rowid)
        out.append(mention.rowid)
    return out


def pack_ids(anchor: list[int], ranked: list[int], pages: dict[int, ore.Page]) -> list[int]:
    chosen = []
    seen = set()
    for rowid in anchor + ranked:
        if rowid in seen or rowid not in pages:
            continue
        seen.add(rowid)
        chosen.append(rowid)
        if len(chosen) >= ore.BUDGET:
            break
    return chosen


def pack_keys(anchor: list[int], ranked: list[int], pages: dict[int, ore.Page]) -> set[str]:
    return {pages[rowid].key for rowid in pack_ids(anchor, ranked, pages)}


def mean(rows: list[dict], key: str) -> float:
    return sum(row[key] for row in rows) / len(rows)


def choose(scores: dict[str, float]) -> str:
    """Adopt a heavier ranker only when calibration support rises by MIN_GAIN."""
    if scores["ce2"] >= scores["ce"] + MIN_GAIN and scores["ce2"] >= scores["term"] + MIN_GAIN:
        return "ce2"
    if scores["ce"] >= scores["term"] + MIN_GAIN and scores["ce"] >= scores["dense"] + MIN_GAIN:
        return "ce"
    if scores["dense"] >= scores["term"] + MIN_GAIN:
        return "dense"
    return "term"


def evaluate(rows: list[dict], db: ore.WikiDB, encoder: ro.Encoder, reranker, hop2: bool) -> list[dict]:
    records = []
    for start in range(0, len(rows), 50):
        batch = rows[start : start + 50]
        print(f"action batch {start}:{start + len(batch)} / {len(rows)}", flush=True)
        bm25_lists = db.search_many([row["question"] for row in batch], ore.FIRST_HOP_K, workers=6)
        print("action bm25 done", flush=True)
        needed = [rowid for ids in bm25_lists for rowid in ids]
        pages = db.fetch(needed)
        encoder.encode_pages(needed, pages)
        queries = encoder.encode_queries([(row["id"], row["question"]) for row in batch])
        dense_lists = [
            ore.order_by_dense(ids, queries[row["id"]], encoder.cache)
            for row, ids in zip(batch, bm25_lists)
        ]
        hop1_map: dict[str, list[int]] = {}
        extra: list[int] = []
        anchors: dict[str, list[int]] = {}
        for row, dense in zip(batch, dense_lists):
            anchor = [rowid for rowid in dense[: ore.ANCHOR_K] if rowid in pages]
            anchors[row["id"]] = anchor
            ids = []
            blocked = set(anchor)
            for rowid in anchor:
                ids.extend(link_rowids(pages[rowid], db.titles, blocked))
                blocked.update(ids)
            hop1_map[row["id"]] = ids
            extra.extend(ids)
        missing = [rowid for rowid in dict.fromkeys(extra) if rowid not in pages]
        if missing:
            pages.update(db.fetch(missing))
            encoder.encode_pages(missing, pages)

        hop2_map: dict[str, list[int]] = {}
        extra2: list[int] = []
        if hop2:
            for row in batch:
                qvec = queries[row["id"]]
                ranked_hop = sorted(
                    [rowid for rowid in hop1_map[row["id"]] if rowid in encoder.cache],
                    key=lambda rowid: float(qvec @ encoder.cache[rowid]),
                    reverse=True,
                )
                blocked = set(anchors[row["id"]]) | set(hop1_map[row["id"]])
                second = []
                for rowid in ranked_hop[:HOP2_KEEP]:
                    page = pages.get(rowid)
                    if page is None:
                        continue
                    second.extend(link_rowids(page, db.titles, blocked))
                    blocked.update(second)
                hop2_map[row["id"]] = second
                extra2.extend(second)
        else:
            for row in batch:
                hop2_map[row["id"]] = []
        missing2 = [rowid for rowid in dict.fromkeys(extra2) if rowid not in pages]
        if missing2:
            pages.update(db.fetch(missing2))

        print(f"action candidates hop2 {len(extra2)}", flush=True)
        pairs = []
        owners = []
        for row in batch:
            cands = hop1_map[row["id"]] + hop2_map[row["id"]]
            for rowid in cands:
                page = pages.get(rowid)
                if page is None:
                    continue
                pairs.append((row["question"], page.title + ". " + page.text))
                owners.append((row["id"], rowid))
        ce_scores: dict[tuple[str, int], float] = {}
        print(f"action cross-encode pairs {len(pairs)}", flush=True)
        if pairs:
            values = reranker.predict(pairs, batch_size=32, show_progress_bar=False)
            for (qid, rowid), value in zip(owners, values):
                ce_scores[(qid, rowid)] = float(value)

        for row, dense in zip(batch, dense_lists):
            gold = set(row["gold_keys"])
            anchor = anchors[row["id"]]
            qvec = queries[row["id"]]
            hop1 = [rowid for rowid in hop1_map[row["id"]] if rowid in pages]
            hop2_ids = [rowid for rowid in hop2_map[row["id"]] if rowid in pages]
            dense_keys = {pages[rowid].key for rowid in dense[: ore.BUDGET] if rowid in pages}
            kept = [pages[rowid] for rowid in anchor if rowid in pages]

            def dense_key(rowid: int) -> float:
                vec = encoder.cache.get(rowid)
                return float(qvec @ vec) if vec is not None else -1.0

            def term_key(rowid: int):
                page = pages[rowid]
                return (ore.new_term_count(row["question"], kept, page), dense_key(rowid))

            term_order = sorted(hop1, key=term_key, reverse=True)
            dense_order = sorted(hop1, key=dense_key, reverse=True)
            ce_order = sorted(hop1, key=lambda rowid: ce_scores.get((row["id"], rowid), -1e9), reverse=True)
            both = hop1 + hop2_ids
            ce2_order = sorted(both, key=lambda rowid: ce_scores.get((row["id"], rowid), -1e9), reverse=True)
            gold_pool = {pages[rowid].key for rowid in both}
            chosen_ids = pack_ids(anchor, ce_order, pages)
            ce_keys = {pages[rowid].key for rowid in chosen_ids}
            pool_keys = {pages[rowid].key for rowid in dense if rowid in pages}
            outside = [key for key in gold if key not in pool_keys]
            records.append(
                {
                    "id": row["id"],
                    "type": row["type"],
                    "dense": support(dense_keys, gold),
                    "term": support(pack_keys(anchor, term_order, pages), gold),
                    "dense_rank": support(pack_keys(anchor, dense_order, pages), gold),
                    "ce": support(ce_keys, gold),
                    "ce2": support(pack_keys(anchor, ce2_order, pages), gold) if hop2_ids else None,
                    "oracle": support(pack_keys(anchor, [rowid for rowid in both if pages[rowid].key in gold], pages), gold),
                    "pool": support(dense_keys | gold_pool, gold),
                    "outside": len(outside),
                    "outside_hit": sum(1 for key in outside if key in ce_keys),
                    "rowids": chosen_ids,
                    "titles": [pages[rowid].title for rowid in chosen_ids],
                    "keys": [pages[rowid].key for rowid in chosen_ids],
                }
            )
        encoder.cache.clear()
        print(f"scored {start + len(batch)}", flush=True)
    return records


def run_cal(limit: int) -> None:
    rows = ro.load_hotpot()
    cal_idx, _, _ = ro.split_indices(rows)
    chosen = [rows[i] for i in cal_idx]
    if limit:
        chosen = chosen[:limit]
    db = ore.WikiDB()
    encoder = ro.Encoder()
    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda:0", max_length=256)
    try:
        records = evaluate(chosen, db, encoder, reranker, hop2=True)
    finally:
        encoder.close()
        db.close()
        del reranker
    scores = {
        "dense": mean(records, "dense"),
        "term": mean(records, "term"),
        "dense_rank": mean(records, "dense_rank"),
        "ce": mean(records, "ce"),
        "ce2": mean(records, "ce2"),
        "oracle": mean(records, "oracle"),
        "pool": mean(records, "pool"),
    }
    # The selector reads "dense" as the dense ranking of hyperlinks.
    decision_scores = {
        "term": scores["term"],
        "dense": scores["dense_rank"],
        "ce": scores["ce"],
        "ce2": scores["ce2"],
    }
    picked = choose(decision_scores) if not limit else "unfrozen"
    payload = {
        "n": len(chosen),
        "limited": bool(limit),
        "min_gain": MIN_GAIN,
        "hop2_keep": HOP2_KEEP,
        "scores": scores,
        "ranker": picked,
        "frozen_before_test": not bool(limit),
        "rule": (
            "Lock the two highest dense pages in the BM25 top 100. "
            "Admissible actions are hyperlinks in those pages"
            + (" and hyperlinks in the three highest-scoring linked pages." if picked == "ce2" else ".")
            + " Rank those corpus pages with the frozen ranker and keep two."
        ),
    }
    if limit:
        path = ro.RESULT_DIR / "action_cal_debug.json"
    else:
        path = RESULT
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(scores, indent=2), flush=True)
    print(f"frozen ranker {picked}", flush=True)


def _paired(left: list[float], right: list[float]) -> dict:
    import numpy as np

    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
    return {"n": int(len(diff)), "delta": float(diff.mean()) if len(diff) else 0.0, "se": se}


def run_test() -> None:
    if not RESULT.exists():
        raise SystemExit("action_hparams.json is missing")
    hparams = json.loads(RESULT.read_text(encoding="utf-8"))
    if hparams.get("ranker") != "ce" or not hparams.get("frozen_before_test"):
        raise SystemExit(f"refusing to score an unfrozen ranker: {hparams.get('ranker')}")
    rows = ro.load_hotpot()
    cal_idx, _, _ = ro.split_indices(rows)
    blocked = {rows[i]["id"] for i in cal_idx}
    chosen = [row for row in rows if row["id"] not in blocked]
    pred_path = ro.RESULT_DIR / "action_retrieve_predictions.jsonl"
    done = set()
    if pred_path.exists():
        with pred_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    done.add(json.loads(line)["id"])
    pending = [row for row in chosen if row["id"] not in done]
    print(f"action test pending {len(pending)} done {len(done)}", flush=True)
    db = ore.WikiDB()
    encoder = ro.Encoder()
    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda:0", max_length=256)
    try:
        with pred_path.open("a" if done else "w", encoding="utf-8") as handle:
            for start in range(0, len(pending), 50):
                batch = pending[start : start + 50]
                records = evaluate(batch, db, encoder, reranker, hop2=False)
                for row in records:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
    finally:
        encoder.close()
        db.close()
        del reranker
    saved = []
    with pred_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                saved.append(json.loads(line))
    summary = {
        "n": len(saved),
        "ranker": "ce",
        "dense": mean(saved, "dense"),
        "term": mean(saved, "term"),
        "dense_rank": mean(saved, "dense_rank"),
        "ce": mean(saved, "ce"),
        "outside": sum(row["outside"] for row in saved),
        "outside_hit": sum(row["outside_hit"] for row in saved),
        "ce_vs_dense": _paired([row["ce"] for row in saved], [row["dense"] for row in saved]),
        "ce_vs_term": _paired([row["ce"] for row in saved], [row["term"] for row in saved]),
        "term_vs_dense": _paired([row["term"] for row in saved], [row["dense"] for row in saved]),
    }
    out = ro.RESULT_DIR / "action_retrieve_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["cal", "test"], required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.stage == "cal":
        run_cal(args.limit)
        return
    run_test()


if __name__ == "__main__":
    main()
