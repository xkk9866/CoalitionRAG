"""One cross-encoder policy over admissible hyperlink actions.

The observation is the BM25 top 100. The anchor is chosen by the cross-encoder.
The next page must be a hyperlink in that anchor. A second page may be a
hyperlink in the page just chosen, and only if it outscores the next first-hop
link. Calibration adopts the policy when support rises by 0.01. Otherwise the
already tested link ranker stays in place.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("OPEN_CUDA", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse

import open_retrieve as ore
import run_action as act
import run_open as ro

RESULT = ro.RESULT_DIR / "policy_hparams.json"
MIN_GAIN = 0.01


def score_ids(reranker, question: str, ids: list[int], pages: dict[int, ore.Page]) -> dict[int, float]:
    valid = [rowid for rowid in ids if rowid in pages]
    if not valid:
        return {}
    pairs = [(question, pages[rowid].title + ". " + pages[rowid].text) for rowid in valid]
    values = reranker.predict(pairs, batch_size=64, show_progress_bar=False)
    return {rowid: float(value) for rowid, value in zip(valid, values)}


def top_ids(ids: list[int], scores: dict[int, float], k: int, blocked: set[int]) -> list[int]:
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


def evaluate(rows: list[dict], db: ore.WikiDB, encoder: ro.Encoder, reranker) -> list[dict]:
    records = []
    for start in range(0, len(rows), 40):
        batch = rows[start : start + 40]
        print(f"policy batch {start}:{start + len(batch)} / {len(rows)}", flush=True)
        bm25_lists = db.search_many([row["question"] for row in batch], ore.FIRST_HOP_K, workers=6)
        print("policy bm25 done", flush=True)
        needed = [rowid for ids in bm25_lists for rowid in ids]
        pages = db.fetch(needed)
        encoder.encode_pages(needed, pages)
        queries = encoder.encode_queries([(row["id"], row["question"]) for row in batch])
        dense_lists = [
            ore.order_by_dense(ids, queries[row["id"]], encoder.cache)
            for row, ids in zip(batch, bm25_lists)
        ]
        first_scores: dict[str, dict[int, float]] = {}
        for row, ids in zip(batch, bm25_lists):
            first_scores[row["id"]] = score_ids(reranker, row["question"], ids, pages)

        link_map: dict[str, list[int]] = {}
        ace_link_map: dict[str, list[int]] = {}
        extra: list[int] = []
        anchors_dense: dict[str, list[int]] = {}
        anchors_ce: dict[str, list[int]] = {}
        for row, ids, dense in zip(batch, bm25_lists, dense_lists):
            dense_anchor = [rowid for rowid in dense[: ore.ANCHOR_K] if rowid in pages]
            ce_anchor = top_ids(ids, first_scores[row["id"]], ore.ANCHOR_K, set())
            anchors_dense[row["id"]] = dense_anchor
            anchors_ce[row["id"]] = ce_anchor
            blocked_d = set(dense_anchor)
            blocked_c = set(ce_anchor)
            d_links: list[int] = []
            c_links: list[int] = []
            for rowid in dense_anchor:
                d_links.extend(act.link_rowids(pages[rowid], db.titles, blocked_d))
                blocked_d.update(d_links)
            for rowid in ce_anchor:
                page = pages.get(rowid)
                if page is None:
                    continue
                c_links.extend(act.link_rowids(page, db.titles, blocked_c))
                blocked_c.update(c_links)
            link_map[row["id"]] = d_links
            ace_link_map[row["id"]] = c_links
            extra.extend(d_links)
            extra.extend(c_links)
        missing = [rowid for rowid in dict.fromkeys(extra) if rowid not in pages]
        if missing:
            pages.update(db.fetch(missing))
        link_scores: dict[str, dict[int, float]] = {}
        for row in batch:
            ids = list(dict.fromkeys(link_map[row["id"]] + ace_link_map[row["id"]]))
            link_scores[row["id"]] = score_ids(reranker, row["question"], ids, pages)

        hop2_map: dict[str, list[int]] = {}
        extra2: list[int] = []
        for row in batch:
            ranked = top_ids(ace_link_map[row["id"]], link_scores[row["id"]], 1, set(anchors_ce[row["id"]]))
            blocked = set(anchors_ce[row["id"]]) | set(ace_link_map[row["id"]])
            second: list[int] = []
            if ranked and ranked[0] in pages:
                second = act.link_rowids(pages[ranked[0]], db.titles, blocked)
            hop2_map[row["id"]] = second
            extra2.extend(second)
        missing2 = [rowid for rowid in dict.fromkeys(extra2) if rowid not in pages]
        if missing2:
            pages.update(db.fetch(missing2))
        hop2_scores: dict[str, dict[int, float]] = {}
        for row in batch:
            hop2_scores[row["id"]] = score_ids(reranker, row["question"], hop2_map[row["id"]], pages)
        print(f"policy scored {start + len(batch)}", flush=True)

        for row, dense in zip(batch, dense_lists):
            gold = set(row["gold_keys"])
            dense_anchor = anchors_dense[row["id"]]
            ce_anchor = anchors_ce[row["id"]]
            dense_keys = {pages[rowid].key for rowid in dense[: ore.BUDGET] if rowid in pages}
            ce_order = top_ids(link_map[row["id"]], link_scores[row["id"]], ore.BUDGET, set(dense_anchor))
            ace_order = top_ids(ace_link_map[row["id"]], link_scores[row["id"]], ore.BUDGET, set(ce_anchor))
            locked = top_ids(ace_link_map[row["id"]], link_scores[row["id"]], 1, set(ce_anchor))
            rest = top_ids(ace_link_map[row["id"]], link_scores[row["id"]], 1, set(ce_anchor) | set(locked))
            hop2_best = top_ids(hop2_map[row["id"]], hop2_scores[row["id"]], 1, set(ce_anchor) | set(locked))
            seq = list(ce_anchor)
            if locked:
                seq.append(locked[0])
            if hop2_best and rest:
                if hop2_scores[row["id"]].get(hop2_best[0], -1e9) > link_scores[row["id"]].get(rest[0], -1e9):
                    seq.append(hop2_best[0])
                else:
                    seq.append(rest[0])
            elif hop2_best:
                seq.append(hop2_best[0])
            elif rest:
                seq.append(rest[0])
            seq = act.pack_ids(seq, top_ids(dense, first_scores[row["id"]], ore.BUDGET, set(seq)), pages)
            records.append(
                {
                    "id": row["id"],
                    "dense": act.support(dense_keys, gold),
                    "ce": act.support(act.pack_keys(dense_anchor, ce_order, pages), gold),
                    "ace": act.support(act.pack_keys(ce_anchor, ace_order, pages), gold),
                    "seq": act.support({pages[rowid].key for rowid in seq if rowid in pages}, gold),
                }
            )
        encoder.cache.clear()
    return records


def choose(scores: dict[str, float]) -> str:
    if scores["seq"] >= scores["ace"] + MIN_GAIN and scores["seq"] >= scores["ce"] + MIN_GAIN:
        return "seq"
    if scores["ace"] >= scores["ce"] + MIN_GAIN:
        return "ace"
    return "ce"


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
        records = evaluate(chosen, db, encoder, reranker)
    finally:
        encoder.close()
        db.close()
        del reranker
    scores = {name: act.mean(records, name) for name in ("dense", "ce", "ace", "seq")}
    picked = choose(scores) if not limit else "unfrozen"
    payload = {
        "n": len(chosen),
        "limited": bool(limit),
        "min_gain": MIN_GAIN,
        "scores": scores,
        "policy": picked,
        "frozen_before_test": not bool(limit),
        "reference_ce": 0.8325,
    }
    path = ro.RESULT_DIR / "policy_cal_debug.json" if limit else RESULT
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(scores, indent=2), flush=True)
    print(f"frozen policy {picked}", flush=True)


def evaluate_ace(rows: list[dict], db: ore.WikiDB, reranker) -> list[dict]:
    """Score only the frozen ace policy: cross-encoder anchor, then its links."""
    records = []
    for start in range(0, len(rows), 40):
        batch = rows[start : start + 40]
        print(f"ace batch {start}:{start + len(batch)} / {len(rows)}", flush=True)
        bm25_lists = db.search_many([row["question"] for row in batch], ore.FIRST_HOP_K, workers=8)
        print("ace bm25 done", flush=True)
        needed = [rowid for ids in bm25_lists for rowid in ids]
        pages = db.fetch(needed)
        first_scores: dict[str, dict[int, float]] = {}
        for row, ids in zip(batch, bm25_lists):
            first_scores[row["id"]] = score_ids(reranker, row["question"], ids, pages)

        ace_link_map: dict[str, list[int]] = {}
        anchors_ce: dict[str, list[int]] = {}
        extra: list[int] = []
        for row, ids in zip(batch, bm25_lists):
            ce_anchor = top_ids(ids, first_scores[row["id"]], ore.ANCHOR_K, set())
            anchors_ce[row["id"]] = ce_anchor
            blocked = set(ce_anchor)
            links: list[int] = []
            for rowid in ce_anchor:
                page = pages.get(rowid)
                if page is None:
                    continue
                links.extend(act.link_rowids(page, db.titles, blocked))
                blocked.update(links)
            ace_link_map[row["id"]] = links
            extra.extend(links)
        missing = [rowid for rowid in dict.fromkeys(extra) if rowid not in pages]
        if missing:
            pages.update(db.fetch(missing))

        for row, ids in zip(batch, bm25_lists):
            gold = set(row["gold_keys"])
            ce_anchor = anchors_ce[row["id"]]
            link_scores = score_ids(reranker, row["question"], ace_link_map[row["id"]], pages)
            ace_order = top_ids(ace_link_map[row["id"]], link_scores, ore.BUDGET, set(ce_anchor))
            chosen = act.pack_ids(ce_anchor, ace_order, pages)
            keys = {pages[rowid].key for rowid in chosen if rowid in pages}
            pool = {pages[rowid].key for rowid in ids if rowid in pages}
            outside = [key for key in gold if key not in pool]
            records.append(
                {
                    "id": row["id"],
                    "type": row["type"],
                    "ace": act.support(keys, gold),
                    "outside": len(outside),
                    "outside_hit": sum(1 for key in outside if key in keys),
                    "rowids": chosen,
                    "titles": [pages[rowid].title for rowid in chosen],
                    "keys": [pages[rowid].key for rowid in chosen],
                }
            )
        print(f"ace scored {start + len(batch)}", flush=True)
    return records


def _paired(left: list[float], right: list[float]) -> dict:
    import numpy as np

    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
    return {"n": int(len(diff)), "delta": float(diff.mean()) if len(diff) else 0.0, "se": se}


def run_test() -> None:
    if not RESULT.exists():
        raise SystemExit("policy_hparams.json is missing")
    hparams = json.loads(RESULT.read_text(encoding="utf-8"))
    if hparams.get("policy") != "ace" or not hparams.get("frozen_before_test"):
        raise SystemExit(f"refusing to score an unfrozen policy: {hparams.get('policy')}")
    rows = ro.load_hotpot()
    cal_idx, _, _ = ro.split_indices(rows)
    blocked = {rows[i]["id"] for i in cal_idx}
    chosen = [row for row in rows if row["id"] not in blocked]
    pred_path = ro.RESULT_DIR / "policy_retrieve_predictions.jsonl"
    done = set()
    if pred_path.exists():
        with pred_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    done.add(json.loads(line)["id"])
    pending = [row for row in chosen if row["id"] not in done]
    print(f"ace test pending {len(pending)} done {len(done)}", flush=True)
    db = ore.WikiDB()
    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda:0", max_length=256)
    try:
        with pred_path.open("a" if done else "w", encoding="utf-8") as handle:
            for start in range(0, len(pending), 40):
                batch = pending[start : start + 40]
                records = evaluate_ace(batch, db, reranker)
                for row in records:
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
    base = {}
    with (ro.RESULT_DIR / "action_retrieve_predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                base[row["id"]] = row
    paired = [row for row in saved if row["id"] in base]
    by_type: dict[str, list[float]] = {}
    for row in saved:
        by_type.setdefault(row["type"], []).append(row["ace"])
    summary = {
        "n": len(saved),
        "policy": "ace",
        "ace": act.mean(saved, "ace"),
        "both": sum(1 for row in saved if row["ace"] == 1.0),
        "outside": sum(row["outside"] for row in saved),
        "outside_hit": sum(row["outside_hit"] for row in saved),
        "ace_vs_dense": _paired([row["ace"] for row in paired], [base[row["id"]]["dense"] for row in paired]),
        "ace_vs_ce": _paired([row["ace"] for row in paired], [base[row["id"]]["ce"] for row in paired]),
        "by_type": {name: sum(values) / len(values) for name, values in sorted(by_type.items())},
    }
    out = ro.RESULT_DIR / "policy_retrieve_summary.json"
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
    if args.limit:
        raise SystemExit("the frozen policy is scored once, without a limit")
    run_test()


if __name__ == "__main__":
    main()
