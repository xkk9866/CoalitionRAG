"""Bind a proper name that is not a Wikipedia title, then follow that page's links.

The cross-encoder and the four-page budget stay as they are. The only new
action is the binding: the longest capitalized span that is not itself a
title is looked up as a phrase, and the article that states the phrase is
opened. Its hyperlinks are scored with the original question. A previous
calibration already rejected deleting the anchor's tokens from the question.
This comparison uses a different 200 questions. Binding is kept only when
support rises by 0.01.
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
from collections import defaultdict

import numpy as np

import open_retrieve as ore
import run_action as act
import run_fair
import run_open as ro

RESULT = ro.RESULT_DIR / "bind_hparams.json"
MIN_GAIN = 0.01
ARMS = ("flat", "ace", "bind")
WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
SKIP = {"what", "when", "where", "who", "which", "why", "how"}


def unbound_names(question: str, titles: dict[str, int]) -> list[str]:
    spans: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if len(buf) >= 2:
            spans.append(" ".join(buf))
        buf.clear()

    for tok in WORD.findall(question):
        if tok[0].isupper() and tok.lower() not in SKIP:
            buf.append(tok)
        else:
            flush()
    flush()
    names = []
    for span in spans:
        key = ore.title_key(span)
        if key and key not in titles and not ore.is_disambiguation(key):
            names.append(span)
    names.sort(key=len, reverse=True)
    return names


def phrase_hit(db: ore.WikiDB, span: str) -> int | None:
    toks = [tok for tok in ore.title_tokens(span) if tok.isalnum()]
    if len(toks) < 2:
        return None
    query = '"' + " ".join(toks) + '"'
    try:
        rows = db.con.execute(
            "SELECT rowid FROM pages WHERE pages MATCH ? ORDER BY rank LIMIT 1",
            (query,),
        ).fetchall()
    except Exception:
        return None
    if not rows:
        return None
    return int(rows[0]["rowid"])


def bind_page(db: ore.WikiDB, question: str) -> tuple[int | None, str]:
    for span in unbound_names(question, db.titles):
        rowid = phrase_hit(db, span)
        if rowid is None:
            continue
        page = db.fetch([rowid]).get(rowid)
        if page is None or ore.is_disambiguation(page.key):
            continue
        return rowid, span
    return None, ""


def fresh_indices(rows: list[dict], seed: int = 23, extra_ids: set[str] | None = None) -> list[int]:
    """200 questions that were not used to choose or answer the earlier controller."""
    cal, test, second = ro.split_indices(rows)
    blocked = set(cal + test + second)
    if extra_ids:
        blocked.update(i for i, row in enumerate(rows) if row["id"] in extra_ids)
    types = [row["type"] for row in rows]
    rng = np.random.default_rng(seed)
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, kind in enumerate(types):
        if i not in blocked:
            buckets[kind].append(i)
    total = sum(len(v) for v in buckets.values())
    chosen: list[int] = []
    for kind in sorted(buckets):
        idxs = buckets[kind][:]
        rng.shuffle(idxs)
        nt = int(round(200 * len(idxs) / total))
        chosen.extend(idxs[:nt])
    rng.shuffle(chosen)
    return chosen[:200]


def evaluate(rows: list[dict], db: ore.WikiDB, reranker, finder=bind_page) -> list[dict]:
    records = []
    for start in range(0, len(rows), 40):
        batch = rows[start : start + 40]
        t0 = time.perf_counter()
        print(f"bind batch {start}:{start + len(batch)} / {len(rows)}", flush=True)
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

        bindings: dict[str, tuple[int | None, str]] = {}
        for row in batch:
            bindings[row["id"]] = finder(db, row["question"])
        extra = [rowid for rowid, _span in bindings.values() if rowid is not None and rowid not in pages]
        if extra:
            pages.update(db.fetch(extra))

        offers: dict[str, list[int]] = {}
        anchors: dict[str, list[int]] = {}
        for row, ids in zip(batch, bm25_lists):
            anchor = run_fair.top_scored(ids, flat_scores.get(row["id"], {}), ore.ANCHOR_K, set())
            anchors[row["id"]] = anchor
            bound, _span = bindings[row["id"]]
            sources = []
            if bound is not None:
                sources.append(bound)
            sources.extend(anchor)
            blocked = set()
            found = []
            for rowid in sources:
                page = pages.get(rowid)
                if page is None:
                    continue
                for link_id, _phrase in run_fair.link_offers(page, db.titles, blocked):
                    found.append(link_id)
                    blocked.add(link_id)
            offers[row["id"]] = found

        link_ids = [rowid for row in batch for rowid in offers[row["id"]]]
        missing = [rowid for rowid in dict.fromkeys(link_ids) if rowid not in pages]
        if missing:
            pages.update(db.fetch(missing))
        pairs = []
        owners = []
        for row in batch:
            for rowid in offers[row["id"]]:
                page = pages.get(rowid)
                if page is None:
                    continue
                pairs.append((row["question"], page.title + ". " + page.text))
                owners.append((row["id"], rowid))
        values = run_fair.batched_scores(reranker, pairs)
        link_scores: dict[str, dict[int, float]] = {}
        for (qid, rowid), value in zip(owners, values):
            link_scores.setdefault(qid, {})[rowid] = value

        for row, ids in zip(batch, bm25_lists):
            gold = set(row["gold_keys"])
            anchor = anchors[row["id"]]
            bound, span = bindings[row["id"]]
            sources = ([bound] if bound is not None else []) + anchor
            offer_ids = offers[row["id"]]
            ace_links = []
            blocked = set(anchor)
            for rowid in anchor:
                page = pages.get(rowid)
                if page is None:
                    continue
                for link_id, _phrase in run_fair.link_offers(page, db.titles, blocked):
                    ace_links.append(link_id)
                    blocked.add(link_id)
            packed = {
                "flat": run_fair.top_scored(ids, flat_scores.get(row["id"], {}), ore.BUDGET, set()),
                "ace": act.pack_ids(anchor, run_fair.top_scored(ace_links, link_scores.get(row["id"], {}), ore.BUDGET, set(anchor)), pages),
                "bind": act.pack_ids(sources, run_fair.top_scored(offer_ids, link_scores.get(row["id"], {}), ore.BUDGET, set(sources)), pages),
            }
            record = {"id": row["id"], "type": row["type"], "bound": span}
            for name, chosen in packed.items():
                keys = {pages[rowid].key for rowid in chosen if rowid in pages}
                record[name] = act.support(keys, gold)
                record[name + "_ids"] = chosen
            records.append(record)
        print(f"bind scored {start + len(batch)} sec {time.perf_counter() - t0:.1f}", flush=True)
    return records


def run_cal() -> None:
    rows = ro.load_hotpot()
    db = ore.WikiDB()
    demo = unbound_names("What was the third studio album released by Richard Melville Hall?", db.titles)
    if not demo or demo[0] != "Richard Melville Hall":
        db.close()
        raise SystemExit(f"name binding failed the Moby check: {demo}")
    chosen_idx = fresh_indices(rows)
    chosen = [rows[i] for i in chosen_idx]
    reranker = run_fair.load_reranker()
    try:
        records = evaluate(chosen, db, reranker)
    finally:
        db.close()
        del reranker
    scores = {name: act.mean(records, name) for name in ARMS}
    policy = "bind" if scores["bind"] >= scores["ace"] + MIN_GAIN else "ace"
    by_type = {}
    for kind in sorted({row["type"] for row in records}):
        subset = [row for row in records if row["type"] == kind]
        by_type[kind] = {"n": len(subset), **{name: act.mean(subset, name) for name in ARMS}}
    payload = {
        "n": len(chosen),
        "seed": 23,
        "excluded": "seed-7 calibration 200, seed-7 answer 300, seed-11 second 300",
        "cal_ids": [row["id"] for row in chosen],
        "min_gain": MIN_GAIN,
        "scores": scores,
        "by_type": by_type,
        "bound_found": sum(1 for row in records if row["bound"]),
        "policy": policy,
        "frozen_before_test": True,
        "rule": (
            "Open the article that states the longest capitalized span which is not a title. "
            "Score that article's hyperlinks with the original question. "
            "Keep binding only when support exceeds the unaltered controller by 0.01."
        ),
    }
    RESULT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pred = ro.RESULT_DIR / "bind_cal_predictions.jsonl"
    with pred.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({k: payload[k] for k in payload if k != "cal_ids"}, indent=2), flush=True)


def run_test() -> None:
    if not RESULT.exists():
        raise SystemExit("bind_hparams.json is missing")
    hparams = json.loads(RESULT.read_text(encoding="utf-8"))
    if hparams.get("policy") != "bind" or not hparams.get("frozen_before_test"):
        raise SystemExit(f"refusing to score a policy that did not clear calibration: {hparams.get('policy')}")
    blocked = set(hparams["cal_ids"])
    chosen = [row for row in run_fair.hotpot_rows("test") if row["id"] not in blocked]
    pred_path = ro.RESULT_DIR / "bind_retrieve_predictions.jsonl"
    done = set()
    if pred_path.exists():
        with pred_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    done.add(json.loads(line)["id"])
    pending = [row for row in chosen if row["id"] not in done]
    print(f"bind test pending {len(pending)} done {len(done)}", flush=True)
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
        "policy": "bind",
        "scores": {name: act.mean(saved, name) for name in ARMS},
        "frozen_policy": "bind",
    }
    (ro.RESULT_DIR / "bind_retrieve_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["cal", "test"], required=True)
    args = parser.parse_args()
    if args.stage == "cal":
        run_cal()
        return
    run_test()


if __name__ == "__main__":
    main()
