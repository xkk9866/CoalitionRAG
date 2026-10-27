"""Dossier-level anchor choice.

The greedy controller keeps the two best observed pages and then ranks their
links. Here the anchor pair is chosen for the dossier it makes legal: among the
cross-encoder's top M observed pages, every pair is scored by the summed
relevance probability of its best legal four-page dossier. M = 2 is the greedy
controller. The open variant also lets unused observed candidates compete for
the two remaining slots.

Calibration uses 200 questions that no earlier controller used. A variant is
adopted only when it beats the greedy controller by 0.01; otherwise the greedy
controller stays. The test split excludes the calibration questions and is
scored once.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time

os.environ.setdefault("OPEN_CUDA", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import open_retrieve as ore
import run_action as act
import run_bind
import run_fair
import run_open as ro

RESULT = ro.RESULT_DIR / "reach_hparams.json"
MIN_GAIN = 0.01
WIDTHS = (2, 4, 8)
VARIANTS = ("reach4", "reach4open", "reach8", "reach8open", "reach2open")
ARMS = ("flat", "ace") + VARIANTS
CAL_SEED = 37
PRIOR_CAL_SEEDS = (23, 29, 31)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def best_dossier(
    cand: list[int],
    width: int,
    links: dict[int, list[int]],
    prob: dict[int, float],
    open_pool: bool,
) -> tuple[list[int], float]:
    best_ids: list[int] = []
    best_value = -1.0
    pool = cand[:width]
    for a, b in itertools.combinations(pool, 2):
        anchor = [a, b]
        legal = []
        seen = {a, b}
        for host in anchor:
            for rowid in links.get(host, []):
                if rowid not in seen:
                    seen.add(rowid)
                    legal.append(rowid)
        if open_pool:
            for rowid in cand:
                if rowid not in seen:
                    seen.add(rowid)
                    legal.append(rowid)
        legal.sort(key=lambda rowid: prob.get(rowid, 0.0), reverse=True)
        fill = legal[: ore.BUDGET - 2]
        value = prob[a] + prob[b] + sum(prob.get(rowid, 0.0) for rowid in fill)
        if value > best_value + 1e-12:
            best_value = value
            best_ids = anchor + fill
    return best_ids, best_value


def evaluate(rows: list[dict], db: ore.WikiDB, reranker) -> list[dict]:
    records = []
    widest = max(WIDTHS)
    for start in range(0, len(rows), 40):
        batch = rows[start : start + 40]
        t0 = time.perf_counter()
        bm25_lists = db.search_many([row["question"] for row in batch], ore.FIRST_HOP_K, workers=8)
        pages = db.fetch([rowid for ids in bm25_lists for rowid in ids])
        pairs, owners = [], []
        for row, ids in zip(batch, bm25_lists):
            for rowid in ids:
                page = pages.get(rowid)
                if page is not None:
                    pairs.append((row["question"], page.title + ". " + page.text))
                    owners.append((row["id"], rowid))
        flat: dict[str, dict[int, float]] = {}
        for (qid, rowid), value in zip(owners, run_fair.batched_scores(reranker, pairs)):
            flat.setdefault(qid, {})[rowid] = value

        cands: dict[str, list[int]] = {}
        links: dict[str, dict[int, list[int]]] = {}
        for row, ids in zip(batch, bm25_lists):
            cand = run_fair.top_scored(ids, flat.get(row["id"], {}), widest, set())
            cands[row["id"]] = cand
            per = {}
            for rowid in cand:
                page = pages.get(rowid)
                per[rowid] = [] if page is None else [r for r, _ in run_fair.link_offers(page, db.titles, {rowid})]
            links[row["id"]] = per
        wanted = {r for per in links.values() for lst in per.values() for r in lst}
        missing = [r for r in wanted if r not in pages]
        if missing:
            pages.update(db.fetch(missing))
        pairs, owners = [], []
        for row in batch:
            need = []
            seen = set(flat.get(row["id"], {}))
            for lst in links[row["id"]].values():
                for rowid in lst:
                    if rowid not in seen and rowid in pages:
                        seen.add(rowid)
                        need.append(rowid)
            for rowid in need:
                page = pages[rowid]
                pairs.append((row["question"], page.title + ". " + page.text))
                owners.append((row["id"], rowid))
        link_scores: dict[str, dict[int, float]] = {}
        for (qid, rowid), value in zip(owners, run_fair.batched_scores(reranker, pairs)):
            link_scores.setdefault(qid, {})[rowid] = value

        for row, ids in zip(batch, bm25_lists):
            qid = row["id"]
            gold = set(row["gold_keys"])
            raw = dict(flat.get(qid, {}))
            raw.update(link_scores.get(qid, {}))
            prob = {rowid: sigmoid(value) for rowid, value in raw.items()}
            cand = cands[qid]
            per = {k: [r for r in v if r in pages] for k, v in links[qid].items()}
            packed = {"flat": run_fair.top_scored(ids, flat.get(qid, {}), ore.BUDGET, set())}
            anchor = cand[:2]
            offers = []
            seen = set(anchor)
            for host in anchor:
                for rowid in per.get(host, []):
                    if rowid not in seen:
                        seen.add(rowid)
                        offers.append(rowid)
            packed["ace"] = act.pack_ids(anchor, run_fair.top_scored(offers, prob, ore.BUDGET, set(anchor)), pages)
            values = {}
            for name in VARIANTS:
                width = int(name[5])
                chosen, value = best_dossier(cand, width, per, prob, name.endswith("open"))
                packed[name] = act.pack_ids(chosen[:2], chosen[2:], pages)
                values[name] = value
            pool = {pages[r].key for r in ids if r in pages}
            outside = [key for key in gold if key not in pool]
            record = {
                "id": qid,
                "type": row["type"],
                "outside": len(outside),
                "ce_abstracts": len(raw),
            }
            for name, chosen in packed.items():
                keys = {pages[r].key for r in chosen if r in pages}
                record[name] = act.support(keys, gold)
                record[name + "_hit"] = sum(1 for key in outside if key in keys)
                record[name + "_ids"] = chosen
            records.append(record)
        print(f"reach {start + len(batch)} / {len(rows)} sec {time.perf_counter() - t0:.1f}", flush=True)
    return records


def prior_cal_ids(rows: list[dict]) -> set[str]:
    blocked: set[str] = set()
    for seed in PRIOR_CAL_SEEDS:
        idx = run_bind.fresh_indices(rows, seed=seed, extra_ids=blocked)
        blocked.update(rows[i]["id"] for i in idx)
    return blocked


def cal_rows() -> list[dict]:
    rows = ro.load_hotpot()
    blocked = prior_cal_ids(rows)
    idx = run_bind.fresh_indices(rows, seed=CAL_SEED, extra_ids=blocked)
    return [rows[i] for i in idx]


def choose(scores: dict[str, float]) -> str:
    best = max(VARIANTS, key=lambda name: scores[name])
    if scores[best] >= scores["ace"] + MIN_GAIN:
        return best
    return "ace"


def run_cal() -> None:
    chosen = cal_rows()
    db = ore.WikiDB()
    reranker = run_fair.load_reranker()
    try:
        records = evaluate(chosen, db, reranker)
    finally:
        db.close()
    with (ro.RESULT_DIR / "reach_cal_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")
    scores = {name: act.mean(records, name) for name in ARMS}
    payload = {
        "n": len(records),
        "seed": CAL_SEED,
        "min_gain": MIN_GAIN,
        "scores": scores,
        "policy": choose(scores),
        "frozen_before_test": True,
        "ce_abstracts_mean": sum(r["ce_abstracts"] for r in records) / len(records),
        "rule": "Adopt the best dossier-level variant only if it beats greedy anchors by 0.01.",
    }
    RESULT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


def run_test() -> None:
    hparams = json.loads(RESULT.read_text(encoding="utf-8"))
    if hparams["policy"] not in VARIANTS or not hparams.get("frozen_before_test"):
        raise SystemExit(f"calibration kept {hparams['policy']}; nothing to test")
    blocked = {row["id"] for row in cal_rows()}
    chosen = [row for row in run_fair.hotpot_rows("test") if row["id"] not in blocked]
    path = ro.RESULT_DIR / "reach_test_predictions.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    pending = [row for row in chosen if row["id"] not in done]
    print(f"reach test pending {len(pending)} done {len(done)} policy {hparams['policy']}", flush=True)
    db = ore.WikiDB()
    reranker = run_fair.load_reranker()
    try:
        with path.open("a", encoding="utf-8") as handle:
            for start in range(0, len(pending), 40):
                for row in evaluate(pending[start : start + 40], db, reranker):
                    handle.write(json.dumps(row) + "\n")
                handle.flush()
    finally:
        db.close()
    saved = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = {
        "n": len(saved),
        "policy": hparams["policy"],
        "scores": {name: act.mean(saved, name) for name in ARMS},
        "both": {name: sum(1 for r in saved if r[name] == 1.0) for name in ARMS},
    }
    (ro.RESULT_DIR / "reach_test_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["cal", "test"], required=True)
    args = parser.parse_args()
    if args.stage == "cal":
        run_cal()
    else:
        run_test()


if __name__ == "__main__":
    main()
