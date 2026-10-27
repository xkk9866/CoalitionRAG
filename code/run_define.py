"""Open the one article that states a question phrase nobody else states.

Name binding on a disjoint 200 tied the existing controller, because a
capitalized phrase often occurs in many articles. Here the phrase must match
exactly one article. That article is opened, and its hyperlinks are scored
with the original question. The comparison uses another 200 questions.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("OPEN_CUDA", "-1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse

import open_retrieve as ore
import run_action as act
import run_bind
import run_fair
import run_open as ro

RESULT = ro.RESULT_DIR / "define_hparams.json"
MIN_GAIN = 0.01


def define_page(db: ore.WikiDB, question: str) -> tuple[int | None, str]:
    tokens = ore.title_tokens(question)
    for n in range(min(6, len(tokens)), 1, -1):
        for i in range(0, len(tokens) - n + 1):
            span_toks = tokens[i : i + n]
            long_toks = [tok for tok in span_toks if len(tok) >= 5]
            if len(long_toks) < 2:
                continue
            query = '"' + " ".join(span_toks) + '"'
            try:
                rows = db.con.execute(
                    "SELECT rowid FROM pages WHERE pages MATCH ? LIMIT 2",
                    (query,),
                ).fetchall()
            except Exception:
                continue
            if len(rows) != 1:
                continue
            rowid = int(rows[0]["rowid"])
            page = db.fetch([rowid]).get(rowid)
            if page is None or ore.is_disambiguation(page.key):
                continue
            return rowid, " ".join(span_toks)
    return None, ""


def load_reranker():
    from sentence_transformers import CrossEncoder

    device = os.environ.get("BIND_DEVICE", "cpu")
    return CrossEncoder("BAAI/bge-reranker-v2-m3", device=device, max_length=256)


def run_cal() -> None:
    rows = ro.load_hotpot()
    blocked = set()
    bind_path = ro.RESULT_DIR / "bind_hparams.json"
    if bind_path.exists():
        blocked.update(json.loads(bind_path.read_text(encoding="utf-8")).get("cal_ids", []))
    db = ore.WikiDB()
    demo_id, demo_span = define_page(db, "What was the third studio album released by Richard Melville Hall?")
    demo = db.fetch([demo_id]).get(demo_id) if demo_id is not None else None
    if demo is None or demo.title != "Moby" or "richard melville hall" not in demo_span:
        db.close()
        raise SystemExit(f"definition check failed: {demo_span} {None if demo is None else demo.title}")
    chosen_idx = run_bind.fresh_indices(rows, seed=29, extra_ids=blocked)
    chosen = [rows[i] for i in chosen_idx]
    reranker = load_reranker()
    try:
        records = run_bind.evaluate(chosen, db, reranker, finder=define_page)
    finally:
        db.close()
        del reranker
    scores = {name: act.mean(records, name) for name in run_bind.ARMS}
    policy = "bind" if scores["bind"] >= scores["ace"] + MIN_GAIN else "ace"
    by_type = {}
    for kind in sorted({row["type"] for row in records}):
        subset = [row for row in records if row["type"] == kind]
        by_type[kind] = {"n": len(subset), **{name: act.mean(subset, name) for name in run_bind.ARMS}}
    payload = {
        "n": len(chosen),
        "seed": 29,
        "excluded": "seed-7 calibration, seed-7 answer 300, seed-11 second 300, seed-23 name-binding calibration",
        "cal_ids": [row["id"] for row in chosen],
        "min_gain": MIN_GAIN,
        "scores": scores,
        "by_type": by_type,
        "bound_found": sum(1 for row in records if row["bound"]),
        "policy": policy,
        "frozen_before_test": True,
        "rule": (
            "Open the only article that contains the longest question phrase with two words of five letters or more. "
            "Score its hyperlinks with the original question. "
            "Keep the article only when support exceeds the unaltered controller by 0.01."
        ),
    }
    RESULT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pred = ro.RESULT_DIR / "define_cal_predictions.jsonl"
    with pred.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({k: payload[k] for k in payload if k != "cal_ids"}, indent=2), flush=True)


def run_test() -> None:
    if not RESULT.exists():
        raise SystemExit("define_hparams.json is missing")
    hparams = json.loads(RESULT.read_text(encoding="utf-8"))
    if hparams.get("policy") != "bind" or not hparams.get("frozen_before_test"):
        raise SystemExit(f"refusing to score a policy that did not clear calibration: {hparams.get('policy')}")
    blocked = set(hparams["cal_ids"])
    chosen = [row for row in run_fair.hotpot_rows("test") if row["id"] not in blocked]
    pred_path = ro.RESULT_DIR / "define_retrieve_predictions.jsonl"
    done = set()
    if pred_path.exists():
        with pred_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    done.add(json.loads(line)["id"])
    pending = [row for row in chosen if row["id"] not in done]
    print(f"define test pending {len(pending)} done {len(done)}", flush=True)
    db = ore.WikiDB()
    reranker = load_reranker()
    try:
        with pred_path.open("a" if done else "w", encoding="utf-8") as handle:
            for start in range(0, len(pending), 40):
                batch = pending[start : start + 40]
                for row in run_bind.evaluate(batch, db, reranker, finder=define_page):
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
    finally:
        db.close()
        del reranker
    saved = [json.loads(line) for line in pred_path.open(encoding="utf-8") if line.strip()]
    summary = {
        "n": len(saved),
        "policy": "bind",
        "scores": {name: act.mean(saved, name) for name in run_bind.ARMS},
        "frozen_policy": "bind",
    }
    (ro.RESULT_DIR / "define_retrieve_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
