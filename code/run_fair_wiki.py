"""Second-dataset retrieval on questions whose gold titles all exist in the 2017 index.

Uses the same arms as the Hotpot controls. The Hotpot calibration already froze
the policy. This script does not change that choice.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("OPEN_CUDA", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse

import open_retrieve as ore
import run_fair


def covered_rows(dataset: str, titles: dict[str, int]) -> list[dict]:
    from datasets import load_dataset

    rows = []
    if dataset == "2wiki":
        raw = load_dataset("framolfese/2WikiMultihopQA", split="validation")
        for row in raw:
            gold = list(dict.fromkeys(row["supporting_facts"]["title"]))
            keys = [ore.title_key(title) for title in gold]
            if not gold or any(not key or key not in titles for key in keys):
                continue
            rows.append(
                {
                    "id": str(row["id"]),
                    "question": row["question"],
                    "type": str(row["type"]),
                    "gold_keys": keys,
                }
            )
        return rows
    raw = load_dataset("dgslibisey/MuSiQue", split="validation")
    for row in raw:
        gold = [page["title"] for page in row["paragraphs"] if page["is_supporting"]]
        gold = list(dict.fromkeys(gold))
        keys = [ore.title_key(title) for title in gold]
        if not gold or any(not key or key not in titles for key in keys):
            continue
        hops = len(row["question_decomposition"])
        rows.append(
            {
                "id": str(row["id"]),
                "question": row["question"],
                "type": f"{hops}hop",
                "gold_keys": keys,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["2wiki", "musique"], required=True)
    args = parser.parse_args()
    hparams = json.loads(run_fair.RESULT.read_text(encoding="utf-8"))
    if hparams.get("policy") not in run_fair.POLICIES or not hparams.get("frozen_before_test"):
        raise SystemExit("the Hotpot rule is not frozen")
    db = ore.WikiDB()
    try:
        chosen = covered_rows(args.dataset, db.titles)
        print(f"{args.dataset} covered {len(chosen)}", flush=True)
        pred_path = run_fair.ro.RESULT_DIR / f"fair_{args.dataset}_predictions.jsonl"
        done = set()
        if pred_path.exists():
            with pred_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        done.add(json.loads(line)["id"])
        pending = [row for row in chosen if row["id"] not in done]
        print(f"{args.dataset} pending {len(pending)} done {len(done)}", flush=True)
        reranker = run_fair.load_reranker()
        try:
            with pred_path.open("a" if done else "w", encoding="utf-8") as handle:
                for start in range(0, len(pending), 40):
                    batch = pending[start : start + 40]
                    for row in run_fair.evaluate(batch, db, reranker):
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
        finally:
            del reranker
        saved = []
        with pred_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    saved.append(json.loads(line))
        summary = run_fair.summarize(saved, hparams["policy"])
        summary["dataset"] = args.dataset
        summary["frozen_policy"] = hparams["policy"]
        path = run_fair.ro.RESULT_DIR / f"fair_{args.dataset}_summary.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        db.close()


if __name__ == "__main__":
    main()
