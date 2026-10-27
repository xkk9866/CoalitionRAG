"""Answer the frozen ace pages once, on the predeclared 300-question slice."""

from __future__ import annotations

import argparse
import gc
import json
import os

os.environ.setdefault("OPEN_CUDA", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

import open_retrieve as ore
import run_experiment as rx
import run_open as ro

PRED = ro.RESULT_DIR / "policy_retrieve_predictions.jsonl"


def paired(left, right):
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
    return {"n": int(len(diff)), "delta": float(diff.mean()), "se": se}


def load_f1(path, method: str) -> dict[str, float]:
    found = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["method"] == method:
                found[row["id"]] = float(row["f1"])
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["3b", "7b"], required=True)
    args = parser.parse_args()
    if args.model == "7b":
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        cache_path = ro.RESULT_DIR / "gen_cache_policy7b.jsonl"
        base_path = ro.RESULT_DIR / "open_predictions.jsonl"
        out_path = ro.RESULT_DIR / "policy_answer_7b_summary.json"
        pred_path = ro.RESULT_DIR / "policy_answer_7b_predictions.jsonl"
        batch_size = 4
    else:
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        cache_path = ro.RESULT_DIR / "gen_cache_policy3b.jsonl"
        base_path = ro.RESULT_DIR / "open3b_predictions.jsonl"
        out_path = ro.RESULT_DIR / "policy_answer_3b_summary.json"
        pred_path = ro.RESULT_DIR / "policy_answer_3b_predictions.jsonl"
        batch_size = 8
    rows = ro.load_hotpot()
    _, test_idx, _ = ro.split_indices(rows)
    chosen = [rows[i] for i in test_idx]
    saved = {}
    with PRED.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                saved[row["id"]] = row
    missing = [row["id"] for row in chosen if row["id"] not in saved]
    if missing:
        raise SystemExit(f"ace pages are not ready for {len(missing)} test questions")
    dense_f1 = load_f1(base_path, "dense")
    links_f1 = load_f1(base_path, "links")
    db = ore.WikiDB()
    prompts = []
    metas = []
    try:
        for row in chosen:
            packed = saved[row["id"]]
            pages = db.fetch(packed["rowids"])
            ordered = [pages[rowid] for rowid in packed["rowids"] if rowid in pages]
            prompt = ore.render_pages(row["question"], ordered)
            prompts.append(prompt)
            metas.append((row, packed, prompt))
    finally:
        db.close()
    del db
    gc.collect()
    print("reader prompts ready", len(prompts), flush=True)
    cache = rx.load_cache(cache_path)
    rx.generate_missing(prompts, cache, batch_size=batch_size, model_name=model_name, cache_path=cache_path)
    f1s = []
    ems = []
    base = []
    links = []
    rows_out = []
    by_type: dict[str, list[float]] = {}
    for row, packed, prompt in metas:
        gen = cache[rx.prompt_key(prompt)]
        pred = rx.extract_answer(gen["text"])
        em, f1 = rx.best_em_f1(pred, row["answers"])
        f1s.append(f1)
        ems.append(em)
        base.append(dense_f1[row["id"]])
        links.append(links_f1[row["id"]])
        by_type.setdefault(row["type"], []).append(f1)
        rows_out.append(
            {
                "id": row["id"],
                "type": row["type"],
                "pred": pred,
                "gold": row["answers"][0],
                "em": em,
                "f1": f1,
                "support": packed["ace"],
                "dense_f1": dense_f1[row["id"]],
                "links_f1": links_f1[row["id"]],
                "tokens": gen["prompt_tokens"],
            }
        )
    summary = {
        "n": len(rows_out),
        "model": model_name,
        "em": float(np.mean(ems)),
        "f1": float(np.mean(f1s)),
        "support": float(np.mean([row["support"] for row in rows_out])),
        "tokens": float(np.mean([row["tokens"] for row in rows_out])),
        "versus_dense_f1": paired(f1s, base),
        "versus_links_f1": paired(f1s, links),
        "by_type": {name: float(np.mean(values)) for name, values in sorted(by_type.items())},
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pred_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows_out),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
