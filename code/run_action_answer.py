"""Answer the frozen cross-encoder pages once, on the predeclared 300-question slice."""

from __future__ import annotations

import json
import os

os.environ.setdefault("OPEN_CUDA", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from pathlib import Path

import numpy as np

import open_retrieve as ore
import run_experiment as rx
import run_open as ro

PRED = ro.RESULT_DIR / "action_retrieve_predictions.jsonl"
MODEL = "Qwen/Qwen2.5-3B-Instruct"
CACHE = ro.RESULT_DIR / "gen_cache_action3b.jsonl"
BASE = ro.RESULT_DIR / "open3b_predictions.jsonl"
OUT = ro.RESULT_DIR / "action_answer_3b_summary.json"


def paired(left, right):
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
    return {"n": int(len(diff)), "delta": float(diff.mean()), "se": se}


def main() -> None:
    rows = ro.load_hotpot()
    _, test_idx, _ = ro.split_indices(rows)
    chosen = [rows[i] for i in test_idx]
    saved = {}
    with PRED.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                saved[row["id"]] = row
    dense_f1 = {}
    with BASE.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["method"] == "dense":
                dense_f1[row["id"]] = float(row["f1"])
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
    import gc

    gc.collect()
    print("reader prompts ready", len(prompts), flush=True)
    cache = rx.load_cache(CACHE)
    rx.generate_missing(prompts, cache, batch_size=8, model_name=MODEL, cache_path=CACHE)
    f1s = []
    ems = []
    base = []
    rows_out = []
    for row, packed, prompt in metas:
        gen = cache[rx.prompt_key(prompt)]
        pred = rx.extract_answer(gen["text"])
        em, f1 = rx.best_em_f1(pred, row["answers"])
        f1s.append(f1)
        ems.append(em)
        base.append(dense_f1[row["id"]])
        rows_out.append(
            {
                "id": row["id"],
                "type": row["type"],
                "pred": pred,
                "gold": row["answers"][0],
                "em": em,
                "f1": f1,
                "support": packed["ce"],
                "dense_f1": dense_f1[row["id"]],
            }
        )
    summary = {
        "n": len(rows_out),
        "model": MODEL,
        "em": float(np.mean(ems)),
        "f1": float(np.mean(f1s)),
        "support": float(np.mean([row["support"] for row in rows_out])),
        "versus_dense_f1": paired(f1s, base),
    }
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (ro.RESULT_DIR / "action_answer_3b_predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows_out),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
