"""Read the frozen fair pages and score answer F1.

The retrieval file is not modified. Each arm and model uses its own generation
cache. A clipped run shortens every page until the prompt is no longer than
the dense reader's prompt for that question.
"""

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
import run_fair
import run_open as ro


def paired(left, right) -> dict:
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
    rng = np.random.default_rng(0)
    if len(diff) > 1:
        draws = rng.integers(0, len(diff), size=(10000, len(diff)))
        means = diff[draws].mean(axis=1)
        lo, hi = (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))
    else:
        lo = hi = float(diff.mean()) if len(diff) else 0.0
    return {
        "n": int(len(diff)),
        "delta": float(diff.mean()) if len(diff) else 0.0,
        "se": se,
        "boot_low": lo,
        "boot_high": hi,
    }


def load_f1(path, method: str) -> dict[str, float]:
    found = {}
    if not path.exists():
        return found
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("method") == method:
                found[row["id"]] = float(row["f1"])
    return found


def load_tokens(path, method: str) -> dict[str, int]:
    found = {}
    if not path.exists():
        return found
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("method") == method and "tokens" in row:
                found[row["id"]] = int(row["tokens"])
    return found


def clip_prompt(question: str, pages: list[ore.Page], tokenizer, target: int) -> str:
    full = ore.render_pages(question, pages)
    if len(tokenizer(full, add_special_tokens=False)["input_ids"]) <= target:
        return full
    words = [page.text.split() for page in pages]
    hi = max((len(item) for item in words), default=0)
    best = ore.render_pages(
        question,
        [ore.Page(page.rowid, page.title, "", page.key) for page in pages],
    )
    lo = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        clipped = [
            ore.Page(page.rowid, page.title, " ".join(tokens[:mid]), page.key)
            for page, tokens in zip(pages, words)
        ]
        prompt = ore.render_pages(question, clipped)
        n_tok = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if n_tok <= target:
            best = prompt
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=run_fair.ARMS)
    parser.add_argument("--model", required=True, choices=["3b", "7b", "14b"])
    parser.add_argument("--scope", required=True, choices=["test300", "second300", "all"])
    parser.add_argument("--clip-dense", action="store_true")
    args = parser.parse_args()
    if args.model == "14b":
        model_name = "Qwen/Qwen2.5-14B-Instruct"
        batch_size = 2
    elif args.model == "7b":
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        batch_size = 4
    else:
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        batch_size = 8
    tag = f"{args.arm}_{args.model}_{args.scope}"
    if args.clip_dense:
        tag += "_clip"
    cache_path = ro.RESULT_DIR / f"gen_cache_fair_{tag}.jsonl"
    out_path = ro.RESULT_DIR / f"fair_answer_{tag}_summary.json"
    pred_path = ro.RESULT_DIR / f"fair_answer_{tag}_predictions.jsonl"
    rows = ro.load_hotpot()
    cal_idx, test_idx, second_idx = ro.split_indices(rows)
    if args.scope == "test300":
        chosen = [rows[i] for i in test_idx]
    elif args.scope == "second300":
        chosen = [rows[i] for i in second_idx]
    else:
        blocked = {rows[i]["id"] for i in cal_idx}
        chosen = [row for row in rows if row["id"] not in blocked]
    saved = {}
    src = ro.RESULT_DIR / "fair_retrieve_predictions.jsonl"
    with src.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                saved[row["id"]] = row
    missing = [row["id"] for row in chosen if row["id"] not in saved]
    if missing:
        raise SystemExit(f"fair pages are not ready for {len(missing)} questions")
    dense_f1 = load_f1(ro.RESULT_DIR / "open_predictions.jsonl", "dense")
    links_f1 = load_f1(ro.RESULT_DIR / "open_predictions.jsonl", "links")
    dense_tokens = load_tokens(ro.RESULT_DIR / "open_predictions.jsonl", "dense")
    tokenizer = None
    if args.clip_dense:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    db = ore.WikiDB()
    prompts = []
    metas = []
    try:
        for row in chosen:
            packed = saved[row["id"]]
            ids = packed[args.arm + "_ids"]
            pages = db.fetch(ids)
            ordered = [pages[rowid] for rowid in ids if rowid in pages]
            if args.clip_dense:
                target = dense_tokens.get(row["id"], 446)
                prompt = clip_prompt(row["question"], ordered, tokenizer, target)
            else:
                prompt = ore.render_pages(row["question"], ordered)
            prompts.append(prompt)
            metas.append((row, packed, prompt))
    finally:
        db.close()
    del db
    gc.collect()
    print("reader prompts ready", len(prompts), args.arm, args.scope, flush=True)
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
        by_type.setdefault(row["type"], []).append(f1)
        item = {
            "id": row["id"],
            "type": row["type"],
            "arm": args.arm,
            "pred": pred,
            "gold": row["answers"][0],
            "em": em,
            "f1": f1,
            "support": packed[args.arm],
            "tokens": gen["prompt_tokens"],
            "clipped": bool(args.clip_dense),
        }
        if row["id"] in dense_f1:
            item["dense_f1"] = dense_f1[row["id"]]
            base.append(f1)
            links.append(links_f1[row["id"]])
        rows_out.append(item)
    summary = {
        "n": len(rows_out),
        "arm": args.arm,
        "model": model_name,
        "scope": args.scope,
        "clipped": bool(args.clip_dense),
        "em": float(np.mean(ems)),
        "f1": float(np.mean(f1s)),
        "support": float(np.mean([row["support"] for row in rows_out])),
        "tokens": float(np.mean([row["tokens"] for row in rows_out])),
        "by_type": {name: float(np.mean(values)) for name, values in sorted(by_type.items())},
    }
    if base:
        paired_f1 = [row["f1"] for row in rows_out if "dense_f1" in row]
        paired_dense = [row["dense_f1"] for row in rows_out if "dense_f1" in row]
        paired_links = [links_f1[row["id"]] for row in rows_out if "dense_f1" in row]
        summary["versus_dense_f1"] = paired(paired_f1, paired_dense)
        summary["versus_links_f1"] = paired(paired_f1, paired_links)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pred_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows_out), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
