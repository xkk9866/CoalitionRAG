"""Restate the missing fact, then follow a hyperlink.

The anchor is still the two strongest cross-encoder pages in the BM25 top 100.
A 3B reader then writes the one question those pages leave open, naming the
entity as the pages name it. Only hyperlinks of the anchors may be opened, and
they are ordered by that restated question. The original question, scored by
the same cross-encoder on the same links, is the control. The restatement is
kept only when calibration support rises by 0.01. Earlier calibrations already
rejected deleting anchor tokens and binding a unique phrase.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("OPEN_CUDA", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import time

import open_retrieve as ore
import run_action as act
import run_bind
import run_experiment as rx
import run_fair
import run_open as ro

RESULT = ro.RESULT_DIR / "nextq_hparams.json"
CACHE = ro.RESULT_DIR / "gen_cache_nextq.jsonl"
MIN_GAIN = 0.01
ARMS = ("flat", "ace", "nextq")
MODEL = "Qwen/Qwen2.5-3B-Instruct"


def render_anchors(pages: dict[int, ore.Page], anchor: list[int]) -> str:
    blocks = []
    for rank, rowid in enumerate(anchor, start=1):
        page = pages.get(rowid)
        if page is None:
            continue
        blocks.append(f"[{rank}] {page.title}\n{page.text}")
    return "\n\n".join(blocks)


def next_prompt(question: str, rendered: str) -> str:
    return (
        "The pages name an entity that the question describes but do not state the answer.\n"
        "Write the one next question that asks for the missing fact.\n"
        "Use the entity's name as the pages give it.\n"
        "Reply with only that question.\n\n"
        f"{rendered}\n\n"
        f"Question: {question}\n"
        "Next question:"
    )


def clean_question(text: str) -> str:
    line = (text or "").strip().splitlines()[0].strip() if text else ""
    for prefix in ("Next question:", "Question:"):
        if line.lower().startswith(prefix.lower()):
            line = line[len(prefix) :].strip()
    return line.strip(" \"'")


def load_cache() -> dict[str, str]:
    found = {}
    if not CACHE.exists():
        return found
    with CACHE.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            found[row["key"]] = row["text"]
    return found


def generate(prompts: list[str]) -> dict[str, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache = load_cache()
    missing = [prompt for prompt in dict.fromkeys(prompts) if rx.prompt_key(prompt) not in cache]
    if not missing:
        return {prompt: cache[rx.prompt_key(prompt)] for prompt in prompts}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    free, _total = torch.cuda.mem_get_info()
    if free >= 8 * 1024**3:
        print(f"moving 3B to cuda:0 with {free / 1024**3:.1f} GiB free", flush=True)
        model.to("cuda:0")
    else:
        print(f"3B stays on CPU; cuda free {free / 1024**3:.1f} GiB", flush=True)
    model.eval()
    with CACHE.open("a", encoding="utf-8") as handle:
        for start in range(0, len(missing), 4):
            chunk = missing[start : start + 4]
            chats = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in chunk
            ]
            enc = tokenizer(chats, return_tensors="pt", padding=True, truncation=True, max_length=1536)
            enc = {key: value.to(model.device) for key, value in enc.items()}
            with torch.inference_mode():
                out = model.generate(
                    **enc,
                    max_new_tokens=32,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            input_len = enc["input_ids"].shape[1]
            for prompt, row in zip(chunk, out):
                text = clean_question(tokenizer.decode(row[input_len:], skip_special_tokens=True))
                cache[rx.prompt_key(prompt)] = text
                handle.write(json.dumps({"key": rx.prompt_key(prompt), "text": text}, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"next questions {min(start + 4, len(missing))}/{len(missing)}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {prompt: cache[rx.prompt_key(prompt)] for prompt in prompts}


def blocked_ids() -> set[str]:
    found = set()
    for name in ("bind_hparams.json", "define_hparams.json"):
        path = ro.RESULT_DIR / name
        if path.exists():
            found.update(json.loads(path.read_text(encoding="utf-8")).get("cal_ids", []))
    return found


def observe(rows: list[dict], db: ore.WikiDB, reranker) -> list[dict]:
    """Anchors, legal links, and the original-question controller."""
    packed = []
    for start in range(0, len(rows), 40):
        batch = rows[start : start + 40]
        t0 = time.perf_counter()
        print(f"observe {start}:{start + len(batch)} / {len(rows)}", flush=True)
        bm25_lists = db.search_many([row["question"] for row in batch], ore.FIRST_HOP_K, workers=8)
        pages = db.fetch([rowid for ids in bm25_lists for rowid in ids])
        pairs = []
        owners = []
        for row, ids in zip(batch, bm25_lists):
            for rowid in ids:
                page = pages.get(rowid)
                if page is None:
                    continue
                pairs.append((row["question"], page.title + ". " + page.text))
                owners.append((row["id"], rowid))
        values = run_fair.batched_scores(reranker, pairs)
        scores: dict[str, dict[int, float]] = {}
        for (qid, rowid), value in zip(owners, values):
            scores.setdefault(qid, {})[rowid] = value
        link_pairs = []
        link_owners = []
        partials = []
        for row, ids in zip(batch, bm25_lists):
            anchor = run_fair.top_scored(ids, scores.get(row["id"], {}), ore.ANCHOR_K, set())
            offers = []
            blocked = set(anchor)
            for rowid in anchor:
                page = pages.get(rowid)
                if page is None:
                    continue
                for link_id, _phrase in run_fair.link_offers(page, db.titles, blocked):
                    offers.append(link_id)
                    blocked.add(link_id)
            partials.append((row, ids, anchor, offers, scores.get(row["id"], {})))
        extra = [rowid for _row, _ids, _anchor, offers, _scores in partials for rowid in offers if rowid not in pages]
        if extra:
            pages.update(db.fetch(extra))
        for row, _ids, _anchor, offers, _scores in partials:
            for rowid in offers:
                page = pages.get(rowid)
                if page is None:
                    continue
                link_pairs.append((row["question"], page.title + ". " + page.text))
                link_owners.append((row["id"], rowid))
        link_values = run_fair.batched_scores(reranker, link_pairs)
        link_scores: dict[str, dict[int, float]] = {}
        bodies: dict[str, dict[int, str]] = {}
        for (qid, rowid), value in zip(link_owners, link_values):
            link_scores.setdefault(qid, {})[rowid] = value
            page = pages[rowid]
            bodies.setdefault(qid, {})[rowid] = page.title + ". " + page.text
        for row, ids, anchor, offers, flat_scores in partials:
            gold = set(row["gold_keys"])
            flat_ids = run_fair.top_scored(ids, flat_scores, ore.BUDGET, set())
            ace_ids = act.pack_ids(
                anchor,
                run_fair.top_scored(offers, link_scores.get(row["id"], {}), ore.BUDGET, set(anchor)),
                pages,
            )
            record = {
                "id": row["id"],
                "type": row["type"],
                "question": row["question"],
                "anchor_ids": anchor,
                "offer_ids": offers,
                "bodies": bodies.get(row["id"], {}),
                "rendered": render_anchors(pages, anchor),
                "flat": act.support({pages[i].key for i in flat_ids if i in pages}, gold),
                "ace": act.support({pages[i].key for i in ace_ids if i in pages}, gold),
                "flat_ids": flat_ids,
                "ace_ids": ace_ids,
            }
            packed.append(record)
        print(f"observed {start + len(batch)} sec {time.perf_counter() - t0:.1f}", flush=True)
    return packed


def score_next(records: list[dict], db: ore.WikiDB) -> list[dict]:
    import gc

    import torch

    prompts = [next_prompt(record["question"], record["rendered"]) for record in records]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    rewrites = generate(prompts)
    reranker = run_fair.load_reranker()
    try:
        pairs = []
        owners = []
        for record, prompt in zip(records, prompts):
            rewrite = rewrites.get(prompt) or record["question"]
            record["nextq_text"] = rewrite
            bodies = {int(key): value for key, value in record["bodies"].items()}
            record["bodies"] = bodies
            for rowid in record["offer_ids"]:
                body = bodies.get(rowid)
                if body is None:
                    continue
                pairs.append((rewrite, body))
                owners.append((record["id"], rowid))
        values = run_fair.batched_scores(reranker, pairs)
    finally:
        del reranker
    scores: dict[str, dict[int, float]] = {}
    for (qid, rowid), value in zip(owners, values):
        scores.setdefault(qid, {})[rowid] = value
    id_keys = {}
    needed = [rowid for record in records for rowid in record["anchor_ids"] + record["offer_ids"]]
    fetched = db.fetch(list(dict.fromkeys(needed)))
    for rowid, page in fetched.items():
        id_keys[rowid] = page.key
    finished = []
    for record in records:
        chosen = act.pack_ids(
            record["anchor_ids"],
            run_fair.top_scored(record["offer_ids"], scores.get(record["id"], {}), ore.BUDGET, set(record["anchor_ids"])),
            fetched,
        )
        finished.append(record)
        record["nextq_ids"] = chosen
        record["_keys"] = id_keys
    return finished


def attach_support(records: list[dict], rows_by_id: dict[str, dict]) -> list[dict]:
    out = []
    for record in records:
        gold = set(rows_by_id[record["id"]]["gold_keys"])
        keys = record.pop("_keys")
        chosen = record["nextq_ids"]
        record["nextq"] = act.support({keys[rowid] for rowid in chosen if rowid in keys}, gold)
        record.pop("bodies", None)
        record.pop("rendered", None)
        out.append(record)
    return out


def run_cal(limit: int) -> None:
    rows = ro.load_hotpot()
    chosen_idx = run_bind.fresh_indices(rows, seed=31, extra_ids=blocked_ids())
    if limit:
        chosen_idx = chosen_idx[:limit]
    chosen = [rows[i] for i in chosen_idx]
    by_id = {row["id"]: row for row in chosen}
    db = ore.WikiDB()
    reranker = run_fair.load_reranker()
    try:
        observed = observe(chosen, db, reranker)
    finally:
        del reranker
    records = score_next(observed, db)
    db.close()
    records = attach_support(records, by_id)
    for record in records[:8]:
        print(json.dumps({"q": record["question"], "next": record["nextq_text"], "ace": record["ace"], "nextq": record["nextq"]}, ensure_ascii=False), flush=True)
    scores = {name: act.mean(records, name) for name in ARMS}
    policy = "nextq" if (not limit and scores["nextq"] >= scores["ace"] + MIN_GAIN) else ("unfrozen" if limit else "ace")
    payload = {
        "n": len(records),
        "seed": 31,
        "limited": bool(limit),
        "excluded": "seed-7 calibration and answer slices, seed-11 second 300, seed-23 and seed-29 calibrations",
        "cal_ids": [row["id"] for row in chosen],
        "min_gain": MIN_GAIN,
        "scores": scores,
        "policy": policy,
        "frozen_before_test": not bool(limit) and policy == "nextq",
        "rule": (
            "Read the two anchor abstracts and write the question they leave open. "
            "Rank their hyperlinks by that question with the same cross-encoder. "
            "Keep the restatement only when support exceeds the original question by 0.01."
        ),
    }
    path = ro.RESULT_DIR / ("nextq_cal_debug.json" if limit else RESULT)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pred = ro.RESULT_DIR / ("nextq_cal_debug.jsonl" if limit else "nextq_cal_predictions.jsonl")
    with pred.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({k: payload[k] for k in payload if k != "cal_ids"}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["cal"], required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    run_cal(args.limit)


if __name__ == "__main__":
    main()
