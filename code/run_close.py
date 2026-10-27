"""Read the two sentences that close the question.

The pages are the ones the frozen hyperlink controller already committed.
Every sentence in those abstracts is scored by the same cross-encoder.
The reader sees only the two highest sentences, then answers.
This is compared with reading the four abstracts on the 200 calibration
questions, which were not used to score answers. The sentence reader is
kept only when token F1 rises by 0.02.
"""

from __future__ import annotations

import json
import os
import re

os.environ.setdefault("OPEN_CUDA", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse

import open_retrieve as ore
import run_experiment as rx
import run_fair
import run_open as ro

RESULT = ro.RESULT_DIR / "close_hparams.json"
CACHE = ro.RESULT_DIR / "gen_cache_close.jsonl"
MIN_GAIN = 0.02
MODEL = "Qwen/Qwen2.5-3B-Instruct"
SPLIT = re.compile(r"(?<=[.!?])\s+")


def sentences(page: ore.Page) -> list[str]:
    found = []
    for part in SPLIT.split(page.text or ""):
        part = part.strip()
        if len(part) >= 20:
            found.append(part)
    if not found and page.text:
        found.append(page.text.strip())
    return found


def full_prompt(question: str, pages: list[ore.Page]) -> str:
    return ore.render_pages(question, pages)


def sent_prompt(question: str, picked: list[tuple[str, str]]) -> str:
    lines = [
        "Answer the question using the sentences.",
        "Reply with only the short answer.",
        "",
    ]
    for rank, (title, sent) in enumerate(picked, start=1):
        lines.append(f"[{rank}] {title}")
        lines.append(sent)
        lines.append("")
    lines.append(f"Question: {question}")
    lines.append("Answer:")
    return "\n".join(lines)


def generate(prompts: list[str]) -> dict[str, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache = {}
    if CACHE.exists():
        with CACHE.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    cache[row["key"]] = row["text"]
    missing = [prompt for prompt in dict.fromkeys(prompts) if rx.prompt_key(prompt) not in cache]
    if missing:
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
            for start in range(0, len(missing), 8):
                chunk = missing[start : start + 8]
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
                    text = tokenizer.decode(row[input_len:], skip_special_tokens=True).strip().splitlines()[0].strip()
                    cache[rx.prompt_key(prompt)] = text
                    handle.write(json.dumps({"key": rx.prompt_key(prompt), "text": text}, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"answers {min(start + 8, len(missing))}/{len(missing)}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {prompt: cache[rx.prompt_key(prompt)] for prompt in prompts}


def pick(question: str, pages: list[ore.Page], reranker) -> list[tuple[str, str]]:
    pairs = []
    owners = []
    for page in pages:
        for sent in sentences(page):
            pairs.append((question, sent))
            owners.append((page.title, sent))
    if not pairs:
        return [(page.title, page.text) for page in pages[:2]]
    values = run_fair.batched_scores(reranker, pairs)
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    picked = []
    seen = set()
    for index in order:
        item = owners[index]
        if item in seen:
            continue
        seen.add(item)
        picked.append(item)
        if len(picked) >= 2:
            break
    return picked


def run_cal() -> None:
    rows = {row["id"]: row for row in run_fair.hotpot_rows("cal")}
    records = [json.loads(line) for line in (ro.RESULT_DIR / "chain_cal_predictions.jsonl").open(encoding="utf-8")]
    db = ore.WikiDB()
    reranker = run_fair.load_reranker()
    prepared = []
    pairs = []
    owners = []
    try:
        for record in records:
            row = rows[record["id"]]
            fetched = db.fetch(record["ace_ids"])
            ordered = [fetched[rowid] for rowid in record["ace_ids"] if rowid in fetched]
            local = []
            for page in ordered:
                for sent in sentences(page):
                    local.append((page.title, sent))
            prepared.append({"row": row, "ordered": ordered, "local": local, "start": len(pairs)})
            for title, sent in local:
                pairs.append((row["question"], sent))
                owners.append((record["id"], title, sent))
        values = run_fair.batched_scores(reranker, pairs)
    finally:
        db.close()
        del reranker
    prompts = []
    packed = []
    cursor = 0
    for item in prepared:
        row = item["row"]
        local = item["local"]
        scored = list(zip(values[cursor : cursor + len(local)], local))
        cursor += len(local)
        scored.sort(key=lambda pair: pair[0], reverse=True)
        chosen = []
        seen = set()
        for _score, sent in scored:
            if sent in seen:
                continue
            seen.add(sent)
            chosen.append(sent)
            if len(chosen) >= 2:
                break
        if not chosen:
            chosen = [(page.title, page.text) for page in item["ordered"][:2]]
        full = full_prompt(row["question"], item["ordered"])
        sent = sent_prompt(row["question"], chosen)
        prompts.extend([full, sent])
        packed.append({"id": row["id"], "type": row["type"], "answers": row["answers"], "full": full, "sent": sent})
    texts = generate(prompts)
    scored = []
    for record in packed:
        full_em, full_f1 = rx.best_em_f1(texts[record["full"]], record["answers"])
        sent_em, sent_f1 = rx.best_em_f1(texts[record["sent"]], record["answers"])
        scored.append(
            {
                "id": record["id"],
                "type": record["type"],
                "full_em": full_em,
                "full_f1": full_f1,
                "sent_em": sent_em,
                "sent_f1": sent_f1,
                "full_pred": texts[record["full"]],
                "sent_pred": texts[record["sent"]],
            }
        )
    full_f1 = sum(row["full_f1"] for row in scored) / len(scored)
    sent_f1 = sum(row["sent_f1"] for row in scored) / len(scored)
    policy = "sent" if sent_f1 >= full_f1 + MIN_GAIN else "full"
    payload = {
        "n": len(scored),
        "min_gain": MIN_GAIN,
        "full_f1": full_f1,
        "sent_f1": sent_f1,
        "full_em": sum(row["full_em"] for row in scored) / len(scored),
        "sent_em": sum(row["sent_em"] for row in scored) / len(scored),
        "policy": policy,
        "frozen_before_test": True,
        "rule": (
            "On the pages already committed, score each sentence with the cross-encoder. "
            "Show the reader the two highest sentences. "
            "Keep that reader only when calibration token F1 exceeds the four-abstract reader by 0.02."
        ),
    }
    RESULT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pred = ro.RESULT_DIR / "close_cal_predictions.jsonl"
    with pred.open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["cal"], required=True)
    args = parser.parse_args()
    run_cal()


if __name__ == "__main__":
    main()
