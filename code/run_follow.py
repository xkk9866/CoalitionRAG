"""One-shot evaluation of the citation-follow rule.

The rule is read from results/follow_hparams.json. This script does not retune it.
Gold titles are used only when the predictions are scored.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import run_experiment as rx

ROOT = Path(__file__).resolve().parents[1]
HPARAMS = rx.RESULT_DIR / "follow_hparams.json"


def stratified_remaining(types: list[str], blocked: set[int], n_test: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, kind in enumerate(types):
        if i not in blocked:
            buckets[kind].append(i)
    total = sum(len(v) for v in buckets.values())
    test: list[int] = []
    for kind in sorted(buckets):
        idxs = buckets[kind][:]
        rng.shuffle(idxs)
        nt = int(round(n_test * len(idxs) / total))
        test.extend(idxs[:nt])
    rng.shuffle(test)
    return test[:n_test]


def load_split(dataset: str, seed: int) -> list[dict]:
    from datasets import load_dataset

    if dataset == "hotpot":
        raw = load_dataset("hotpot_qa", "distractor", split="validation")
        types = list(raw["type"])

        def convert(i: int) -> dict:
            row = raw[i]
            titles, texts, gold = rx._join_hotpot_like(row)
            return {
                "id": row["id"],
                "question": row["question"],
                "answers": [row["answer"]],
                "type": row["type"],
                "titles": titles,
                "texts": texts,
                "gold_titles": gold,
            }

    elif dataset == "2wiki":
        raw = load_dataset("framolfese/2WikiMultihopQA", split="validation")
        types = list(raw["type"])

        def convert(i: int) -> dict:
            row = raw[i]
            titles, texts, gold = rx._join_hotpot_like(row)
            return {
                "id": str(row["id"]),
                "question": row["question"],
                "answers": [row["answer"]],
                "type": row["type"],
                "titles": titles,
                "texts": texts,
                "gold_titles": gold,
            }

    elif dataset == "musique":
        raw = load_dataset("dgslibisey/MuSiQue", split="validation")
        types = [f"{len(row['question_decomposition'])}hop" for row in raw]

        def convert(i: int) -> dict:
            row = raw[i]
            titles = [p["title"] for p in row["paragraphs"]]
            texts = [p["paragraph_text"] for p in row["paragraphs"]]
            gold = {p["title"] for p in row["paragraphs"] if p["is_supporting"]}
            aliases = [row["answer"]] + list(row["answer_aliases"] or [])
            return {
                "id": row["id"],
                "question": row["question"],
                "answers": aliases,
                "type": types[i],
                "titles": titles,
                "texts": texts,
                "gold_titles": gold,
            }

    else:
        raise ValueError(dataset)

    if seed == 7:
        _cal, test_i = rx.stratified_indices(types, 200, 300, 7)
        chosen = test_i
    else:
        cal_i, test_i = rx.stratified_indices(types, 200, 300, 7)
        chosen = stratified_remaining(types, set(cal_i + test_i), 300, seed)
    return [convert(i) for i in chosen]


def dense_pages(ex: dict) -> list[dict]:
    return [rx._page_of(ex, idx, "dense") for idx in rx._dense_order(ex)]


def bm25_pages(ex: dict) -> list[dict]:
    scores = rx.cached_agent_scores(ex, "bm25")
    pages = []
    held: set[str] = set()
    for idx in rx.top_indices(scores, len(scores)):
        title = ex["titles"][idx]
        if title in held:
            continue
        held.add(title)
        pages.append(rx._page_of(ex, int(idx), "bm25"))
        if len(pages) == rx.COVER_SLOTS:
            break
    return pages


def select_pages(method: str, ex: dict) -> list[dict]:
    if method == "dense4":
        return dense_pages(ex)
    if method == "bm25":
        return bm25_pages(ex)
    if method == "rrf":
        return rx.rrf_pages(ex)[1]
    if method == "follow":
        return rx.select_follow(ex, "dense")[1]
    if method == "anchor":
        return rx.select_anchor(ex, rx.build_offers(ex, rx.MAIN_AGENTS), True, False)[1]
    raise KeyError(method)


def cite_rows(ex: dict, pages: list[dict], dense_titles: set[str]) -> list[dict]:
    rows = []
    for i, page in enumerate(pages):
        if page["title"] in dense_titles:
            continue
        key = page["title"].lower().strip()
        citer = ""
        for prev in pages[:i]:
            blob = (ex["titles"][prev["idx"]] + " " + ex["texts"][prev["idx"]]).lower()
            if len(key) > 3 and key in blob:
                citer = prev["title"]
                break
        longer = False
        for other in ex["titles"]:
            folded = other.lower().strip()
            if folded != key and len(folded) > len(key) and key in folded:
                longer = True
                break
        rows.append(
            {
                "title": page["title"],
                "citer": citer,
                "gold": page["title"] in ex["gold_titles"],
                "longer_title_hit": longer,
                "chars": len(key),
            }
        )
    return rows


def spans_for(qa: rx.SpanReader, jobs: list[dict], by_id: dict[tuple[str, str], dict]) -> dict[tuple[str, str, int], str]:
    questions = []
    contexts = []
    keys = []
    seen = set()
    for job in jobs:
        ex = by_id[(job["dataset"], job["id"])]
        for page in job["pages"]:
            key = (job["dataset"], job["id"], page["idx"])
            if key in seen:
                continue
            seen.add(key)
            questions.append(ex["question"])
            contexts.append(rx.clip_words(ex["texts"][page["idx"]], 160))
            keys.append(key)
    found: dict[tuple[str, str, int], str] = {}
    if not questions:
        return found
    outputs = qa.answer_batch(questions, contexts, batch_size=16)
    for key, (answer, score) in zip(keys, outputs):
        if score >= 0.05 and answer:
            found[key] = answer
    return found


def paired(rows_a: list[dict], rows_b: list[dict], key: str = "f1") -> dict:
    other = {row["id"]: row for row in rows_b}
    diffs = np.array([row[key] - other[row["id"]][key] for row in rows_a], dtype=np.float64)
    changed = np.array([row["n_extra"] > 0 for row in rows_a], dtype=bool)
    out = {
        "mean": float(diffs.mean()) if len(diffs) else 0.0,
        "se": float(diffs.std(ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else 0.0,
        "n": int(len(diffs)),
    }
    if changed.any():
        sub = diffs[changed]
        out["changed_mean"] = float(sub.mean())
        out["changed_se"] = float(sub.std(ddof=1) / np.sqrt(len(sub))) if len(sub) > 1 else 0.0
        out["changed_n"] = int(len(sub))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--methods", default="dense4,bm25,rrf,follow,anchor")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not HPARAMS.exists():
        raise SystemExit(f"missing frozen rule: {HPARAMS}")
    rule = json.loads(HPARAMS.read_text(encoding="utf-8"))
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    cache_path = rx.RESULT_DIR / ("gen_cache.jsonl" if "7B" in args.model else f"gen_cache_{args.tag}.jsonl")
    jobs_path = rx.RESULT_DIR / f"{args.tag}_jobs.jsonl"
    rx.COVER_BETA = 0.5

    if args.resume and jobs_path.exists():
        jobs = [json.loads(line) for line in jobs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        print("resumed jobs", len(jobs), flush=True)
    else:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda:0")
        splits = {}
        for dataset in ("hotpot", "2wiki", "musique"):
            print("loading", dataset, "seed", args.seed, flush=True)
            rows = load_split(dataset, args.seed)
            rx.attach_embeddings(rows, encoder)
            splits[dataset] = rows
            print(" n", len(rows), flush=True)
        del encoder

        jobs = []
        by_id: dict[tuple[str, str], dict] = {}
        for dataset, rows in splits.items():
            for ex in rows:
                by_id[(dataset, ex["id"])] = ex
                base_titles = {page["title"] for page in dense_pages(ex)}
                for method in methods:
                    pages = select_pages(method, ex)
                    titles = [page["title"] for page in pages]
                    extra = [title for title in titles if title not in base_titles]
                    agents: list[str] = []
                    for page in pages:
                        if page["agent"] not in agents:
                            agents.append(page["agent"])
                    jobs.append(
                        {
                            "dataset": dataset,
                            "split": "main",
                            "method": method,
                            "id": ex["id"],
                            "type": ex["type"],
                            "answers": ex["answers"],
                            "agents": agents,
                            "titles": titles,
                            "n_extra": len(extra),
                            "cites": cite_rows(ex, pages, base_titles) if method == "follow" else [],
                            "support": rx.support_recall(titles, ex["gold_titles"]),
                            "size": float(len(agents)),
                            "pages": pages,
                        }
                    )
        print("spans", flush=True)
        qa = rx.SpanReader()
        span_map = spans_for(qa, jobs, by_id)
        del qa
        import gc
        import torch

        gc.collect()
        torch.cuda.empty_cache()
        for job in jobs:
            ex = by_id[(job["dataset"], job["id"])]
            blocks = []
            for page in job["pages"]:
                blocks.append(
                    {
                        "agent": page["agent"],
                        "title": page["title"],
                        "text": page["text"],
                        "span": span_map.get((job["dataset"], job["id"], page["idx"]), ""),
                    }
                )
            job["prompt"] = rx.render_prompt(ex["question"], blocks, "rag")
            job.pop("pages")
        rx.RESULT_DIR.mkdir(parents=True, exist_ok=True)
        with jobs_path.open("w", encoding="utf-8") as handle:
            for job in jobs:
                handle.write(json.dumps(job, ensure_ascii=False) + "\n")
        print("wrote jobs", len(jobs), flush=True)
        if args.seed == 7:
            cover_path = rx.RESULT_DIR / "cover_predictions.jsonl"
            if cover_path.exists():
                cover_ids = {
                    (row["dataset"], row["id"])
                    for row in (json.loads(line) for line in cover_path.read_text(encoding="utf-8").splitlines() if line.strip())
                    if row["method"] == "single" and row["split"] == "main"
                }
                new_ids = {(job["dataset"], job["id"]) for job in jobs if job["method"] == methods[0]}
                if cover_ids != new_ids:
                    raise SystemExit(f"seed-7 ids diverged from cover_predictions: {len(cover_ids ^ new_ids)}")
                print("seed-7 ids match the earlier test split", flush=True)

    cache = rx.load_cache(cache_path)
    rx.generate_missing(
        [job["prompt"] for job in jobs],
        cache,
        batch_size=args.batch_size,
        model_name=args.model,
        cache_path=cache_path,
    )
    records = []
    for job in jobs:
        gen = cache[rx.prompt_key(job["prompt"])]
        pred = rx.extract_answer(gen["text"])
        em, f1 = rx.best_em_f1(pred, job["answers"])
        tokens = float(gen["prompt_tokens"] + gen["gen_tokens"])
        records.append(
            {
                "dataset": job["dataset"],
                "split": job["split"],
                "method": job["method"],
                "id": job["id"],
                "type": job["type"],
                "pred": pred,
                "gold": job["answers"][0],
                "em": em,
                "f1": f1,
                "support": job["support"],
                "tokens": tokens,
                "latency": float(gen["latency"]),
                "size": job["size"],
                "revised": 0.0,
                "n_extra": job["n_extra"],
                "agents": job["agents"],
                "titles": job["titles"],
                "cites": job.get("cites", []),
            }
        )

    grouped: dict[tuple[str, str], list] = defaultdict(list)
    for row in records:
        grouped[(row["dataset"], row["method"])].append(row)
    summary: dict = {}
    for (dataset, method), rows in sorted(grouped.items()):
        block = rx.summarize(rows)
        block["n_extra"] = float(np.mean([r["n_extra"] for r in rows]))
        if method == "follow":
            cites = [c for r in rows for c in r["cites"]]
            gold_n = sum(1 for c in cites if c["gold"])
            longer_n = sum(1 for c in cites if c["longer_title_hit"])
            missing_citer = sum(1 for c in cites if not c["citer"])
            block["cite_n"] = len(cites)
            block["cite_gold_precision"] = gold_n / len(cites) if cites else None
            block["cite_longer_title_rate"] = longer_n / len(cites) if cites else None
            block["cite_missing_citer"] = missing_citer
        summary.setdefault(dataset, {})[method] = block

    for dataset, methods_here in summary.items():
        if "follow" in methods_here and (dataset, "dense4") in grouped:
            methods_here["follow"]["versus_dense4"] = paired(
                grouped[(dataset, "follow")], grouped[(dataset, "dense4")]
            )
        if "anchor" in methods_here and (dataset, "dense4") in grouped:
            methods_here["anchor"]["versus_dense4"] = paired(
                grouped[(dataset, "anchor")], grouped[(dataset, "dense4")]
            )
        by_type: dict[str, dict] = {}
        types = sorted({r["type"] for r in grouped[(dataset, next(iter(methods_here)))]})
        for kind in types:
            by_type[kind] = {}
            for method in methods_here:
                subset = [r for r in grouped[(dataset, method)] if r["type"] == kind]
                if not subset:
                    continue
                item = rx.summarize(subset)
                item["n_extra"] = float(np.mean([r["n_extra"] for r in subset]))
                by_type[kind][method] = item
        methods_here["by_type"] = by_type

    # by_type was inserted inside each dataset dict alongside methods, which makes
    # iteration messy. Keep it, and also a sibling key.
    out = {
        "config": {
            "tag": args.tag,
            "seed": args.seed,
            "model": args.model,
            "methods": methods,
            "rule": rule["rule"],
            "disjoint_from_seed7": args.seed != 7,
            "utility": "f1 - tokens/4000",
            "hparams": "code/results/follow_hparams.json",
        },
        "summary": summary,
    }
    out_path = rx.RESULT_DIR / f"{args.tag}_summary.json"
    pred_path = rx.RESULT_DIR / f"{args.tag}_predictions.jsonl"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    with pred_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    brief = {}
    for dataset, block in summary.items():
        brief[dataset] = {
            method: {
                "n": block[method]["n"],
                "em": round(block[method]["em"], 3),
                "f1": round(block[method]["f1"], 3),
                "f1_se": round(block[method]["f1_se"], 3),
                "support": round(block[method]["support"], 3),
                "tokens": round(block[method]["tokens"]),
                "utility": round(block[method]["utility"], 3),
                "n_extra": round(block[method]["n_extra"], 3),
            }
            for method in block
            if method != "by_type"
        }
    print(json.dumps(brief, indent=2), flush=True)
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
