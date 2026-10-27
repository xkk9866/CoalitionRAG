"""Open-corpus retrieval on HotpotQA's Wikipedia abstracts.

The resolution rule is fixed before any test score. Calibration chooses only
whether a second resolution round clears a predeclared support gain.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Physical GPU 0 is already full. OPEN_CUDA selects the device this process sees as cuda:0.
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("OPEN_CUDA", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import open_retrieve as ore
import run_experiment as rx
from run_follow import stratified_remaining

RESULT_DIR = rx.RESULT_DIR
HPARAMS = RESULT_DIR / "open_hparams.json"
CAL_PATH = RESULT_DIR / "open_cal.json"

MAIN_METHOD = "resolve"
# Channels reported as ablations. The test slice does not choose among them.
RETRIEVAL_METHODS = [
    "dense",
    "bm25",
    "rrf",
    "d2b2",
    "expand",
    "rawtitle",
    "namedrand",
    "namedbm25",
    "links",
    "lexicon",
    "noterm",
    "resolve",
]
MENTION_METHODS = {"rawtitle", "namedrand", "namedbm25", "links", "lexicon", "noterm", "resolve"}


def load_hotpot() -> list[dict]:
    from datasets import load_dataset

    raw = load_dataset("hotpot_qa", "distractor", split="validation")
    rows = []
    for i in range(len(raw)):
        row = raw[i]
        gold = list(dict.fromkeys(row["supporting_facts"]["title"]))
        rows.append(
            {
                "index": i,
                "id": row["id"],
                "question": row["question"],
                "answers": [row["answer"]],
                "type": row["type"],
                "gold_titles": gold,
                "gold_keys": [ore.title_key(title) for title in gold if ore.title_key(title)],
            }
        )
    return rows


def split_indices(rows: list[dict]) -> tuple[list[int], list[int], list[int]]:
    types = [row["type"] for row in rows]
    cal, test = rx.stratified_indices(types, 200, 300, 7)
    blocked = set(cal + test)
    second = stratified_remaining(types, blocked, 300, 11)
    return cal, test, second


class Encoder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda:0")
        self.cache: dict[int, np.ndarray] = {}

    def encode_pages(self, ids: list[int], pages: dict[int, ore.Page]) -> None:
        missing = [rowid for rowid in dict.fromkeys(ids) if rowid not in self.cache and rowid in pages]
        if not missing:
            return
        batch = 128
        for start in range(0, len(missing), batch):
            chunk = missing[start : start + batch]
            texts = [pages[rowid].title + ". " + pages[rowid].text for rowid in chunk]
            vecs = self.model.encode(
                texts,
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for rowid, vec in zip(chunk, vecs):
                self.cache[rowid] = np.asarray(vec, dtype=np.float32)
            if start == 0 or (start // batch) % 20 == 0:
                print(f"encoded {min(start + batch, len(missing))}/{len(missing)}", flush=True)

    def encode_queries(self, pairs: list[tuple[str, str]]) -> dict[str, np.ndarray]:
        if not pairs:
            return {}
        texts = [text for _, text in pairs]
        vecs = self.model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return {key: np.asarray(vec, dtype=np.float32) for (key, _), vec in zip(pairs, vecs)}

    def close(self) -> None:
        del self.model
        self.cache.clear()


def _mean_se(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return float(arr.mean()), se


def _paired(left: list[float], right: list[float]) -> dict:
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
    changed = diff != 0
    out = {
        "n": int(len(diff)),
        "delta": float(diff.mean()) if len(diff) else 0.0,
        "se": se,
        "changed_n": int(changed.sum()),
    }
    if changed.any():
        sub = diff[changed]
        sub_se = float(sub.std(ddof=1) / np.sqrt(len(sub))) if len(sub) > 1 else 0.0
        out["changed_delta"] = float(sub.mean())
        out["changed_se"] = sub_se
    return out


def retrieve_slice(
    rows: list[dict],
    db: ore.WikiDB,
    encoder: Encoder,
    rounds: int,
    methods: list[str],
    require_term: bool = True,
) -> list[dict]:
    # A full validation complement is several thousand questions. Encoding every
    # candidate at once holds hundreds of thousands of vectors, so the slice is
    # scored in batches. The selection rule does not depend on other questions.
    if len(rows) > 200:
        records = []
        for start in range(0, len(rows), 200):
            batch = rows[start : start + 200]
            print(f"batch {start}:{start + len(batch)} / {len(rows)}", flush=True)
            records.extend(
                retrieve_slice(batch, db, encoder, rounds, methods, require_term)
            )
            encoder.cache.clear()
        return records
    started = time.perf_counter()
    print(f"bm25 search n={len(rows)}", flush=True)
    bm25_lists = db.search_many([row["question"] for row in rows], ore.FIRST_HOP_K, workers=8)
    print(f"bm25 done {time.perf_counter() - started:.1f}s", flush=True)
    needed = [rowid for ids in bm25_lists for rowid in ids]
    pages = db.fetch(needed)
    encoder.encode_pages(needed, pages)
    queries = encoder.encode_queries([(row["id"], row["question"]) for row in rows])
    dense_lists = []
    for row, ids in zip(rows, bm25_lists):
        dense_lists.append(ore.order_by_dense(ids, queries[row["id"]], encoder.cache))

    extra: list[int] = []
    for row, dense in zip(rows, dense_lists):
        anchor = dense[: ore.ANCHOR_K]
        for method in methods:
            if method in MENTION_METHODS:
                extra.extend(ore.mention_rowids(method, row["question"], anchor, pages, db.titles))
    missing = [rowid for rowid in dict.fromkeys(extra) if rowid not in pages]
    if missing:
        pages.update(db.fetch(missing))
        encoder.encode_pages(missing, pages)

    if rounds >= 2 and "resolve" in methods:
        extra2: list[int] = []
        for row, dense in zip(rows, dense_lists):
            packed = ore.select_from_orders(
                "resolve",
                row["question"],
                [],
                dense,
                pages,
                db.titles,
                encoder.cache,
                queries[row["id"]],
                rounds=1,
                rng_seed=ore.stable_seed(row["id"]),
                require_term=require_term,
            )
            sources = [int(key) for key in packed["origin"]]
            extra2.extend(ore.mention_rowids("resolve", row["question"], sources, pages, db.titles))
        missing2 = [rowid for rowid in dict.fromkeys(extra2) if rowid not in pages]
        if missing2:
            pages.update(db.fetch(missing2))
            encoder.encode_pages(missing2, pages)

    expand_lists: list[list[int]] = []
    if "expand" in methods:
        print("expansion search", flush=True)
        expand_needed: list[int] = []
        expand_queries_text = []
        for row, dense in zip(rows, dense_lists):
            titles = [pages[rowid].title for rowid in dense[: ore.ANCHOR_K] if rowid in pages]
            expand_queries_text.append(row["question"] + " " + " ".join(titles))
        expand_lists = db.search_many(expand_queries_text, 50, workers=8)
        for ids in expand_lists:
            expand_needed.extend(ids)
        missing_e = [rowid for rowid in dict.fromkeys(expand_needed) if rowid not in pages]
        if missing_e:
            pages.update(db.fetch(missing_e))
            encoder.encode_pages(missing_e, pages)
        expand_queries = encoder.encode_queries(
            [
                (
                    row["id"] + "::expand",
                    row["question"] + " " + " ".join(pages[i].title for i in dense[: ore.ANCHOR_K] if i in pages),
                )
                for row, dense in zip(rows, dense_lists)
            ]
        )
        expand_dense = []
        for row, ids in zip(rows, expand_lists):
            expand_dense.append(ore.order_by_dense(ids, expand_queries[row["id"] + "::expand"], encoder.cache))
    else:
        expand_dense = [[] for _ in rows]

    records = []
    for row, bm25_ids, dense_ids, second_ids in zip(rows, bm25_lists, dense_lists, expand_dense):
        q_vec = queries[row["id"]]
        hint_ids = ore.mention_rowids("resolve", row["question"], dense_ids[: ore.ANCHOR_K], pages, db.titles)
        bm25_hint = db.bm25_scores(row["question"], hint_ids) if "namedbm25" in methods else {}
        anchor_pages = [pages[i] for i in dense_ids[: ore.ANCHOR_K] if i in pages]
        named = set()
        for page in anchor_pages:
            for mention in ore.page_mentions(page, db.titles, "resolve"):
                named.add(mention.key)
        gold = set(row["gold_keys"])
        bm25_keys = {pages[i].key for i in bm25_ids if i in pages}
        outside = [key for key in row["gold_keys"] if key not in bm25_keys]
        for method in methods:
            method_rounds = rounds if method == "resolve" else 1
            if method == "expand":
                packed = ore.fill_expand(dense_ids[: ore.ANCHOR_K], second_ids, pages)
            else:
                packed = ore.select_from_orders(
                    method,
                    row["question"],
                    bm25_ids,
                    dense_ids,
                    pages,
                    db.titles,
                    encoder.cache,
                    q_vec,
                    rounds=method_rounds,
                    rng_seed=ore.stable_seed(row["id"]),
                    bm25_hint=bm25_hint,
                    require_term=require_term,
                )
            keys = packed["keys"]
            key_set = set(keys)
            dense4 = set(dense_ids[: ore.BUDGET])
            added_ids = [rowid for rowid in packed["rowids"] if rowid not in dense4]
            added_gold = sum(1 for rowid in added_ids if rowid in pages and pages[rowid].key in gold)
            records.append(
                {
                    "id": row["id"],
                    "type": row["type"],
                    "method": method,
                    "titles": packed["titles"],
                    "keys": keys,
                    "rowids": packed["rowids"],
                    "origin": packed["origin"],
                    "support": (len(key_set & gold) / len(gold)) if gold else 0.0,
                    "n_gold": len(gold),
                    "gold_in_corpus": sum(1 for key in row["gold_keys"] if key in db.titles),
                    "outside_gold": len(outside),
                    "outside_recovered": sum(1 for key in outside if key in key_set),
                    "added": len(added_ids),
                    "added_gold": added_gold,
                    "anchor_names_gold": len(named & gold),
                    "collisions": int(packed.get("collisions", 0)),
                    "link_added": sum(1 for rowid in added_ids if packed["origin"].get(str(rowid)) == "link"),
                    "title_added": sum(1 for rowid in added_ids if packed["origin"].get(str(rowid)) == "title"),
                }
            )
    print(f"retrieval records {len(records)} in {time.perf_counter() - started:.1f}s", flush=True)
    return records


def summarize_retrieval(records: list[dict]) -> dict:
    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_method[row["method"]].append(row)
    dense = {row["id"]: row for row in by_method.get("dense", [])}
    summary = {"n": len(dense), "methods": {}}
    for method, rows in by_method.items():
        support, support_se = _mean_se([row["support"] for row in rows])
        added = sum(row["added"] for row in rows)
        added_gold = sum(row["added_gold"] for row in rows)
        outside = sum(row["outside_gold"] for row in rows)
        recovered = sum(row["outside_recovered"] for row in rows)
        named = sum(row["anchor_names_gold"] for row in rows)
        gold_n = sum(row["n_gold"] for row in rows)
        in_corpus = sum(row["gold_in_corpus"] for row in rows)
        block = {
            "n": len(rows),
            "support": support,
            "support_se": support_se,
            "added_pages": added,
            "added_gold": added_gold,
            "outside_gold_pages": outside,
            "outside_recovered_pages": recovered,
            "anchor_named_gold_pages": named,
            "gold_pages": gold_n,
            "gold_pages_in_corpus": in_corpus,
            "collisions": sum(row["collisions"] for row in rows),
            "link_added": sum(row["link_added"] for row in rows),
            "title_added": sum(row["title_added"] for row in rows),
        }
        if dense and method != "dense":
            aligned_m = []
            aligned_d = []
            for row in rows:
                base = dense.get(row["id"])
                if base is None:
                    continue
                aligned_m.append(row["support"])
                aligned_d.append(base["support"])
            block["versus_dense_support"] = _paired(aligned_m, aligned_d)
        by_type: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            by_type[row["type"]].append(row["support"])
        block["by_type"] = {name: {"n": len(vals), "support": _mean_se(vals)[0], "support_se": _mean_se(vals)[1]} for name, vals in sorted(by_type.items())}
        summary["methods"][method] = block
    return summary


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_cal(rows: list[dict], cal_idx: list[int], limit: int) -> None:
    chosen = [rows[i] for i in cal_idx]
    if limit:
        chosen = chosen[:limit]
    db = ore.WikiDB()
    present = sum(sum(key in db.titles for key in row["gold_keys"]) for row in chosen)
    total = sum(len(row["gold_keys"]) for row in chosen)
    print(f"calibration gold titles in corpus {present}/{total}", flush=True)
    encoder = Encoder()
    try:
        one = retrieve_slice(chosen, db, encoder, rounds=1, methods=RETRIEVAL_METHODS)
        two_rows = [row for row in chosen]
        two = retrieve_slice(two_rows, db, encoder, rounds=2, methods=["resolve"])
    finally:
        encoder.close()
        db.close()
    for row in two:
        row["method"] = "resolve2"
    summary = summarize_retrieval(one + two)
    payload = {"n": len(chosen), "limited": bool(limit), "summary": summary}
    if limit:
        write_json(RESULT_DIR / "open_cal_debug.json", payload)
        print("limited calibration was not frozen", flush=True)
        return
    write_json(CAL_PATH, payload)
    s1 = summary["methods"]["resolve"]["support"]
    s_noterm = summary["methods"]["noterm"]["support"]
    require_term = not (s_noterm >= s1 + ore.SECOND_ROUND_MIN_GAIN)
    s2 = summary["methods"]["resolve2"]["support"]
    # resolve2 was scored with the default term filter. If calibration drops
    # the filter, the second-round comparison has to be recomputed.
    if not require_term:
        db = ore.WikiDB()
        encoder = Encoder()
        try:
            two_b = retrieve_slice(chosen, db, encoder, rounds=2, methods=["resolve"], require_term=False)
        finally:
            encoder.close()
            db.close()
        s2 = summarize_retrieval(two_b)["methods"]["resolve"]["support"]
        summary["methods"]["resolve2_noterm"] = summarize_retrieval(two_b)["methods"]["resolve"]
        write_json(CAL_PATH, {"n": len(chosen), "limited": False, "summary": summary})
    rounds = 2 if s2 >= (s_noterm if not require_term else s1) + ore.SECOND_ROUND_MIN_GAIN else 1
    kept = s_noterm if not require_term else s1
    rule = (
        "Lock the two pages with the highest dense score inside the BM25 top 100. "
        "From those pages, a mention is a Wikipedia hyperlink target or a maximal "
        "multi-word title. Unlinked single-token titles are ignored. "
    )
    if require_term:
        rule += "A mention is fetched only when its page adds a question term missing from the locked pages. "
    else:
        rule += "The question-term filter was dropped on the calibration slice. "
    rule += "Rank admitted pages by the number of new question terms, then by dense score, and keep at most two. "
    if rounds == 2:
        rule += "One further round may replace the last page when a newly kept page names a better target. "
    else:
        rule += "A second round did not clear the calibration gain, so retrieval stops after one expansion. "
    rule += "Empty slots stay with the dense order."
    hparams = {
        "main_method": MAIN_METHOD,
        "rounds": rounds,
        "require_term": require_term,
        "second_round_min_gain": ore.SECOND_ROUND_MIN_GAIN,
        "page_words": ore.PAGE_WORDS,
        "first_hop_k": ore.FIRST_HOP_K,
        "budget": ore.BUDGET,
        "anchor_k": ore.ANCHOR_K,
        "cal_n": len(chosen),
        "cal_support": {name: block["support"] for name, block in summary["methods"].items()},
        "rule": rule,
        "frozen_before_test": True,
    }
    write_json(HPARAMS, hparams)
    print(
        f"frozen rounds={rounds} require_term={require_term} kept={kept:.4f} resolve2={s2:.4f}",
        flush=True,
    )


def load_hparams() -> dict:
    if not HPARAMS.exists():
        raise SystemExit("open_hparams.json is missing; run calibration before test scoring")
    return json.loads(HPARAMS.read_text(encoding="utf-8"))


def run_retrieve(rows: list[dict], cal_idx: list[int], limit: int) -> None:
    hparams = load_hparams()
    blocked = set(cal_idx)
    chosen = [row for row in rows if row["index"] not in blocked]
    if limit:
        chosen = chosen[:limit]
    require_term = bool(hparams.get("require_term", True))
    methods = list(RETRIEVAL_METHODS)
    # Once the term filter is dropped, noterm is the same rule as resolve.
    if not require_term:
        methods = [name for name in methods if name != "noterm"]
    tag = "open_retrieve_debug" if limit else "open_retrieve"
    pred_path = RESULT_DIR / f"{tag}_predictions.jsonl"
    done: set[str] = set()
    if pred_path.exists() and not limit:
        with pred_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("method") == "dense":
                    done.add(row["id"])
    pending = [row for row in chosen if row["id"] not in done]
    print(f"retrieve pending {len(pending)} done {len(done)} methods {len(methods)}", flush=True)
    db = ore.WikiDB()
    encoder = Encoder()
    try:
        with pred_path.open("a" if done and not limit else "w", encoding="utf-8") as handle:
            for start in range(0, len(pending), 200):
                batch = pending[start : start + 200]
                print(f"retrieve batch {start}:{start + len(batch)} / {len(pending)}", flush=True)
                records = retrieve_slice(
                    batch,
                    db,
                    encoder,
                    rounds=int(hparams["rounds"]),
                    methods=methods,
                    require_term=require_term,
                )
                encoder.cache.clear()
                for row in records:
                    slim = {k: v for k, v in row.items() if k != "rowids"}
                    handle.write(json.dumps(slim, ensure_ascii=False) + "\n")
                handle.flush()
    finally:
        encoder.close()
        db.close()
    saved = []
    with pred_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                saved.append(json.loads(line))
    summary = summarize_retrieval(saved)
    summary["rounds"] = hparams["rounds"]
    summary["require_term"] = require_term
    summary["excluded_calibration"] = len(cal_idx)
    write_json(RESULT_DIR / f"{tag}_summary.json", summary)
    resolve = summary["methods"].get("resolve", {})
    dense = summary["methods"].get("dense", {})
    print(
        f"test retrieval dense={dense.get('support')} resolve={resolve.get('support')} n={summary['n']}",
        flush=True,
    )


def pages_for(db: ore.WikiDB, rowids: list[int]) -> list[ore.Page]:
    fetched = db.fetch(rowids)
    return [fetched[rowid] for rowid in rowids if rowid in fetched]


def bridge_prompt(question: str, page: ore.Page) -> str:
    return (
        "Which Wikipedia article is needed next to answer the question?\n"
        "Reply with only the article title.\n\n"
        f"Question: {question}\n"
        f"Page: {page.title}\n"
        f"{page.text}\n"
        "Title:"
    )


def build_genquery(rows, dense_lists, pages, db, generated: dict[str, str]) -> dict[str, dict]:
    out = {}
    for row, dense in zip(rows, dense_lists):
        anchor = [rowid for rowid in dense[: ore.ANCHOR_K] if rowid in pages]
        title = generated.get(row["id"], "")
        key = ore.title_key(title)
        chosen = list(anchor)
        origin = {}
        hit = db.titles.get(key)
        if hit is not None and hit not in chosen:
            chosen.append(hit)
            origin[hit] = "generated"
        extra = db.search(title or row["question"], 10)
        for rowid in extra:
            if rowid not in chosen:
                chosen.append(rowid)
                origin[rowid] = "generated"
                break
        chosen = ore._take(dense, chosen, ore.BUDGET)
        packed = ore._pack(chosen, db.fetch(chosen), origin)
        out[row["id"]] = packed
    return out


def rerank_lists(rows, bm25_lists, pages) -> dict[str, list[int]]:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda:0", max_length=256)
    orders = {}
    try:
        for row, ids in zip(rows, bm25_lists):
            use = [rowid for rowid in ids[:20] if rowid in pages]
            if not use:
                orders[row["id"]] = []
                continue
            pairs = [(row["question"], pages[rowid].title + ". " + pages[rowid].text) for rowid in use]
            scores = model.predict(pairs, batch_size=32, show_progress_bar=False)
            order = [rowid for _, rowid in sorted(zip(scores, use), key=lambda item: float(item[0]), reverse=True)]
            orders[row["id"]] = order
    finally:
        del model
    return orders


def cache_path(model_name: str, tag: str) -> Path:
    if "Qwen2.5-7B" in model_name:
        return RESULT_DIR / "gen_cache_open7b.jsonl"
    safe = tag.replace("/", "_")
    return RESULT_DIR / f"gen_cache_{safe}.jsonl"


def run_answer(rows: list[dict], indices: list[int], model_name: str, tag: str, limit: int, batch_size: int) -> None:
    hparams = load_hparams()
    chosen = [rows[i] for i in indices]
    if limit:
        chosen = chosen[:limit]
    require_term = bool(hparams.get("require_term", True))
    methods = list(RETRIEVAL_METHODS)
    if not require_term:
        methods = [name for name in methods if name != "noterm"]
    db = ore.WikiDB()
    encoder = Encoder()
    try:
        records = retrieve_slice(
            chosen,
            db,
            encoder,
            rounds=int(hparams["rounds"]),
            methods=methods,
            require_term=require_term,
        )
    finally:
        encoder.close()
        import torch

        torch.cuda.empty_cache()
    by_id_method = {(row["id"], row["method"]): row for row in records}
    # Rebuild dense anchors for the generated-title baseline from the dense records.
    dense_rows = {row["id"]: row for row in records if row["method"] == "dense"}
    jobs = []
    page_cache: dict[int, ore.Page] = {}
    needed = [rowid for row in records for rowid in row["rowids"]]
    page_cache.update(db.fetch(needed))

    hop_ids = db.search_many([row["question"] for row in chosen], ore.FIRST_HOP_K, workers=8)
    page_cache.update(db.fetch([rowid for ids in hop_ids for rowid in ids[:20]]))
    try:
        rerank = rerank_lists(chosen, hop_ids, page_cache)
    except Exception as exc:
        print(f"reranker unavailable: {exc}", flush=True)
        rerank = {}
    if rerank:
        for row in chosen:
            order = rerank.get(row["id"], [])
            packed = ore._pack(order[: ore.BUDGET], page_cache, {})
            # Pages outside the first fetch are ignored; search again only for missing ids.
            missing = [rowid for rowid in order[: ore.BUDGET] if rowid not in page_cache]
            if missing:
                page_cache.update(db.fetch(missing))
                packed = ore._pack(order[: ore.BUDGET], page_cache, {})
            gold = set(row["gold_keys"])
            keys = packed["keys"]
            records.append(
                {
                    "id": row["id"],
                    "type": row["type"],
                    "method": "rerank",
                    "titles": packed["titles"],
                    "keys": keys,
                    "rowids": packed["rowids"],
                    "origin": {},
                    "support": (len(set(keys) & gold) / len(gold)) if gold else 0.0,
                    "answers": row["answers"],
                }
            )

    bridge_jobs = []
    for row in chosen:
        dense = dense_rows[row["id"]]
        first = dense["rowids"][:1]
        page = page_cache.get(first[0]) if first else None
        if page is None:
            continue
        prompt = bridge_prompt(row["question"], page)
        bridge_jobs.append({"id": row["id"], "prompt": prompt, "answers": row["answers"], "type": row["type"]})
    cache = rx.load_cache(cache_path(model_name, tag))
    rx.generate_missing(
        [job["prompt"] for job in bridge_jobs],
        cache,
        batch_size=batch_size,
        model_name=model_name,
        cache_path=cache_path(model_name, tag),
    )
    generated = {}
    for job in bridge_jobs:
        text = cache[rx.prompt_key(job["prompt"])]["text"]
        generated[job["id"]] = rx.extract_answer(text).split("\n")[0].strip().strip('"')
    # Generated titles may point outside the pages already fetched.
    gen_packed = {}
    for row in chosen:
        dense = dense_rows[row["id"]]
        title = generated.get(row["id"], "")
        key = ore.title_key(title)
        chosen_ids = list(dense["rowids"][: ore.ANCHOR_K])
        origin = {}
        hit = db.titles.get(key)
        if hit is not None and hit not in chosen_ids:
            chosen_ids.append(hit)
            origin[hit] = "generated"
        for rowid in db.search(title or row["question"], 10):
            if rowid not in chosen_ids:
                chosen_ids.append(rowid)
                origin[rowid] = "generated"
                break
        chosen_ids = ore._take(dense["rowids"], chosen_ids, ore.BUDGET)
        fetched = db.fetch(chosen_ids)
        page_cache.update(fetched)
        packed = ore._pack(chosen_ids, page_cache, origin)
        gold = set(row["gold_keys"])
        keys = packed["keys"]
        gen_packed[row["id"]] = packed
        records.append(
            {
                "id": row["id"],
                "type": row["type"],
                "method": "genquery",
                "titles": packed["titles"],
                "keys": keys,
                "rowids": packed["rowids"],
                "origin": packed["origin"],
                "support": (len(set(keys) & gold) / len(gold)) if gold else 0.0,
                "generated_title": title,
            }
        )

    answer_methods = [row["method"] for row in records]
    answer_methods = list(dict.fromkeys(answer_methods))
    reader_prompts = []
    reader_meta = []
    for row in chosen:
        for method in answer_methods:
            packed = next(item for item in records if item["id"] == row["id"] and item["method"] == method)
            pages = [page_cache[rowid] for rowid in packed["rowids"] if rowid in page_cache]
            prompt = ore.render_pages(row["question"], pages)
            reader_prompts.append(prompt)
            reader_meta.append((row, method, packed, prompt))
    rx.generate_missing(
        reader_prompts,
        cache,
        batch_size=batch_size,
        model_name=model_name,
        cache_path=cache_path(model_name, tag),
    )
    answer_rows = []
    for row, method, packed, prompt in reader_meta:
        gen = cache[rx.prompt_key(prompt)]
        pred = rx.extract_answer(gen["text"])
        em, f1 = rx.best_em_f1(pred, row["answers"])
        answer_rows.append(
            {
                "id": row["id"],
                "type": row["type"],
                "method": method,
                "pred": pred,
                "gold": row["answers"][0],
                "em": em,
                "f1": f1,
                "support": packed["support"],
                "tokens": float(gen["prompt_tokens"] + gen["gen_tokens"]),
                "latency": float(gen["latency"]),
                "titles": packed["titles"],
                "origin": packed.get("origin", {}),
                "model": model_name,
            }
        )
    db.close()
    summary = summarize_answers(answer_rows)
    summary["rounds"] = hparams["rounds"]
    summary["model"] = model_name
    summary["n_questions"] = len(chosen)
    write_json(RESULT_DIR / f"{tag}_summary.json", summary)
    write_jsonl(RESULT_DIR / f"{tag}_predictions.jsonl", answer_rows)
    print(json.dumps({m: {"f1": summary["methods"][m]["f1"], "support": summary["methods"][m]["support"]} for m in summary["methods"] if m in {"dense", "resolve", "d2b2", "expand"}}, indent=2), flush=True)


def summarize_answers(rows: list[dict]) -> dict:
    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    dense = {row["id"]: row for row in by_method.get("dense", [])}
    summary = {"n": len(dense), "methods": {}}
    for method, group in by_method.items():
        f1, f1_se = _mean_se([row["f1"] for row in group])
        em, em_se = _mean_se([row["em"] for row in group])
        support, support_se = _mean_se([row["support"] for row in group])
        tokens, tokens_se = _mean_se([row["tokens"] for row in group])
        block = {
            "n": len(group),
            "em": em,
            "em_se": em_se,
            "f1": f1,
            "f1_se": f1_se,
            "support": support,
            "support_se": support_se,
            "tokens": tokens,
            "tokens_se": tokens_se,
        }
        if dense and method != "dense":
            mf, df, ms, ds = [], [], [], []
            for row in group:
                base = dense.get(row["id"])
                if base is None:
                    continue
                mf.append(row["f1"])
                df.append(base["f1"])
                ms.append(row["support"])
                ds.append(base["support"])
            block["versus_dense_f1"] = _paired(mf, df)
            block["versus_dense_support"] = _paired(ms, ds)
        by_type: dict[str, list[dict]] = defaultdict(list)
        for row in group:
            by_type[row["type"]].append(row)
        block["by_type"] = {}
        for name, group_rows in sorted(by_type.items()):
            tf1, tse = _mean_se([row["f1"] for row in group_rows])
            ts, tss = _mean_se([row["support"] for row in group_rows])
            block["by_type"][name] = {"n": len(group_rows), "f1": tf1, "f1_se": tse, "support": ts, "support_se": tss}
        summary["methods"][method] = block
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["cal", "retrieve", "answer"], required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tag", default="")
    parser.add_argument("--slice", choices=["test", "s11"], default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    rows = load_hotpot()
    cal_idx, test_idx, second_idx = split_indices(rows)
    if args.stage == "cal":
        run_cal(rows, cal_idx, args.limit)
        return
    if args.stage == "retrieve":
        run_retrieve(rows, cal_idx, args.limit)
        return
    tag = args.tag or ("open_s11" if args.slice == "s11" else "open")
    if args.limit and args.tag == "":
        tag = tag + "_debug"
    indices = second_idx if args.slice == "s11" else test_idx
    run_answer(rows, indices, args.model, tag, args.limit, args.batch_size)


if __name__ == "__main__":
    main()
