"""Mainstream dense and cross-encoder baselines on the same observation.

Each model reranks the BM25 top 100 and keeps four pages. No hyperlinks.
E5 uses the released query and passage prefixes. BGE-large uses the released
query instruction. The MS MARCO MiniLM cross-encoder is the BEIR reranker.
Results are written per model and do not touch the earlier prediction files.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import traceback

import faulthandler

faulthandler.enable()

os.environ.setdefault("OPEN_CUDA", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

import open_retrieve as ore
import run_action as act
import run_fair
import run_open as ro

QUERY_BGE = "Represent this sentence for searching relevant passages: "


def safe_lists(db: ore.WikiDB, questions: list[str]) -> list[list[int]]:
    """FTS top 100. A very wide OR query can exhaust memory; shorten that one query."""
    found = []
    for question in questions:
        lists = None
        for limit in (32, 12, 6):
            try:
                rows = db.con.execute(
                    "SELECT rowid FROM pages WHERE pages MATCH ? ORDER BY rank LIMIT ?",
                    (short_query(question, limit), ore.FIRST_HOP_K),
                ).fetchall()
                lists = [int(row[0]) for row in rows]
                break
            except (MemoryError, Exception) as exc:
                if type(exc).__name__ not in {"MemoryError", "OperationalError"}:
                    raise
                print(f"fts retry limit {limit} {type(exc).__name__}", flush=True)
                lists = []
                try:
                    db.con.close()
                except Exception:
                    pass
                db.con = sqlite3.connect(ore.DB_PATH)
                db.con.execute("PRAGMA query_only=ON")
                db.con.row_factory = sqlite3.Row
        found.append(lists or [])
    return found


def short_query(question: str, limit: int) -> str:
    toks = []
    for tok in ore.content_tokens(question):
        if tok.isalnum():
            toks.append(tok)
        if len(toks) >= limit:
            break
    if not toks:
        toks = ["history"]
    return " OR ".join(dict.fromkeys(toks))


def support_of(ids: list[int], pages: dict[int, ore.Page], gold: set[str]) -> float:
    keys = {pages[rowid].key for rowid in ids if rowid in pages}
    return act.support(keys, gold)


def dense_arm(rows, db, model_name: str, query_fn, passage_fn, tag: str) -> None:
    from sentence_transformers import SentenceTransformer

    path = ro.RESULT_DIR / f"mainstream_{tag}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    pending = [row for row in rows if row["id"] not in done]
    print(f"{tag} pending {len(pending)} done {len(done)}", flush=True)
    model = SentenceTransformer(model_name, device="cuda:0")
    try:
        with path.open("a", encoding="utf-8") as handle:
            for start in range(0, len(pending), 20):
                batch = pending[start : start + 20]
                t0 = time.perf_counter()
                lists = safe_lists(db, [row["question"] for row in batch])
                pages = db.fetch([rowid for ids in lists for rowid in ids])
                flat_ids = []
                flat_text = []
                owners = []
                for row, ids in zip(batch, lists):
                    for rowid in ids:
                        page = pages.get(rowid)
                        if page is None:
                            continue
                        flat_ids.append((row["id"], rowid))
                        flat_text.append(passage_fn(page))
                        owners.append(row["id"])
                vectors = model.encode(flat_text, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
                queries = model.encode(
                    [query_fn(row["question"]) for row in batch],
                    batch_size=32,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                grouped: dict[str, list[tuple[int, np.ndarray]]] = {}
                for (qid, rowid), vec in zip(flat_ids, vectors):
                    grouped.setdefault(qid, []).append((rowid, np.asarray(vec, dtype=np.float32)))
                qmap = {row["id"]: np.asarray(vec, dtype=np.float32) for row, vec in zip(batch, queries)}
                for row in batch:
                    gold = set(row["gold_keys"])
                    pairs = grouped.get(row["id"], [])
                    if not pairs:
                        chosen = []
                    else:
                        mat = np.stack([vec for _, vec in pairs])
                        order = np.argsort(-(mat @ qmap[row["id"]]))
                        chosen = []
                        seen = set()
                        for index in order:
                            rowid = pairs[int(index)][0]
                            if rowid in seen:
                                continue
                            seen.add(rowid)
                            chosen.append(rowid)
                            if len(chosen) >= ore.BUDGET:
                                break
                    record = {
                        "id": row["id"],
                        "type": row["type"],
                        tag: support_of(chosen, pages, gold),
                        tag + "_ids": chosen,
                    }
                    handle.write(json.dumps(record) + "\n")
                handle.flush()
                print(f"{tag} {start + len(batch)} / {len(pending)} sec {time.perf_counter() - t0:.1f}", flush=True)
    finally:
        del model


def cross_arm(rows, db) -> None:
    from sentence_transformers import CrossEncoder

    tag = "minilm"
    path = ro.RESULT_DIR / f"mainstream_{tag}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    pending = [row for row in rows if row["id"] not in done]
    print(f"{tag} pending {len(pending)} done {len(done)}", flush=True)
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device="cuda:0", max_length=512)
    try:
        with path.open("a", encoding="utf-8") as handle:
            for start in range(0, len(pending), 20):
                batch = pending[start : start + 20]
                t0 = time.perf_counter()
                lists = safe_lists(db, [row["question"] for row in batch])
                pages = db.fetch([rowid for ids in lists for rowid in ids])
                pairs = []
                owners = []
                for row, ids in zip(batch, lists):
                    for rowid in ids:
                        page = pages.get(rowid)
                        if page is None:
                            continue
                        pairs.append((row["question"], page.title + ". " + page.text))
                        owners.append((row["id"], rowid))
                values = model.predict(pairs, batch_size=64, show_progress_bar=False)
                grouped: dict[str, list[tuple[int, float]]] = {}
                for (qid, rowid), value in zip(owners, values):
                    grouped.setdefault(qid, []).append((rowid, float(value)))
                for row in batch:
                    gold = set(row["gold_keys"])
                    ranked = sorted(grouped.get(row["id"], []), key=lambda item: item[1], reverse=True)
                    chosen = []
                    seen = set()
                    for rowid, _value in ranked:
                        if rowid in seen:
                            continue
                        seen.add(rowid)
                        chosen.append(rowid)
                        if len(chosen) >= ore.BUDGET:
                            break
                    handle.write(json.dumps({
                        "id": row["id"],
                        "type": row["type"],
                        tag: support_of(chosen, pages, gold),
                        tag + "_ids": chosen,
                    }) + "\n")
                handle.flush()
                print(f"{tag} {start + len(batch)} / {len(pending)} sec {time.perf_counter() - t0:.1f}", flush=True)
    finally:
        del model


def summarize(tag: str, rows: list[dict]) -> None:
    saved = {
        json.loads(line)["id"]: json.loads(line)
        for line in (ro.RESULT_DIR / f"mainstream_{tag}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    fair = {
        json.loads(line)["id"]: json.loads(line)
        for line in (ro.RESULT_DIR / "fair_retrieve_predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    scores = []
    types = []
    versus = {"flat": [], "ace": []}
    for row in rows:
        record = saved[row["id"]]
        scores.append(float(record[tag]))
        types.append(row["type"])
        base = fair.get(row["id"])
        if base is not None:
            versus["flat"].append(float(record[tag]) - float(base["flat"]))
            versus["ace"].append(float(record[tag]) - float(base["ace"]))
    def pack(values: list[float]) -> dict:
        arr = np.asarray(values, dtype=np.float64)
        return {
            "n": int(arr.size),
            "delta": float(arr.mean()),
            "se": float(arr.std(ddof=1) / np.sqrt(arr.size)),
        }
    by_type = {}
    for kind in sorted(set(types)):
        group = [score for score, value in zip(scores, types) if value == kind]
        by_type[kind] = {"n": len(group), "support": float(np.mean(group))}
    report = {
        "tag": tag,
        "n": len(scores),
        "support": float(np.mean(scores)),
        "complete": float(np.mean([score == 1.0 for score in scores])),
        "by_type": by_type,
        "versus_flat": pack(versus["flat"]),
        "versus_ace": pack(versus["ace"]),
    }
    (ro.RESULT_DIR / f"mainstream_{tag}_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(tag, json.dumps(report), flush=True)


def main() -> None:
    from sentence_transformers import SentenceTransformer  # load torch before the title table
    del SentenceTransformer
    rows = run_fair.hotpot_rows("test")
    db = ore.WikiDB()
    try:
        try:
            dense_arm(
                rows,
                db,
                "intfloat/e5-base-v2",
                lambda question: "query: " + question,
                lambda page: "passage: " + page.title + ". " + page.text,
                "e5",
            )
            summarize("e5", rows)
            dense_arm(
                rows,
                db,
                "BAAI/bge-large-en-v1.5",
                lambda question: QUERY_BGE + question,
                lambda page: page.title + ". " + page.text,
                "bge_large",
            )
            summarize("bge_large", rows)
            cross_arm(rows, db)
            summarize("minilm", rows)
        except Exception:
            traceback.print_exc()
            raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
