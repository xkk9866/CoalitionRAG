"""Calibration-only ceiling of mention resolution.

Measures how many gold pages are named by the dense anchor, before any test
score is used to choose a ranker.
"""

from __future__ import annotations

import os

os.environ["OPEN_CUDA"] = "0"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import open_retrieve as ore
import run_open as ro


def support(keys: set[str], gold: set[str]) -> float:
    if not gold:
        return 0.0
    return len(keys & gold) / len(gold)


def mention_keys(kind: str, anchor_ids: list[int], pages: dict, titles: dict) -> set[str]:
    keys = set()
    for rowid in anchor_ids:
        page = pages.get(rowid)
        if page is None:
            continue
        mode = "links" if kind == "links" else "resolve"
        for mention in ore.page_mentions(page, titles, mode):
            target = pages.get(mention.rowid)
            if target is not None:
                keys.add(target.key)
    return keys


def main() -> None:
    rows = ro.load_hotpot()
    cal_idx, _, _ = ro.split_indices(rows)
    chosen = [rows[i] for i in cal_idx]
    db = ore.WikiDB()
    encoder = ro.Encoder()
    try:
        bm25_lists = db.search_many([row["question"] for row in chosen], ore.FIRST_HOP_K, workers=8)
        needed = [rowid for ids in bm25_lists for rowid in ids]
        pages = db.fetch(needed)
        encoder.encode_pages(needed, pages)
        queries = encoder.encode_queries([(row["id"], row["question"]) for row in chosen])
        dense_lists = [
            ore.order_by_dense(ids, queries[row["id"]], encoder.cache)
            for row, ids in zip(chosen, bm25_lists)
        ]
        extra: list[int] = []
        for dense in dense_lists:
            anchor = dense[: ore.ANCHOR_K]
            extra.extend(ore.mention_rowids("resolve", "", anchor, pages, db.titles))
        missing = [rowid for rowid in dict.fromkeys(extra) if rowid not in pages]
        if missing:
            pages.update(db.fetch(missing))
        sums = {name: 0.0 for name in ("dense4", "link_oracle", "resolve_oracle", "link_pool", "resolve_pool")}
        named_out = {"link": 0, "resolve": 0, "outside": 0, "gold": 0}
        for row, dense in zip(chosen, dense_lists):
            gold = set(row["gold_keys"])
            named_out["gold"] += len(gold)
            anchor = [rowid for rowid in dense[: ore.ANCHOR_K] if rowid in pages]
            dense_keys = {pages[rowid].key for rowid in dense[: ore.BUDGET] if rowid in pages}
            bm25_keys = {pages[rowid].key for rowid in dense if rowid in pages}
            outside = [key for key in gold if key not in bm25_keys]
            named_out["outside"] += len(outside)
            link_keys = mention_keys("links", anchor, pages, db.titles)
            resolve_keys = mention_keys("resolve", anchor, pages, db.titles)
            named_out["link"] += len([key for key in outside if key in link_keys])
            named_out["resolve"] += len([key for key in outside if key in resolve_keys])
            # Oracle fills the two open slots with gold mentions, then dense pages.
            def oracle(pool: set[str]) -> set[str]:
                chosen_keys = set(list(dense_keys)[: ore.ANCHOR_K]) if False else {pages[i].key for i in anchor}
                for key in gold:
                    if len(chosen_keys) >= ore.BUDGET:
                        break
                    if key in pool and key not in chosen_keys:
                        chosen_keys.add(key)
                for key in dense_keys:
                    if len(chosen_keys) >= ore.BUDGET:
                        break
                    chosen_keys.add(key)
                return chosen_keys

            sums["dense4"] += support(dense_keys, gold)
            sums["link_oracle"] += support(oracle(link_keys), gold)
            sums["resolve_oracle"] += support(oracle(resolve_keys), gold)
            sums["link_pool"] += support(dense_keys | link_keys, gold)
            sums["resolve_pool"] += support(dense_keys | resolve_keys, gold)
        n = len(chosen)
        print(f"n {n}", flush=True)
        for name, total in sums.items():
            print(f"{name} {total / n:.4f}", flush=True)
        print(
            f"outside {named_out['outside']} link_hit {named_out['link']} "
            f"resolve_hit {named_out['resolve']} gold {named_out['gold']}",
            flush=True,
        )
    finally:
        encoder.close()
        db.close()


if __name__ == "__main__":
    main()
