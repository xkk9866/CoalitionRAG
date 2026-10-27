"""Calibration-only: which named-title policy raises support on every dataset."""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import run_experiment as rx

rx.COVER_BETA = 0.5
POLICIES = ("dense", "bm25", "bridge", "term", "vote", "from1")


def support(ex, pages):
    gold = set(ex["gold_titles"])
    if not gold or not pages:
        return 0.0
    got = {p["title"] for p in pages}
    return len(gold & got) / len(gold)


def main():
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda:0")
    table = {p: {} for p in POLICIES}
    for ds in ("hotpot", "2wiki", "musique"):
        print("dataset", ds, flush=True)
        cal, _test = rx.load_examples(ds, 200, 300, 7)
        del _test
        rx.attach_embeddings(cal, encoder)
        buckets = {p: [] for p in POLICIES}
        for ex in cal:
            dense_pages = [rx._page_of(ex, idx, "dense") for idx in rx._dense_order(ex)]
            base = support(ex, dense_pages)
            dense_titles = {p["title"] for p in dense_pages}
            gold = set(ex["gold_titles"])
            for policy in POLICIES:
                pages = rx.select_follow(ex, policy)[1]
                titles = {p["title"] for p in pages}
                added = titles - dense_titles
                buckets[policy].append(
                    (support(ex, pages), base, int(titles != dense_titles), len(added & gold), len(added))
                )
        for policy in POLICIES:
            table[policy][ds] = buckets[policy]
    for policy, rows in table.items():
        print(f"\n== {policy}")
        gains = []
        for ds, items in rows.items():
            sup = sum(r[0] for r in items) / len(items)
            base = sum(r[1] for r in items) / len(items)
            changed = sum(r[2] for r in items) / len(items)
            gold_in = sum(r[3] for r in items)
            added = sum(r[4] for r in items)
            prec = gold_in / added if added else float("nan")
            gains.append(sup - base)
            print(
                f"  {ds:8} support {sup:.4f} dense4 {base:.4f} gain {sup - base:+.4f} "
                f"changed {changed:.3f} gold_prec {prec:.3f} added {added}"
            )
        print(f"  min_gain {min(gains):+.4f} mean_gain {sum(gains) / len(gains):+.4f}")


if __name__ == "__main__":
    main()
