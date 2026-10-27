"""Calibration-only comparison of anchor fill against mention following."""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import run_experiment as rx

rx.COVER_BETA = 0.5


def support(ex, pages):
    gold = set(ex["gold_titles"])
    if not gold:
        return 0.0
    got = {p["title"] for p in pages}
    return len(gold & got) / len(gold)


def main():
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda:0")
    rules = ("anchor", "anchor_term", "anchor_nocite", "mention")
    table = {rule: {} for rule in rules}
    for ds in ("hotpot", "2wiki", "musique"):
        print("dataset", ds, flush=True)
        cal, _test = rx.load_examples(ds, 200, 300, 7)
        del _test
        rx.attach_embeddings(cal, encoder)
        buckets = {rule: [] for rule in rules}
        for ex in cal:
            offers = rx.build_offers(ex, rx.MAIN_AGENTS)
            dense_pages = [rx._page_of(ex, idx, "dense") for idx in rx._dense_order(ex)]
            dense_titles = {p["title"] for p in dense_pages}
            base = support(ex, dense_pages)
            specs = {
                "anchor": rx.select_anchor(ex, offers, True, False)[1],
                "anchor_term": rx.select_anchor(ex, offers, True, True)[1],
                "anchor_nocite": rx.select_anchor(ex, offers, False, False)[1],
                "mention": rx.mention_pages(ex)[1],
            }
            gold = set(ex["gold_titles"])
            for key, pages in specs.items():
                titles = {p["title"] for p in pages}
                added = titles - dense_titles
                buckets[key].append(
                    (support(ex, pages), base, int(titles != dense_titles), len(added & gold), len(added))
                )
        for key in rules:
            table[key][ds] = buckets[key]
    for key, rows in table.items():
        print(f"\n== {key}")
        gains = []
        for ds, items in rows.items():
            sup = sum(r[0] for r in items) / len(items)
            base = sum(r[1] for r in items) / len(items)
            changed = sum(r[2] for r in items) / len(items)
            gold_in = sum(r[3] for r in items)
            swaps = sum(r[4] for r in items)
            prec = gold_in / swaps if swaps else float("nan")
            gains.append(sup - base)
            print(
                f"  {ds:8} support {sup:.4f} dense {base:.4f} gain {sup - base:+.4f} "
                f"changed {changed:.3f} gold_prec {prec:.3f} added {swaps}"
            )
        print(f"  min_gain {min(gains):+.4f} mean_gain {sum(gains) / len(gains):+.4f}")


if __name__ == "__main__":
    main()
