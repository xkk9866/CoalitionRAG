"""Calibration-only probe. Does not read the test split."""

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


def incoming(ex, pages):
    dense_titles = []
    for offer_title in ex["titles"]:
        pass
    return None


def stats(name, rows):
    gains = []
    print(f"\n== {name}")
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
            f"changed {changed:.3f} swap_gold_prec {prec:.3f} swaps {swaps}"
        )
    print(f"  min_gain {min(gains):+.4f} mean_gain {sum(gains) / len(gains):+.4f}")


def main():
    print("loading encoder", flush=True)
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda:0")
    rows = {
        "hop1": {},
        "hop2_msg": {},
        "hop2_core": {},
        "rrf": {},
        "mention": {},
    }
    for ds in ("hotpot", "2wiki", "musique"):
        print("dataset", ds, flush=True)
        cal, _test = rx.load_examples(ds, 200, 300, 7)
        del _test
        rx.attach_embeddings(cal, encoder)
        buckets = {k: [] for k in rows}
        for ex in cal:
            offers = rx.build_offers(ex, rx.MAIN_AGENTS)
            _, dense_pages = rx.select_hop(ex, offers, rounds=0)
            # rounds=0 returns the dense pack
            dense_titles = {p["title"] for p in dense_pages}
            base = support(ex, dense_pages)
            specs = {
                "hop1": rx.select_hop(ex, offers, rounds=1, message_only=True)[1],
                "hop2_msg": rx.select_hop(ex, offers, rounds=2, message_only=True)[1],
                "hop2_core": rx.select_hop(ex, offers, rounds=2, message_only=False)[1],
                "rrf": rx.rrf_pages(ex)[1],
                "mention": rx.mention_pages(ex)[1],
            }
            gold = set(ex["gold_titles"])
            for key, pages in specs.items():
                titles = {p["title"] for p in pages}
                added = titles - dense_titles
                gold_hits = len(added & gold)
                buckets[key].append(
                    (support(ex, pages), base, int(titles != dense_titles), gold_hits, len(added))
                )
        for key in rows:
            rows[key][ds] = buckets[key]
    for key, table in rows.items():
        stats(key, table)


if __name__ == "__main__":
    main()
