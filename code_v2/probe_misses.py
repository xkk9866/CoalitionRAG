"""Error analysis on MuSiQue: which hop of the gold chain is missed by a retrieval run (given it is in the pool).

usage: probe_misses.py <retrieval tag> <pool file>
"""
import sys
from collections import Counter

import numpy as np
import torch

from common import DATA, RES, jload, load_bench


def hop_order(sample, index):
    out = []
    for step in sample["question_decomposition"]:
        p = sample["paragraphs"][step["paragraph_support_idx"]]
        out.append(index.get(p["title"] + "\n" + p["paragraph_text"]))
    return out


def main(tag, pool_file):
    docs, qs = load_bench("musique")
    index = {d: i for i, d in enumerate(docs)}
    raw = jload(DATA / "musique.json")
    pools = jload(RES / "pools" / f"{pool_file}_musique.json")
    runs = jload(RES / "retrieval" / f"{tag}_musique.json")
    cache = torch.load(RES / "cache" / "f_cdn_musique.pt")
    miss_pos, tot_pos = Counter(), Counter()
    unary_rank_missed, cond_rank_missed = [], []
    for s, q, pool, r, c in zip(raw, qs, pools, runs, cache):
        order = hop_order(s, index)
        H = len(order)
        top5 = set(r["top5"])
        pos = {d: i for i, d in enumerate(pool)}
        a = c["a"]
        for h, d in enumerate(order):
            if d is None or d not in pos:
                continue
            tot_pos[(H, h)] += 1
            if d not in top5:
                miss_pos[(H, h)] += 1
                j = pos[d]
                unary_rank_missed.append(int((a > a[j]).sum()))
                if h > 0 and order[h - 1] in pos:
                    i = pos[order[h - 1]]
                    row = c["C"][i].clone(); row[i] = -1e9
                    cond_rank_missed.append(int((row > row[j]).sum()))
    for H in (2, 3, 4):
        print(f"{H}-hop: " + "  ".join(f"hop{h + 1} miss {miss_pos[(H, h)]}/{tot_pos[(H, h)]}" for h in range(H)))
    print("unary rank of missed gold: median", np.median(unary_rank_missed), "share rank<5", np.mean(np.array(unary_rank_missed) < 5))
    print("conditional rank given previous gold hop: median", np.median(cond_rank_missed),
          "share rank<5", np.mean(np.array(cond_rank_missed) < 5), "n", len(cond_rank_missed))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
