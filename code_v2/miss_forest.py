"""Why does the forest miss in-pool MuSiQue gold passages? For each missed gold passage j (in the pool, not in the
top-5 by marginal) reports its hop position, its marginal rank, whether its annotated reason is in the top-5, and how
strongly the network scores the annotated reason edge relative to all other reasons of j.
usage: miss_forest.py <cache tag> <retrieval tag> [n]
"""
import sys
from collections import Counter

import numpy as np
import torch

from common import RES, jload, load_bench, DATA
from forest import incoming, marginals
from forest_structure import Q, musique_parents


def main(tag, rtag, n=1000):
    docs, qs = load_bench("musique")
    raw = jload(DATA / "musique.json")
    index = {d: i for i, d in enumerate(docs)}
    parents = musique_parents(raw, index)
    pools = jload(RES / "pools" / "exp20_s5_k8_musique.json")
    runs = jload(RES / "retrieval" / f"{rtag}_musique.json")
    cache = torch.load(RES / "cache" / f"{tag}_musique.pt")
    n = min(n, len(cache))
    stat = Counter()
    rk_m, rk_edge, share_edge, nulls = [], [], [], []
    for qi in range(n):
        q, pool, c, par = qs[qi], pools[qi], cache[qi], parents[qi]
        pos = {d: i for i, d in enumerate(pool)}
        top5 = set(runs[qi]["top5"])
        sq, S = c["sq"].double(), c["S"].double()
        m = marginals(sq, S)
        null, wq, W, _ = incoming(sq, S)
        d = null + wq + W.sum(0)
        H = len(q["gold"])
        for g in q["gold"]:
            if g not in pos:
                stat["out_of_pool"] += 1
                continue
            j = pos[g]
            rp = par.get(g, {Q})
            kind = "q" if rp == {Q} else "chain"
            stat[f"{kind}_total"] += 1
            if g in top5:
                continue
            stat[f"{kind}_missed"] += 1
            rk_m.append(int((m > m[j]).sum()))
            nulls.append(float(null[j] / d[j]))
            if kind == "chain":
                ps = [pos[p] for p in rp if p in pos]
                if not ps:
                    stat["chain_missed_parent_out_of_pool"] += 1
                    continue
                i = ps[0]
                stat["chain_missed_parent_in_top5" if pool[i] in top5 else "chain_missed_parent_not_top5"] += 1
                col = S[:, j].clone(); col[j] = -1e9
                rk_edge.append(int((col > col[i]).sum()))
                share_edge.append(float(W[i, j] / d[j]))
    print(dict(stat))
    print("marginal rank of missed gold: median", np.median(rk_m), "quartiles", np.percentile(rk_m, [25, 75]))
    print("rank of annotated reason among j's passage reasons: median", np.median(rk_edge),
          "share top1", np.mean(np.array(rk_edge) == 0), "share top3", np.mean(np.array(rk_edge) < 3))
    print("local prob of annotated reason w(i->j)/d_j: median", np.median(share_edge))
    print("local null prob of missed gold: median", np.median(nulls))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1000)
