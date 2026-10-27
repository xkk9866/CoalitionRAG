"""Bayes decision for complete evidence: argmax_{|S|=k} P(E ⊆ S) under the evidence forest.

P(E ⊆ S) = sum_{G ⊆ S} P(E = G), each term a ratio of determinants (Theorem 1). Exploratory: search restricted to
subsets of the top-m marginals.
usage: cover_prob.py <cache tag> <pool file> [m]
"""
import itertools
import sys

import numpy as np
import torch

from common import DATASETS, RES, jload, load_bench, recall_at
from forest import incoming, laplacian, marginals


def logp_sets(sq, S, cand, kmax):
    """log P(E = G) for every G ⊆ cand with |G| <= kmax."""
    null, wq, W, _ = incoming(sq.double(), S.double())
    n = sq.shape[0]
    logZ = torch.logdet(laplacian(null + wq, W))
    out = {}
    for r in range(kmax + 1):
        for G in itertools.combinations(cand, r):
            g = torch.zeros(n, dtype=torch.bool)
            g[list(G)] = True
            Gi, Ni = g.nonzero().squeeze(1), (~g).nonzero().squeeze(1)
            v = -logZ
            if len(Gi):
                v = v + torch.logdet(laplacian(wq, W, Gi))
            v = v + torch.logdet(laplacian(null, W, Ni))
            out[G] = float(v)
    return out


def main(tag, pool_file, m=10, k=5):
    for ds in DATASETS:
        _, qs = load_bench(ds)
        pools = jload(RES / "pools" / f"{pool_file}_{ds}.json")
        cache = torch.load(RES / "cache" / f"{tag}_{ds}.pt")
        res = {"marg": [], "cover": []}
        pe = {"marg": [], "cover": []}
        for q, pool, c in zip(qs, pools, cache):
            sq, S = c["sq"].double(), c["S"].double()
            mg = marginals(sq, S)
            order = torch.argsort(mg, descending=True).tolist()
            cand = sorted(order[:m])
            lp = logp_sets(sq, S, cand, k)
            best, bv = None, -1.0
            for Sset in itertools.combinations(cand, k):
                v = sum(np.exp(lp[G]) for r in range(k + 1) for G in itertools.combinations(Sset, r))
                if v > bv:
                    best, bv = list(Sset), v
            top = sorted(order[:k])
            vtop = sum(np.exp(lp[G]) for r in range(k + 1) for G in itertools.combinations(top, r))
            for name, sel, v in (("marg", top, vtop), ("cover", best, bv)):
                r5 = recall_at([pool[i] for i in sel], q["gold"])
                res[name].append(r5); pe[name].append(v)
        for name in res:
            r = np.array(res[name])
            print(ds, name, f"R@5 {100 * r.mean():.1f} AR@5 {100 * (r == 1).mean():.1f} mean P(E⊆S) {np.mean(pe[name]):.3f}",
                  flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 10)
