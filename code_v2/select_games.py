"""Exact coalition selection on cached CDN scores under alternative characteristic functions.

  add_logit : v(S) = sum a_i + sum_{i<j} b_ij  (logit dividends)
  add_prob  : same in probability space
  cover     : v(S) = sum_{j in S} max( P(j|q), max_{i in S\\j} P(j|q,p_i) )  -- each member valued under its
              best supporting partner (bounded coalition coverage)
usage: select_games.py <cache tag> <pool file> [n_test]
"""
import itertools
import sys

import numpy as np
import torch

from common import jmerge, DATASETS, RES, jdump, jload, load_bench, recall_at
from cvn import _combos


def best(value_fn, n, k, dev):
    combos = _combos(n, k, dev)
    v = value_fn(combos)
    return combos[v.argmax()].tolist()


def run(tag, pool_file, n_test):
    res = {}
    for ds in DATASETS:
        _, qs = load_bench(ds)
        pools = jload(RES / "pools" / f"{pool_file}_{ds}.json")
        if not (RES / "cache" / f"{tag}_{ds}.pt").exists():
            continue
        cache = torch.load(RES / "cache" / f"{tag}_{ds}.pt")
        n = len(cache)
        qs, pools = qs[:n], pools[:n]
        outs = {g: [] for g in ["add_logit", "add_prob", "cover"]}
        met = {g: {"R@2": [], "R@5": [], "AR@5": []} for g in outs}
        for q, pool, c in zip(qs, pools, cache):
            a, C = c["a"].cuda(), c["C"].cuda()
            m = len(pool)
            D = C - a[None, :]; D.fill_diagonal_(0); B = 0.5 * (D + D.T)
            pa, pc = torch.sigmoid(a), torch.sigmoid(C)
            Dp = pc - pa[None, :]; Dp.fill_diagonal_(0); Bp = 0.5 * (Dp + Dp.T)
            pc_nodiag = pc.clone(); pc_nodiag.fill_diagonal_(-1)

            def add(aa, BB):
                def f(cb):
                    v = aa[cb].sum(-1)
                    for x, y in itertools.combinations(range(cb.shape[1]), 2):
                        v = v + BB[cb[:, x], cb[:, y]]
                    return v
                return f

            def cover(cb):
                sub = pc_nodiag[cb[:, :, None], cb[:, None, :]]          # (M, k, k): sub[m, i, j] = P(j | p_i)
                best_ctx = sub.max(1).values                              # best supporting partner for each j
                return torch.maximum(best_ctx, pa[cb]).sum(-1)

            fns = {"add_logit": add(a, B), "add_prob": add(pa, Bp), "cover": cover}
            for g, fn in fns.items():
                s2 = best(fn, m, 2, a.device); s5 = best(fn, m, 5, a.device)
                t2, t5 = [pool[i] for i in s2], [pool[i] for i in s5]
                outs[g].append({"id": q["id"], "top2": t2, "top5": t5, "rank": t5})
                r5 = recall_at(t5, q["gold"])
                met[g]["R@2"].append(recall_at(t2, q["gold"])); met[g]["R@5"].append(r5); met[g]["AR@5"].append(float(r5 == 1))
        for g in outs:
            jdump(outs[g], RES / "retrieval" / f"{tag}-{g}_{ds}.json")
            res.setdefault(g, {})[ds] = {k: round(100 * float(np.mean(v)), 1) for k, v in met[g].items()}
        print(ds, {g: res[g][ds] for g in outs}, flush=True)
    jmerge(res, RES / "retrieval" / f"{tag}_games_summary.json")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
