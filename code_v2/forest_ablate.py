"""Inference ablations on cached edge logits of one trained network (no new forward passes).

  forest   : exact marginals m = A^-1 w_q (the method)
  unary    : top-k by the question edge alone, sigma(s_q)  (same network, reasons ignored)
  onestep  : one propagation step from the unary marginals (reasons used, chains of length > 2 ignored)
  beam     : beam search over reason chains with the forest's local reason probabilities (path decoding,
             as in PathRetriever / Beam Retrieval); the k-set is the union of the best chains
  cover    : CDN coverage game on sigma(s_q), sigma(s_ij)
usage: forest_ablate.py <cache tag> <pool file> [--cdn]   (--cdn: cache holds CDN 'a','C' keys)
"""
import argparse
import itertools

import numpy as np
import torch

from common import DATASETS, RES, jdump, jload, load_bench, recall_at
from forest import incoming, laplacian, marginals


def unary(sq, S):
    return torch.sigmoid(sq.double())


def onestep(sq, S):
    null, wq, W, _ = incoming(sq.double(), S.double())
    m0 = wq / (null + wq)
    return (wq + W.T @ m0) / (null + wq + W.sum(0))


def beam_sets(sq, S, k, width=5, depth=4):
    """Chains q -> a -> b ...; score = sum of log local reason probabilities w(parent->j)/d_j."""
    null, wq, W, _ = incoming(sq.double(), S.double())
    d = null + wq + W.sum(0)
    lq, lW = (wq / d).log(), (W / d[None, :]).log()
    n = sq.shape[0]
    beams = [((j,), float(lq[j])) for j in range(n)]
    beams = sorted(beams, key=lambda x: -x[1])[:width]
    done = list(beams)
    for _ in range(depth - 1):
        cand = []
        for path, sc in beams:
            for j in range(n):
                if j not in path:
                    cand.append((path + (j,), sc + float(lW[path[-1], j])))
        beams = sorted(cand, key=lambda x: -x[1])[:width]
        done += beams
    # nodes ordered by the best chain that contains them (chains ranked by score)
    out = []
    for path, _ in sorted(done, key=lambda x: -x[1]):
        for j in path:
            if j not in out:
                out.append(j)
        if len(out) >= k:
            break
    return out[:k]


_C = {}


def cover_set(sq, S, k):
    n = sq.shape[0]
    if (n, k) not in _C:
        _C[(n, k)] = torch.tensor(list(itertools.combinations(range(n), k)), device=sq.device)
    cb = _C[(n, k)]
    pa, pc = torch.sigmoid(sq), torch.sigmoid(S).clone()
    pc.fill_diagonal_(-1)
    sub = pc[cb[:, :, None], cb[:, None, :]]
    v = torch.maximum(sub.max(1).values, pa[cb]).sum(-1)
    return cb[v.argmax()].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("pool_file")
    ap.add_argument("--cdn", action="store_true")
    ap.add_argument("--methods", default="forest,unary,onestep,beam,cover")
    ap.add_argument("--save", default="", help="write retrieval files for these methods (comma list)")
    args = ap.parse_args()
    methods = args.methods.split(",")
    save = set(filter(None, args.save.split(",")))
    res = {}
    for ds in DATASETS:
        _, qs = load_bench(ds)
        pools = jload(RES / "pools" / f"{args.pool_file}_{ds}.json")
        f = RES / "cache" / f"{args.tag}_{ds}.pt"
        if not f.exists():
            continue
        cache = torch.load(f)
        met = {m: {"R@2": [], "R@5": [], "AR@5": []} for m in methods}
        outs = {m: [] for m in methods}
        for q, pool, c in zip(qs, pools, cache):
            sq, S = (c["a"], c["C"]) if args.cdn else (c["sq"], c["S"])
            sq, S = sq.cuda().float(), S.cuda().float()
            n = sq.shape[0]
            S = S.masked_fill(torch.eye(n, dtype=torch.bool, device=S.device), float("-inf"))
            for m in methods:
                if m in ("forest", "unary", "onestep"):
                    sc = {"forest": marginals, "unary": unary, "onestep": onestep}[m](sq, S)
                    r = torch.argsort(sc, descending=True).tolist()
                    t2, t5 = r[:2], r[:5]
                elif m == "beam":
                    t5 = beam_sets(sq, S, 5); t2 = t5[:2]
                else:
                    t2, t5 = cover_set(sq, S, 2), cover_set(sq, S, 5)
                t2, t5 = [pool[i] for i in t2], [pool[i] for i in t5]
                r5 = recall_at(t5, q["gold"])
                met[m]["R@2"].append(recall_at(t2, q["gold"])); met[m]["R@5"].append(r5); met[m]["AR@5"].append(float(r5 == 1))
                outs[m].append({"id": q["id"], "top2": t2, "top5": t5, "rank": t5})
        for m in methods:
            res.setdefault(m, {})[ds] = {k: round(100 * float(np.mean(v)), 1) for k, v in met[m].items()}
            res[m][ds]["n"] = len(cache)
            if m in save:
                jdump(outs[m], RES / "retrieval" / f"{args.tag}-{m}_{ds}.json")
        print(ds, len(cache), {m: res[m][ds] for m in methods}, flush=True)
    jdump(res, RES / "retrieval" / f"{args.tag}_ablate_summary.json")


if __name__ == "__main__":
    main()
