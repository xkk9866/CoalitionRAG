"""Feasibility probe: evidence-forest marginals on cached conditional cross-encoder logits.

For a pool of n passages with unary logits a_j = f(q => p_j) and conditional logits C_ij = f(q+p_i => p_j),
edge weights are w_qj = exp(a_j), w_ij = exp(C_ij), and the null root has weight 1. The probability that
p_j lies in the question-rooted tree of a random two-rooted spanning forest solves A m = w_q with
A = diag(1 + w_q + sum_i w_ij) - W^T.
usage: probe_forest.py <cache tag> <pool file>
"""
import sys

import numpy as np
import torch

from common import DATASETS, RES, jload, load_bench, recall_at


def forest_marginals(a, C):
    a, C = a.double(), C.double()
    n = a.shape[0]
    wq = a.exp()
    W = C.exp() * (1 - torch.eye(n, dtype=a.dtype, device=a.device))
    A = torch.diag(1 + wq + W.sum(0)) - W.T
    return torch.linalg.solve(A, wq)


def main(tag, pool_file):
    for ds in DATASETS:
        _, qs = load_bench(ds)
        pools = jload(RES / "pools" / f"{pool_file}_{ds}.json")
        cache = torch.load(RES / "cache" / f"{tag}_{ds}.pt")
        n = len(cache)
        res = {"unary": {"R@2": [], "R@5": [], "AR@5": []}, "forest": {"R@2": [], "R@5": [], "AR@5": []}}
        for q, pool, c in zip(qs[:n], pools[:n], cache):
            m = forest_marginals(c["a"], c["C"])
            for name, s in (("unary", c["a"].double()), ("forest", m)):
                o = torch.argsort(s, descending=True).tolist()
                r5 = recall_at([pool[i] for i in o[:5]], q["gold"])
                res[name]["R@2"].append(recall_at([pool[i] for i in o[:2]], q["gold"]))
                res[name]["R@5"].append(r5); res[name]["AR@5"].append(float(r5 == 1))
        print(ds, n, {k: {m: round(100 * np.mean(v), 1) for m, v in d.items()} for k, d in res.items()}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
