"""Cost-accuracy analysis on cached edge logits: exact forest inference when only the r passages with the largest
question edges may serve as reasons (r = 0 is the pointwise model, r = n the full forest). Scoring cost is
n + r(n-1) cross-encoder evaluations instead of n^2.
usage: forest_cost.py <cache tag> <pool file>
"""
import sys

import numpy as np
import torch

from common import DATASETS, RES, jdump, jload, load_bench, recall_at
from forest import marginals

RS = [0, 1, 2, 3, 5, 8, 12, 20, 1000]


def main(tag, pool_file):
    out = {}
    for ds in DATASETS:
        _, qs = load_bench(ds)
        pools = jload(RES / "pools" / f"{pool_file}_{ds}.json")
        cache = torch.load(RES / "cache" / f"{tag}_{ds}.pt")
        met = {r: {"R@5": [], "AR@5": [], "frac": []} for r in RS}
        for q, pool, c in zip(qs, pools, cache):
            sq, S = c["sq"].double(), c["S"].double()
            n = sq.shape[0]
            order = torch.argsort(sq, descending=True)
            for r in RS:
                keep = torch.zeros(n, dtype=torch.bool)
                keep[order[:r]] = True
                Sr = S.masked_fill(~keep[:, None], float("-inf"))
                m = marginals(sq, Sr)
                t5 = [pool[i] for i in torch.argsort(m, descending=True)[:5].tolist()]
                r5 = recall_at(t5, q["gold"])
                met[r]["R@5"].append(r5); met[r]["AR@5"].append(float(r5 == 1))
                met[r]["frac"].append((n + min(r, n) * (n - 1)) / (n * n))
        out[ds] = {str(r): {k: float(np.mean(v)) for k, v in d.items()} for r, d in met.items()}
        print(ds, {r: (round(100 * v["R@5"], 1), round(100 * v["AR@5"], 1), round(v["frac"], 2)) for r, v in out[ds].items()},
              flush=True)
    jdump(out, RES / "tables" / f"{tag}_cost.json")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
