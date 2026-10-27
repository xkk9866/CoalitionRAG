"""Larger candidate pools for closure inference: an existing pool followed by the remaining first-stage top-N.

usage: union_pool.py <base pool file> <N> <name>
"""
import sys

import numpy as np

from common import DATASETS, RES, jdump, jload, load_bench, recall_at


def main(base, N, name):
    for ds in DATASETS:
        _, qs = load_bench(ds)
        pools = jload(RES / "pools" / f"{base}_{ds}.json")
        fs = jload(RES / "first_stage" / f"bge_{ds}.json")["ranked"]
        out = []
        for p, f in zip(pools, fs):
            seen = set(p)
            out.append(list(p) + [d for d in f[:N] if d not in seen])
        jdump(out, RES / "pools" / f"{name}_{ds}.json")
        rec = np.mean([recall_at(p, q["gold"]) for p, q in zip(out, qs)])
        full = np.mean([all(g in p for g in q["gold"]) for p, q in zip(out, qs)])
        print(f"{ds:16s} coverage {100 * rec:.1f} complete {100 * full:.1f} size {np.mean([len(p) for p in out]):.1f}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
