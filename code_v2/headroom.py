"""Where is the headroom? For each dataset: gold coverage of the candidate pool, and for gold passages missing from
the pool, whether recruitment from a gold passage that *is* in the pool ([q; gold] dense query, 8 partners) would
reach them. Also the coverage after recruiting from all in-pool gold passages (an oracle for evidence-seeded growth).
usage: headroom.py <pool file>
"""
import sys

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from common import DATASETS, RES, jload, load_bench, recall_at
from expand import QP
from first_stage import encode


def main(pool_file, per_seed=8):
    tok = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5", torch_dtype=torch.float16).cuda().eval()
    for ds in DATASETS:
        docs, qs = load_bench(ds)
        pools = jload(RES / "pools" / f"{pool_file}_{ds}.json")
        D = encode(model, tok, docs, "cls").cuda()
        queries, owners = [], []
        for qi, (q, pool) in enumerate(zip(qs, pools)):
            for g in q["gold"]:
                if g in pool:
                    t, _, x = docs[g].partition("\n")
                    queries.append(f"{QP}{q['question']} {t}: {x}"); owners.append((qi, g))
        Q = encode(model, tok, queries, "cls").cuda()
        top = (Q @ D.T).topk(per_seed + 1, dim=1).indices.tolist()
        grown = [set(p) for p in pools]
        for (qi, g), cand in zip(owners, top):
            grown[qi].update(cand)
        cov = np.mean([recall_at(p, q["gold"]) for p, q in zip(pools, qs)])
        gcov = np.mean([recall_at(list(p), q["gold"]) for p, q in zip(grown, qs)])
        full = np.mean([float(all(g in p for g in q["gold"])) for p, q in zip(pools, qs)])
        gfull = np.mean([float(all(g in p for g in q["gold"])) for p, q in zip(grown, qs)])
        print(f"{ds:16s} pool coverage {cov:.3f} (complete {full:.3f})  + gold-seeded growth {gcov:.3f} (complete {gfull:.3f})"
              f"  size {np.mean([len(p) for p in pools]):.1f} -> {np.mean([len(p) for p in grown]):.1f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
