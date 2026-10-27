"""Coverage if the reason givers of closure inference recruit partners ([q; giver] dense query, r per giver),
compared with the dense top-N alone and with the given pool file.
usage: giver_cov.py <closure retrieval tag> <pool file> [N] [r]
"""
import sys

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from common import DATASETS, RES, jload, load_bench, recall_at
from expand import QP
from first_stage import encode


def main(tag, pool_file, N=100, r=8):
    tok = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5", torch_dtype=torch.float16).cuda().eval()
    for ds in DATASETS:
        docs, qs = load_bench(ds)
        run = jload(RES / "retrieval" / f"{tag}_{ds}.json")
        qs = qs[: len(run)]
        fs = jload(RES / "first_stage" / f"bge_{ds}.json")["ranked"]
        pools = jload(RES / "pools" / f"{pool_file}_{ds}.json")
        D = encode(model, tok, docs, "cls").cuda()
        queries, owners = [], []
        for qi, (q, x) in enumerate(zip(qs, run)):
            for g in x["givers"]:
                t, _, s = docs[g].partition("\n")
                queries.append(f"{QP}{q['question']} {t}: {s}"); owners.append(qi)
        Q = encode(model, tok, queries, "cls").cuda()
        top = (Q @ D.T).topk(r + 1, dim=1).indices.tolist()
        grown = [set(f[:N]) for f in fs[: len(qs)]]
        for qi, cand in zip(owners, top):
            grown[qi].update(cand)
        for name, P in (("dense", [set(f[:N]) for f in fs]), ("pool", [set(p) for p in pools]), ("giver", grown)):
            cov = np.mean([recall_at(list(p), q["gold"]) for p, q in zip(P, qs)])
            full = np.mean([all(g in p for g in q["gold"]) for p, q in zip(P, qs)])
            print(f"{ds:16s} {name:6s} coverage {100 * cov:.1f} complete {100 * full:.1f} size {np.mean([len(p) for p in P[:len(qs)]]):.1f}",
                  flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *map(int, sys.argv[3:]))
