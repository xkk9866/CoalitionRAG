"""Coalition-conditioned candidate generation: every seed passage recruits partners by dense retrieval
with the query [question ; seed passage]. Writes results/pools/{name}_{ds}.json = list of candidate id lists
(first-stage order first, then recruited partners) and reports pool recall.
"""
import argparse

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from common import DATASETS, RES, jdump, jload, load_bench, recall_at
from first_stage import encode

QP = "Represent this sentence for searching relevant passages: "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=int, default=20, help="first-stage candidates kept")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--per_seed", type=int, default=8)
    ap.add_argument("--seed_file", default="", help="optional retrieval json whose 'rank' gives seeds (e.g. CVN stage 1)")
    ap.add_argument("--name", default="exp")
    ap.add_argument("--base_pool", default="", help="start from an existing pool (round-2 recruitment)")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5", torch_dtype=torch.float16).cuda().eval()
    for ds in args.datasets.split(","):
        docs, qs = load_bench(ds)
        fs = jload(RES / "first_stage" / f"bge_{ds}.json")["ranked"]
        seeds_src = [x["rank"] for x in jload(RES / "retrieval" / f"{args.seed_file}_{ds}.json")] if args.seed_file else fs
        qs, fs = qs[: len(seeds_src)], fs[: len(seeds_src)]
        D = encode(model, tok, docs, "cls").cuda()
        queries, owners = [], []
        for qi, (q, sr) in enumerate(zip(qs, seeds_src)):
            for s in sr[: args.seeds]:
                title, _, text = docs[s].partition("\n")
                queries.append(f"{QP}{q['question']} {title}: {text}"); owners.append((qi, s))
        Q = encode(model, tok, queries, "cls").cuda()
        top = (Q @ D.T).topk(args.per_seed + 1, dim=1).indices.tolist()
        base = jload(RES / "pools" / f"{args.base_pool}_{ds}.json") if args.base_pool else None
        pools = [list(base[qi]) if base else list(fs[qi][: args.base]) for qi in range(len(qs))]
        for (qi, s), cand in zip(owners, top):
            for c in cand:
                if c != s and c not in pools[qi]:
                    pools[qi].append(c)
        jdump(pools, RES / "pools" / f"{args.name}_{ds}.json")
        rec = np.mean([recall_at(p, q["gold"]) for p, q in zip(pools, qs)])
        base = np.mean([recall_at(f[: args.base], q["gold"]) for f, q in zip(fs, qs)])
        size = np.mean([len(p) for p in pools])
        print(f"{ds:16s} base@{args.base} {base:.3f} -> expanded {rec:.3f} (mean size {size:.1f})", flush=True)


if __name__ == "__main__":
    main()
