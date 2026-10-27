"""Diagnose a trained CVN checkpoint: recall of top-2 by unary/Shapley on train-style pools vs test pools."""
import sys

import numpy as np
import torch

from common import DATA, RES, jl_load, jload, load_bench, recall_at
from cvn import CVN, Encoder, best_coalition, shapley

ck, use_pair, use_set = sys.argv[1], sys.argv[2] == "1", sys.argv[3] == "1"
m = CVN(use_pair=use_pair, use_set=use_set).cuda()
if ck != "none":
    m.load_state_dict(torch.load(RES / "ckpt" / f"{ck}.pt"))
m.eval(); enc = Encoder()
rows = jl_load(DATA / "train" / "train_4000.jsonl")[-300:]
with torch.no_grad():
    for ds in ["hotpotqa", "2wikimultihopqa", "musique"]:
        R = [r for r in rows if r["ds"] == ds][:60]
        rp, rm = [], []
        for r in R:
            ids, att, pm = enc(r["question"], r["pool"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                a, B = m(ids.cuda(), att.cuda(), pm.cuda())
            k = len(r["gold"])
            rm.append(recall_at(best_coalition(a.float(), B.float(), list(range(len(r["pool"]))), k), r["gold"]))
        print(f"train-pool {ds:16s} R@|G| {np.mean(rm):.3f}")
        docs, qs = load_bench(ds)
        pools = jload(RES / "pools" / f"exp20_s5_k8_{ds}.json")
        r5 = []
        for q, pool in list(zip(qs, pools))[:60]:
            ids, att, pm = enc(q["question"], [docs[i] for i in pool])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                a, B = m(ids.cuda(), att.cuda(), pm.cuda())
            s = best_coalition(a.float(), B.float(), list(range(len(pool))), 5)
            r5.append(recall_at([pool[i] for i in s], q["gold"]))
        print(f"test-pool  {ds:16s} R@5 {np.mean(r5):.3f}  alpha={m.alpha.item():.3f}")
