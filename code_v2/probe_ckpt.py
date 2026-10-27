"""Evaluates an intermediate training checkpoint (<name>.resume.pt) on the first n test questions and prints the
same-question numbers of reference runs, so a running training can be judged without stopping it.

usage: probe_ckpt.py <resume name> <tag> <n> [reference runs, comma list]
"""
import sys
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification

from common import DATASETS, RES, jload, load_bench, recall_at
from forest import ENC, evaluate


def main(name, tag, n, refs):
    st = torch.load(RES / "ckpt" / f"{name}.resume.pt", map_location="cpu", weights_only=False)
    print("checkpoint step", st["step"], flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(ENC, attn_implementation="sdpa")
    model.load_state_dict(st["model"]); del st
    model = model.to(torch.bfloat16).cuda()
    args = SimpleNamespace(datasets=",".join(DATASETS), pool_file="exp20_s5_k8", n_test=n, eval_budget=6000, reasons=True)
    evaluate(model, args, tag)
    for ds in DATASETS:
        _, qs = load_bench(ds)
        for ref in refs:
            f = RES / "retrieval" / f"{ref}_{ds}.json"
            if not f.exists():
                continue
            run = jload(f)[:n]
            r5 = [recall_at(r["top5"], q["gold"]) for r, q in zip(run, qs)]
            r2 = [recall_at(r["top2"], q["gold"]) for r, q in zip(run, qs)]
            print(ds, ref, {"R@2": round(100 * np.mean(r2), 1), "R@5": round(100 * np.mean(r5), 1),
                            "AR@5": round(100 * np.mean([x == 1 for x in r5]), 1)}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4].split(",") if len(sys.argv) > 4 else [])
