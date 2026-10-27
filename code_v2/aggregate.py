"""Aggregate retrieval/QA results for a list of methods; paired bootstrap vs. a reference method.

usage: aggregate.py --methods first:bm25,first:bge,zs_rerank_exp,cvn_full --ref cvn_full [--reader qwen25-7b-awq] [--n_test N]
Retrieval metrics use 'top2'/'top5' if present (coalition outputs) else the ranked list.
"""
import argparse

import numpy as np

from common import DATASETS, RES, SHORT, jdump, jload, load_bench, recall_at


def retr(tag, ds, qs):
    if tag.startswith("first:"):
        r = jload(RES / "first_stage" / f"{tag[6:]}_{ds}.json")["ranked"]
        rows = [{"rank": x} for x in r]
    else:
        rows = jload(RES / "retrieval" / f"{tag}_{ds}.json")
    out = {"R@2": [], "R@5": [], "AR@5": []}
    for q, x in zip(qs, rows):
        t2 = x.get("top2") or x["rank"][:2]; t5 = x.get("top5") or x["rank"][:5]
        r5 = recall_at(t5, q["gold"])
        out["R@2"].append(recall_at(t2, q["gold"])); out["R@5"].append(r5); out["AR@5"].append(float(r5 == 1.0))
    return {k: np.array(v) for k, v in out.items()}


def qa(tag, ds, reader):
    try:
        d = jload(RES / "qa" / f"{reader}_{tag.replace(':', '-')}_{ds}.json")
    except FileNotFoundError:
        return None
    s = np.array(d["em_f1"])
    return {"EM": s[:, 0], "F1": s[:, 1]}


def boot_p(a, b, n=10000, seed=0):
    """one-sided p-value that mean(a) <= mean(b) under paired bootstrap."""
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), (n, len(d)))
    return float((d[idx].mean(1) <= 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", required=True)
    ap.add_argument("--ref", default="")
    ap.add_argument("--reader", default="qwen25-7b-awq")
    ap.add_argument("--n_test", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    table = {}
    for ds in DATASETS:
        _, qs = load_bench(ds)
        for m in args.methods.split(","):
            try:
                r = retr(m, ds, qs)
            except FileNotFoundError:
                continue
            n = args.n_test or len(r["R@5"])
            row = {k: v[:n] for k, v in r.items()}
            a = qa(m, ds, args.reader)
            if a is not None and len(a["F1"]) >= n:
                row.update({k: v[:n] for k, v in a.items()})
            table.setdefault(m, {})[ds] = row
    metrics = ["R@2", "R@5", "AR@5", "EM", "F1"]
    print(f"{'method':28s} " + " ".join(f"{SHORT[d]+' '+k:>14s}" for d in DATASETS for k in metrics))
    summary = {}
    for m, per in table.items():
        cells = []
        for ds in DATASETS:
            for k in metrics:
                v = per.get(ds, {}).get(k)
                if v is None:
                    cells.append(f"{'-':>14s}"); continue
                mark = ""
                if args.ref and m != args.ref and k in table.get(args.ref, {}).get(ds, {}):
                    p = boot_p(table[args.ref][ds][k], v)
                    mark = "*" if p < 0.05 else ""
                cells.append(f"{100 * v.mean():13.1f}{mark or ' '}")
                summary.setdefault(m, {}).setdefault(ds, {})[k] = float(v.mean())
        print(f"{m:28s} " + " ".join(cells))
    if args.out:
        jdump(summary, RES / "tables" / f"{args.out}.json")
    print("* = reference significantly better (paired bootstrap, p<0.05)")


if __name__ == "__main__":
    main()
