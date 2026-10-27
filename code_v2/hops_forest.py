"""MuSiQue by hop count and by hop position (which link of the gold chain is retrieved), for several runs.

usage: hops_forest.py run1,run2,...   writes results/tables/hops_forest.json
"""
import re
import sys
from collections import defaultdict

import numpy as np

from common import DATA, RES, jdump, jload, load_bench, recall_at


def hop_chain(sample, index):
    out = []
    for st in sample["question_decomposition"]:
        p = sample["paragraphs"][st["paragraph_support_idx"]]
        out.append(index.get(p["title"] + "\n" + p["paragraph_text"]))
    return out


def main(runs):
    docs, qs = load_bench("musique")
    index = {d: i for i, d in enumerate(docs)}
    raw = jload(DATA / "musique.json")
    chains = [hop_chain(s, index) for s in raw]
    res = {}
    for m in runs:
        rows = (jload(RES / "first_stage" / "bge_musique.json")["ranked"] if m.startswith("first:")
                else jload(RES / "retrieval" / f"{m}_musique.json"))
        top5 = [(r[:5] if m.startswith("first:") else (r.get("top5") or r["rank"][:5])) for r in rows]
        try:
            f1 = [x[1] for x in jload(RES / "qa" / f"qwen25-7b-awq_{m.replace(':', '-')}_musique.json")["em_f1"]]
        except FileNotFoundError:
            f1 = None
        by_h = defaultdict(lambda: {"R@5": [], "AR@5": [], "F1": []})
        by_pos = defaultdict(list)
        for i, (q, c) in enumerate(zip(qs, chains)):
            h = len(c)
            r5 = recall_at(top5[i], q["gold"])
            by_h[h]["R@5"].append(r5); by_h[h]["AR@5"].append(float(r5 == 1))
            if f1 is not None:
                by_h[h]["F1"].append(f1[i])
            for k, d in enumerate(c):
                if d is not None:
                    by_pos[(h, k + 1)].append(float(d in top5[i]))
        res[m] = {"by_hops": {h: {k: float(np.mean(v)) if v else None for k, v in d.items()} | {"n": len(d["R@5"])}
                              for h, d in sorted(by_h.items())},
                  "by_position": {f"{h}-{k}": float(np.mean(v)) for (h, k), v in sorted(by_pos.items())}}
        print(m, {h: (v["n"], round(100 * v["R@5"], 1), round(100 * v["AR@5"], 1), None if v["F1"] is None else round(100 * v["F1"], 1))
                  for h, v in res[m]["by_hops"].items()}, flush=True)
        print("   position recall", {k: round(100 * v, 1) for k, v in res[m]["by_position"].items()}, flush=True)
    jdump(res, RES / "tables" / "hops_forest.json")


if __name__ == "__main__":
    main(sys.argv[1].split(","))
