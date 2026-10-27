"""Print the evidence forest of one test question: top passages by marginal, their most probable reasons,
and what the pointwise view (question edge alone) and the conditional cross-encoder control would select.
usage: case_forest.py <forest tag> <dataset> <question substring>
"""
import sys

import torch

from common import RES, jload, load_bench
from forest import marginals, reason_posteriors


def main(tag, ds, needle):
    docs, qs = load_bench(ds)
    qi = next(i for i, q in enumerate(qs) if needle.lower() in q["question"].lower())
    q = qs[qi]
    pool = jload(RES / "pools" / f"exp20_s5_k8_{ds}.json")[qi]
    c = torch.load(RES / "cache" / f"{tag}_{ds}.pt")[qi]
    sq, S = c["sq"].double(), c["S"].double()
    m = marginals(sq, S)
    pq, pS = reason_posteriors(sq, S)
    print(q["question"], "| answers", q["answers"], "| gold", [docs[g].partition('\n')[0] for g in q["gold"]])
    order = torch.argsort(m, descending=True).tolist()
    for r, j in enumerate(order[:8]):
        cands = [("q", float(pq[j]))] + [(docs[pool[i]].partition("\n")[0], float(pS[i, j])) for i in range(len(pool)) if i != j]
        best = sorted(cands, key=lambda x: -x[1])[:2]
        print(f"{r + 1}. m={float(m[j]):.3f} sigma(s_q)={float(torch.sigmoid(sq[j])):.3f} gold={pool[j] in q['gold']} "
              f"{docs[pool[j]][:90]!r}\n     reasons: " + ", ".join(f"{t} {p:.2f}" for t, p in best) + f", null {1 - sum(p for _, p in cands):.2f}")
    urank = torch.argsort(sq, descending=True).tolist()[:5]
    print("pointwise top-5:", [(docs[pool[j]].partition('\n')[0], pool[j] in q["gold"]) for j in urank])
    for f in (f"f_cdn-cover_{ds}.json",):
        try:
            row = jload(RES / "retrieval" / f)[qi]
            print(f, [(docs[d].partition('\n')[0], d in q["gold"]) for d in row["top5"]])
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
