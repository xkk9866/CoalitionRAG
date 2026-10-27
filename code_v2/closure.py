"""Closure inference for evidence forests: only selected passages give reasons, and givers recruit candidates.

A reason from a passage outside the evidence tree places its child outside the tree, exactly as the null reason
does, so reasons matter only when they come from evidence. Closure inference therefore lets a passage give reasons
only once it is selected, and a passage that gives reasons also brings the passages it points to into the pool:

    P = first-stage top-N,  R = {}                      (R: passages whose outgoing reason edges are scored)
    repeat:  m = exact forest marginals on P with reasons from R;   T = top-k(m)
             if T ⊆ R: stop
             for i in T \\ R: add the partners of i (dense query [q; p_i]) to P, score the reason edges out of i
             R = R ∪ T

The output is a fixed point: the top-k of the exact forest over a pool that is closed under recruitment by its own
reason givers, in which exactly those givers give reasons. A marginal depends only on the question edge and the
edges from R (m_j = (w_qj + sum_{i in R} W_ij m_i) / (1 + w_qj + sum_{i in R} W_ij)), so it does not change with the
pool size. The cost is n + |R|(n-1) cross-encoder evaluations instead of n^2. k is the reader budget.
usage: closure.py --eval_only <ckpt> --name <tag> (--pool_file <pools> | --base N --grow r) [--n_test N]
"""
import argparse
import time

import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from common import jmerge, DATASETS, RES, jdump, jload, load_bench, recall_at
from expand import QP
from first_stage import encode
from forest import ENC, Rows, marginals, run_rows


def pair_rows(rw, q, texts, pairs):
    """Rows (q + p_i => p_j) for (i, j) in pairs; i = -1 is the question edge (q => p_j)."""
    t = rw.tok
    qi = rw.ids(q, rw.qmax)
    rows = []
    for i, j in pairs:
        head = [t.cls_token_id] + qi + ([t.sep_token_id] + rw.ids(texts[i], rw.ctx_len) if i >= 0 else [])
        rows.append(head + [t.sep_token_id, t.sep_token_id] + rw.ids(texts[j], rw.cand_len) + [t.sep_token_id])
    return rows


class Recruiter:
    """Partners of a passage: the dense top-r passages for the query [q; p] (the candidate generation's encoder)."""

    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
        self.model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5", torch_dtype=torch.float16).cuda().eval()
        self.D = None

    def corpus(self, docs):
        self.D = encode(self.model, self.tok, docs, "cls").cuda().half()

    def partners(self, q, docs, ids, r):
        qs = []
        for i in ids:
            t, _, x = docs[i].partition("\n")
            qs.append(f"{QP}{q} {t}: {x}")
        Q = encode(self.model, self.tok, qs, "cls").cuda().half()
        top = (Q @ self.D.T).topk(r + 1, dim=1).indices.tolist()
        return [[c for c in row if c != i][:r] for i, row in zip(ids, top)]


@torch.no_grad()
def closure(model, rw, q, docs, pool, k, budget, rec=None, r=0):
    """Returns the final pool (doc ids), unary logits, reason logits (-inf where not scored), marginals, givers."""
    pool = list(pool)
    texts = [docs[d] for d in pool]
    sq = run_rows(model, pair_rows(rw, q, texts, [(-1, j) for j in range(len(pool))]), rw.tok.pad_token_id, budget).double()
    S = torch.full((len(pool), len(pool)), float("-inf"), device=sq.device, dtype=torch.double)
    R, rounds, cost = [], 0, len(pool)
    while True:
        m = marginals(sq, S)
        T = torch.argsort(m, descending=True)[:k].tolist()
        new = [i for i in T if i not in R]
        if not new:
            return pool, sq, S, m, R, rounds, cost
        old_n = len(pool)
        if rec is not None and r > 0:
            have = set(pool)
            for row in rec.partners(q, docs, [pool[i] for i in new], r):
                for c in row:
                    if c not in have:
                        have.add(c); pool.append(c); texts.append(docs[c])
        n = len(pool)
        pairs = [(-1, j) for j in range(old_n, n)]                                  # question edges of recruits
        pairs += [(i, j) for i in R for j in range(old_n, n)]                       # old givers -> recruits
        pairs += [(i, j) for i in new for j in range(n) if j != i]                  # new givers -> everyone
        lg = run_rows(model, pair_rows(rw, q, texts, pairs), rw.tok.pad_token_id, budget).double()
        cost += len(pairs)
        sq = torch.cat([sq, lg[: n - old_n]])
        S2 = torch.full((n, n), float("-inf"), device=sq.device, dtype=torch.double)
        S2[:old_n, :old_n] = S
        ii = torch.tensor([i for i, _ in pairs[n - old_n:]], device=sq.device)
        jj = torch.tensor([j for _, j in pairs[n - old_n:]], device=sq.device)
        S = S2.index_put((ii, jj), lg[n - old_n:])
        R += new; rounds += 1


@torch.no_grad()
def evaluate(model, args, tag):
    model = model.eval().to(torch.bfloat16)
    rw = Rows()
    rec = Recruiter() if args.grow else None
    summary = {}
    for ds in args.datasets.split(","):
        docs, qs = load_bench(ds)
        if args.pool_file:
            pools = jload(RES / "pools" / f"{args.pool_file}_{ds}.json")
        else:
            pools = [x[: args.base] for x in jload(RES / "first_stage" / f"bge_{ds}.json")["ranked"]]
        if args.n_test:
            qs, pools = qs[: args.n_test], pools[: args.n_test]
        if rec is not None:
            rec.corpus(docs)
        out, cache, t0 = [], [], time.time()
        met = {k: [] for k in ("R@2", "R@5", "AR@5", "uR@5", "cost", "rounds", "givers", "n", "pool_recall")}
        for q, pool in zip(qs, pools):
            pool, sq, S, m, R, rounds, cost = closure(model, rw, q["question"], docs, pool, args.k, args.eval_budget,
                                                     rec, args.grow)
            rank = torch.argsort(m, descending=True).tolist()
            cache.append({"pool": pool, "sq": sq.float().cpu(), "S": S.float().cpu(), "R": R})
            out.append({"id": q["id"], "top2": [pool[i] for i in rank[:2]], "top5": [pool[i] for i in rank[:5]],
                        "rank": [pool[i] for i in rank], "m": [float(m[i]) for i in rank], "givers": [pool[i] for i in R]})
            r5 = recall_at(out[-1]["top5"], q["gold"])
            met["R@2"].append(recall_at(out[-1]["top2"], q["gold"])); met["R@5"].append(r5); met["AR@5"].append(float(r5 == 1))
            met["cost"].append(cost); met["rounds"].append(rounds); met["givers"].append(len(R)); met["n"].append(len(pool))
            met["pool_recall"].append(recall_at(pool, q["gold"]))
            urank = torch.argsort(sq, descending=True).tolist()          # same network and pool, reasons ignored
            met["uR@5"].append(recall_at([pool[i] for i in urank[:5]], q["gold"]))
        summary[ds] = {k: float(np.mean(v)) for k, v in met.items()}
        summary[ds]["sec_per_q"] = (time.time() - t0) / len(qs)
        jdump(out, RES / "retrieval" / f"{tag}_{ds}.json")
        (RES / "cache").mkdir(exist_ok=True)
        torch.save(cache, RES / "cache" / f"{tag}_{ds}.pt")
        print(ds, {k: round(100 * v, 1) if k in ("R@2", "R@5", "AR@5", "uR@5", "pool_recall") else round(v, 2)
                   for k, v in summary[ds].items()}, flush=True)
    jmerge(summary, RES / "retrieval" / f"{tag}_summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--eval_only", required=True, help="checkpoint name (results/ckpt/<name>.pt or .resume.pt)")
    ap.add_argument("--pool_file", default="", help="fixed candidate pools; otherwise first-stage top --base")
    ap.add_argument("--base", type=int, default=100, help="first-stage depth when no pool file is given")
    ap.add_argument("--grow", type=int, default=0, help="partners recruited per reason giver (0: fixed pool)")
    ap.add_argument("--k", type=int, default=5, help="reader budget")
    ap.add_argument("--eval_budget", type=int, default=64000, help="tokens per forward chunk (memory only)")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--n_test", type=int, default=0)
    args = ap.parse_args()
    model = AutoModelForSequenceClassification.from_pretrained(ENC, attn_implementation="sdpa")
    f = RES / "ckpt" / f"{args.eval_only}.pt"
    if args.eval_only == "zero":
        pass
    elif f.exists():
        model.load_state_dict(torch.load(f, map_location="cpu"))
    else:
        model.load_state_dict(torch.load(RES / "ckpt" / f"{args.eval_only}.resume.pt", map_location="cpu",
                                         weights_only=False)["model"])
    evaluate(model.cuda(), args, args.name)


if __name__ == "__main__":
    main()
