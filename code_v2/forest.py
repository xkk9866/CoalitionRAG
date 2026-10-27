"""Latent Evidence Forests.

Question q, candidate pool P = {p_1..p_n}. Every passage chooses one *reason* (its parent): the question q,
another passage p_i, or the null root (it is not evidence). A configuration is admissible iff the reason graph
is acyclic, i.e. a spanning forest of P rooted at {q, null}. The evidence set E is the tree rooted at q.

    P(forest) ∝ prod_j w(parent_j -> j),   w(null->j) = 1,   w(q->j) = exp f(q => p_j),   w(i->j) = exp f(q+p_i => p_j)

with one cross-encoder f. By the matrix-forest theorem (Tutte, with the two roots merged):

    Z            = det A,              A = diag(1 + w_q + sum_i W_ij) - W^T
    log P(E = G) = log det A_G[q] + log det A_{P\\G}[null] - log det A
    P(j in E)    = m_j,                A m = w_q       (absorption probability of the reason walk)

Training maximises log P(E = G) for the gold evidence set; which passage is the reason of which is latent.
Inference returns the top-k passages by m, the Bayes decision for the expected number of retrieved evidence passages.
"""
import argparse
import random
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import jmerge, DATA, DATASETS, RES, jdump, jl_load, jload, load_bench, recall_at

ENC = "BAAI/bge-reranker-v2-m3"


# ------------------------------------------------------------------ forest algebra (float64)
def incoming(sq, S):
    """Per child j, weights of its possible reasons (null, q, p_i), rescaled by a per-child constant.

    Multiplying all incoming weights of one node by a constant leaves every forest probability unchanged,
    so the rescaling only serves numerical range. Returns null (n,), wq (n,), W (n, n) with W[i, j] = w(i -> j),
    and the log scale M (n,).
    """
    n = sq.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=sq.device)
    S = S.masked_fill(eye, float("-inf"))
    M = torch.maximum(torch.maximum(sq, S.max(0).values), torch.zeros_like(sq))
    return torch.exp(-M), torch.exp(sq - M), torch.exp(S - M[None, :]), M


def laplacian(root, W, nodes=None):
    """A_jj = root_j + sum_{i in nodes} W_ij,  A_ji = -W_ij (row = child, column = parent)."""
    if nodes is not None:
        root, W = root[nodes], W[nodes][:, nodes]
    return torch.diag(root + W.sum(0)) - W.T


def log_prob(sq, S, gold):
    """log P(E = gold) under the evidence-forest distribution."""
    null, wq, W, _ = incoming(sq.double(), S.double())
    n = sq.shape[0]
    g = torch.zeros(n, dtype=torch.bool, device=sq.device)
    g[gold] = True
    G, N = g.nonzero().squeeze(1), (~g).nonzero().squeeze(1)
    out = torch.logdet(laplacian(wq, W, G)) - torch.logdet(laplacian(null + wq, W))
    if len(N):
        out = out + torch.logdet(laplacian(null, W, N))
    return out


def marginals(sq, S):
    """m_j = P(j in E): solves A m = w_q."""
    null, wq, W, _ = incoming(sq.double(), S.double())
    return torch.linalg.solve(laplacian(null + wq, W), wq)


def reason_posteriors(sq, S, nodes=None):
    """P(parent_j = x) for x in {q, p_i}; restricted to the trees on `nodes` if given (posterior given E = nodes).

    Edge marginals are derivatives of log Z with respect to the log edge weights.
    Returns Pq (n,), Pw (n, n) with Pw[i, j] = P(parent_j = i); zeros outside `nodes`.
    """
    sq = sq.detach().double().clone().requires_grad_(True)
    S = S.detach().double().clone().requires_grad_(True)
    null, wq, W, M = incoming(sq, S)
    if nodes is None:
        logz = torch.logdet(laplacian(null + wq, W)) + M.sum()
    else:
        nodes = torch.as_tensor(nodes, device=sq.device)
        logz = torch.logdet(laplacian(wq, W, nodes)) + M[nodes].sum()
    gq, gS = torch.autograd.grad(logz, (sq, S))
    return gq, gS


# ------------------------------------------------------------------ cross-encoder edges
class Rows:
    """[CLS] q (SEP reason passage) [SEP][SEP] candidate [SEP] -- same layout and lengths as the CDN control."""

    def __init__(self, qmax=64, ctx_len=160, cand_len=200):
        self.tok = AutoTokenizer.from_pretrained(ENC)
        self.qmax, self.ctx_len, self.cand_len = qmax, ctx_len, cand_len
        self.cache = {}

    def ids(self, text, n):
        key = (text, n)
        if key not in self.cache:
            t, _, x = text.partition("\n")
            self.cache[key] = self.tok(f"{t}: {x}" if x else text, add_special_tokens=False)["input_ids"][:n]
        return self.cache[key]

    def build(self, q, texts, reasons=True):
        """All n unary rows (q => p_j) followed by the n(n-1) reason rows (q + p_i => p_j), i != j."""
        t = self.tok
        qi = self.ids(q, self.qmax)
        cand = [self.ids(x, self.cand_len) for x in texts]
        ctx = [self.ids(x, self.ctx_len) for x in texts]
        tail = [[t.sep_token_id, t.sep_token_id] + c + [t.sep_token_id] for c in cand]
        rows, index = [], []
        for j in range(len(texts)):
            rows.append([t.cls_token_id] + qi + tail[j]); index.append((-1, j))
        for i in range(len(texts) if reasons else 0):
            head = [t.cls_token_id] + qi + [t.sep_token_id] + ctx[i]
            for j in range(len(texts)):
                if i != j:
                    rows.append(head + tail[j]); index.append((i, j))
        return rows, index


def run_rows(model, rows, pad_id, budget, grad=False):
    """Scores rows in length-sorted batches of at most `budget` tokens; returns logits in input order."""
    order = sorted(range(len(rows)), key=lambda k: len(rows[k]))
    out = [None] * len(rows)
    b = 0
    while b < len(order):
        e = b + 1
        while e < len(order) and (e - b + 1) * len(rows[order[e]]) <= budget:
            e += 1
        chunk = order[b:e]
        L = max(len(rows[k]) for k in chunk)
        ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
        att = torch.zeros(len(chunk), L, dtype=torch.long)
        for r, k in enumerate(chunk):
            ids[r, :len(rows[k])] = torch.tensor(rows[k]); att[r, :len(rows[k])] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.set_grad_enabled(grad):
            lg = model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits.squeeze(-1).float()
        for r, k in enumerate(chunk):
            out[k] = lg[r]
        b = e
    return torch.stack(out)


def edge_logits(model, rw, q, texts, budget, grad=False, reasons=True):
    """Question edges s_q (n,) and reason edges S (n, n); without reasons every S entry is -inf (w = 0)."""
    rows, index = rw.build(q, texts, reasons)
    lg = run_rows(model, rows, rw.tok.pad_token_id, budget, grad)
    n = len(texts)
    sq = lg[:n]
    if not reasons:
        return sq, torch.full((n, n), float("-inf"), device=lg.device, dtype=lg.dtype)
    ii = torch.tensor([i for i, _ in index[n:]], device=lg.device)
    jj = torch.tensor([j for _, j in index[n:]], device=lg.device)
    S = torch.full((n, n), float("-inf"), device=lg.device, dtype=lg.dtype)
    S = S.index_put((ii, jj), lg[n:])
    return sq, S


# ------------------------------------------------------------------ training
def train(args):
    """One pass over the training questions; resumable from <name>.resume.pt (saved every --ckpt_every updates).

    The question order and every sampled sub-pool are functions of (seed, epoch, position) only, so a resumed run
    sees exactly the data an uninterrupted run would have seen.
    """
    torch.manual_seed(args.seed)
    rows = jl_load(DATA / "train" / args.train)
    if args.n_train:
        rows = rows[: args.n_train]
    model = AutoModelForSequenceClassification.from_pretrained(ENC, attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    rw = Rows()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = args.epochs * len(rows) // args.accum
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / warm) * max(0.0, 1 - s / max(1, total)))
    resume = RES / "ckpt" / f"{args.name}.resume.pt"
    step, start_ep, start_it = 0, 0, 0
    if resume.exists():
        st = torch.load(resume, map_location="cuda", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        step, start_ep, start_it = st["step"], st["ep"], st["it"]
        print(f"resumed at epoch {start_ep} question {start_it} step {step}", flush=True)
    model.train()
    run, t0 = [], time.time()
    for ep in range(start_ep, args.epochs):
        order = list(range(len(rows)))
        random.Random(f"{args.seed}-order-{ep}").shuffle(order)
        for it in range(start_it if ep == start_ep else 0, len(order)):
            r = rows[order[it]]
            rng = random.Random(f"{args.seed}-{ep}-{it}")
            gold = list(r["gold"])
            neg = [i for i in range(len(r["pool"])) if i not in gold]
            sub = gold + rng.sample(neg, min(args.n_neg, len(neg)))
            rng.shuffle(sub)
            sq, S = edge_logits(model, rw, r["question"], [r["pool"][i] for i in sub], args.train_budget, grad=True,
                                reasons=args.reasons)
            loss = -log_prob(sq, S, [sub.index(g) for g in gold])
            (loss.float() / args.accum).backward()
            run.append(loss.item())
            if (it + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if step % 100 == 0:
                    print(f"ep {ep} step {step}/{total} -logP(E=G) {np.mean(run):.4f} {time.time() - t0:.0f}s", flush=True)
                    run = []
                if step % args.ckpt_every == 0 and it + 1 < len(order):
                    tmp = resume.with_suffix(".tmp")
                    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                                "step": step, "ep": ep, "it": it + 1}, tmp)
                    tmp.replace(resume)
    torch.save(model.state_dict(), RES / "ckpt" / f"{args.name}.pt")
    resume.unlink(missing_ok=True)
    return model


# ------------------------------------------------------------------ inference
@torch.no_grad()
def evaluate(model, args, tag):
    model = model.eval().to(torch.bfloat16)
    rw = Rows()
    summary = {}
    for ds in args.datasets.split(","):
        docs, qs = load_bench(ds)
        pools = jload(RES / "pools" / f"{args.pool_file}_{ds}.json")
        if args.n_test:
            qs, pools = qs[: args.n_test], pools[: args.n_test]
        part = RES / "retrieval" / f"{tag}_{ds}.part.json"
        if part.exists():                      # a finished dataset of an interrupted evaluation
            summary[ds] = jload(part)
            print(ds, "done earlier", flush=True)
            continue
        out, cache, t0 = [], [], time.time()
        met = {k: [] for k in ("R@2", "R@5", "AR@5", "uR@2", "uR@5")}
        for q, pool in zip(qs, pools):
            sq, S = edge_logits(model, rw, q["question"], [docs[i] for i in pool], args.eval_budget, reasons=args.reasons)
            m = marginals(sq, S)
            rank = torch.argsort(m, descending=True).tolist()
            urank = torch.argsort(sq, descending=True).tolist()
            cache.append({"sq": sq.cpu(), "S": S.cpu()})
            out.append({"id": q["id"], "top2": [pool[i] for i in rank[:2]], "top5": [pool[i] for i in rank[:5]],
                        "rank": [pool[i] for i in rank], "m": [float(m[i]) for i in rank]})
            r5 = recall_at(out[-1]["top5"], q["gold"])
            met["R@2"].append(recall_at(out[-1]["top2"], q["gold"])); met["R@5"].append(r5); met["AR@5"].append(float(r5 == 1))
            met["uR@2"].append(recall_at([pool[i] for i in urank[:2]], q["gold"]))
            met["uR@5"].append(recall_at([pool[i] for i in urank[:5]], q["gold"]))
        summary[ds] = {k: float(np.mean(v)) for k, v in met.items()}
        summary[ds]["sec_per_q"] = (time.time() - t0) / len(qs)
        jdump(out, RES / "retrieval" / f"{tag}_{ds}.json")
        (RES / "cache").mkdir(exist_ok=True)
        torch.save(cache, RES / "cache" / f"{tag}_{ds}.pt")
        jdump(summary[ds], part)
        print(ds, {k: round(100 * v, 1) if k != "sec_per_q" else round(v, 2) for k, v in summary[ds].items()}, flush=True)
    jmerge(summary, RES / "retrieval" / f"{tag}_summary.json")
    for ds in args.datasets.split(","):
        (RES / "retrieval" / f"{tag}_{ds}.part.json").unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="forest")
    ap.add_argument("--train", default="train_open.jsonl")
    ap.add_argument("--n_train", type=int, default=0)
    ap.add_argument("--n_neg", type=int, default=8, help="negatives sampled with the gold set per training pool")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--ckpt_every", type=int, default=500, help="updates between resumable checkpoints")
    ap.add_argument("--train_budget", type=int, default=24000, help="tokens per forward chunk (memory only)")
    ap.add_argument("--eval_budget", type=int, default=64000, help="tokens per forward chunk (memory only)")
    ap.add_argument("--pool_file", default="exp20_s5_k8")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--n_test", type=int, default=0)
    ap.add_argument("--eval_only", default="", help="checkpoint name, or 'zero' for the untrained cross-encoder")
    ap.add_argument("--no_reasons", dest="reasons", action="store_false",
                    help="forest without reason edges: the likelihood is exactly pointwise BCE")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    (RES / "ckpt").mkdir(parents=True, exist_ok=True)
    if args.eval_only:
        model = AutoModelForSequenceClassification.from_pretrained(ENC, attn_implementation="sdpa").cuda()
        if args.eval_only != "zero":
            model.load_state_dict(torch.load(RES / "ckpt" / f"{args.eval_only}.pt"))
    elif (RES / "ckpt" / f"{args.name}.pt").exists():   # training finished in an earlier attempt
        model = AutoModelForSequenceClassification.from_pretrained(ENC, attn_implementation="sdpa").cuda()
        model.load_state_dict(torch.load(RES / "ckpt" / f"{args.name}.pt"))
    else:
        model = train(args)
    evaluate(model, args, args.name)


if __name__ == "__main__":
    main()
