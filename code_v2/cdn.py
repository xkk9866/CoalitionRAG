"""Counterfactual Dividend Network (CDN).

One cross-encoder f scores a passage under a context. The 2-additive evidence game is
    a_j  = f(q => p_j)                                            (unary dividend)
    b_ij = 1/2 [ f(q+p_i => p_j) - f(q => p_j) + f(q+p_j => p_i) - f(q => p_i) ]   (pairwise dividend)
    v(S) = sum_{i in S} a_i + sum_{i<j in S} b_ij
i.e. the Harsanyi interaction of two passages is measured as the counterfactual change of the same network's
score when the partner is known. Inputs are texts only. Selection: exact argmax over all size-k coalitions.
Training: relevance supervision under conditions (gold seed => other gold passages positive; distractor seed
=> labels unchanged).
"""
import argparse
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import DATA, DATASETS, RES, jdump, jl_load, jload, load_bench, recall_at
from cvn import best_coalition

ENC = "BAAI/bge-reranker-v2-m3"


class Pairer:
    """Builds cross-encoder inputs: [CLS] q (+ context passage) [SEP][SEP] candidate [SEP]."""

    def __init__(self, ctx_len=160, cand_len=200, qmax=64):
        self.tok = AutoTokenizer.from_pretrained(ENC)
        self.ctx_len, self.cand_len, self.qmax = ctx_len, cand_len, qmax
        self.cache = {}

    def ids(self, text, n):
        key = (text, n)
        if key not in self.cache:
            t, _, x = text.partition("\n")
            self.cache[key] = self.tok(f"{t}: {x}" if x else text, add_special_tokens=False)["input_ids"][:n]
        return self.cache[key]

    def batch(self, q, pairs):
        """pairs: list of (context passage text or None, candidate text)."""
        t = self.tok
        qi = self.ids(q, self.qmax)
        rows = []
        for ctx, cand in pairs:
            left = qi + ([t.sep_token_id] + self.ids(ctx, self.ctx_len) if ctx else [])
            rows.append([t.cls_token_id] + left + [t.sep_token_id, t.sep_token_id] + self.ids(cand, self.cand_len) + [t.sep_token_id])
        L = max(map(len, rows))
        ids = torch.full((len(rows), L), t.pad_token_id)
        att = torch.zeros(len(rows), L, dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, :len(r)] = torch.tensor(r); att[i, :len(r)] = 1
        return ids, att


def score(model, pr, q, pairs, bs=64):
    out = []
    for b in range(0, len(pairs), bs):
        ids, att = pr.batch(q, pairs[b:b + bs])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits.squeeze(-1).float())
    return torch.cat(out)


# ------------------------------------------------------------------ training
def make_examples(r, rng, n_cand):
    """Returns list of (context idx or None, cand idx, label)."""
    gold = set(r["gold"]); n = len(r["pool"])
    neg = [i for i in range(n) if i not in gold]
    cands = list(gold) + rng.sample(neg, min(len(neg), n_cand - len(gold)))
    ex = [(None, j, float(j in gold)) for j in cands]
    seeds = list(gold) + rng.sample(neg, min(len(neg), 1))
    for s in seeds:
        for j in cands:
            if j != s:
                ex.append((s, j, float(j in gold)))
    return ex


def train(args):
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    rows = jl_load(DATA / "train" / args.train)
    if args.n_train:
        rows = rows[: args.n_train]
    model = AutoModelForSequenceClassification.from_pretrained(ENC).cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    pr = Pairer()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = args.epochs * len(rows) // args.accum
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / max(1, int(0.05 * total))) * max(0.0, 1 - s / max(1, total)))
    model.train(); step, run, t0 = 0, 0.0, time.time()
    for ep in range(args.epochs):
        rng.shuffle(rows)
        for it, r in enumerate(rows):
            ex = make_examples(r, rng, args.n_cand)
            pairs = [(r["pool"][c] if c is not None else None, r["pool"][j]) for c, j, _ in ex]
            ids, att = pr.batch(r["question"], pairs)
            y = torch.tensor([e[2] for e in ex], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits.squeeze(-1).float()
            # listwise within each condition group (one positive set per condition) + pointwise BCE
            loss = F.binary_cross_entropy_with_logits(logit, y)
            groups = {}
            for k, (c, _, _) in enumerate(ex):
                groups.setdefault(c, []).append(k)
            lw = 0.0
            for g in groups.values():
                gi = torch.tensor(g, device="cuda")
                if y[gi].sum() > 0:
                    lw = lw - (torch.log_softmax(logit[gi], 0) * (y[gi] / y[gi].sum())).sum()
            loss = (loss + lw / len(groups)) / args.accum
            loss.backward(); run += loss.item()
            if (it + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if step % 100 == 0:
                    print(f"ep {ep} step {step}/{total} loss {run / 100:.4f} {time.time() - t0:.0f}s", flush=True); run = 0.0
    torch.save(model.state_dict(), RES / "ckpt" / f"{args.name}.pt")
    return model


# ------------------------------------------------------------------ inference
@torch.no_grad()
def game(model, pr, q, texts):
    n = len(texts)
    a = score(model, pr, q, [(None, t) for t in texts])
    pairs = [(texts[i], texts[j]) for i in range(n) for j in range(n) if i != j]
    c = score(model, pr, q, pairs)
    C = torch.zeros(n, n, device=a.device)
    k = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                C[i, j] = c[k]; k += 1
    D = C - a[None, :]                      # D[i, j] = change of j's score when i is known
    D.fill_diagonal_(0)
    B = 0.5 * (D + D.T)
    return a, B, C


@torch.no_grad()
def evaluate(model, args, tag):
    model.eval(); pr = Pairer()
    summary = {}
    for ds in args.datasets.split(","):
        docs, qs = load_bench(ds)
        pools = jload(RES / "pools" / f"{args.pool_file}_{ds}.json")
        if args.n_test:
            qs, pools = qs[: args.n_test], pools[: args.n_test]
        out, m = [], {"R@2": [], "R@5": [], "AR@5": [], "uR@5": []}
        cache = []
        for q, pool in zip(qs, pools):
            a, B, Cm = game(model, pr, q["question"], [docs[i] for i in pool])
            cache.append({"a": a.cpu(), "C": Cm.cpu()})
            if args.unary_only:
                B = torch.zeros_like(B)
            phi = a + 0.5 * B.sum(-1)
            cand = list(range(len(pool)))
            s2 = best_coalition(a, B, cand, 2); s5 = best_coalition(a, B, cand, 5)
            s5 = sorted(s5, key=lambda i: -phi[i].item())
            u5 = torch.argsort(a, descending=True)[:5].tolist()
            out.append({"id": q["id"], "top2": [pool[i] for i in s2], "top5": [pool[i] for i in s5],
                        "rank": [pool[i] for i in torch.argsort(phi, descending=True).tolist()],
                        "unary_top5": [pool[i] for i in u5]})
            r5 = recall_at(out[-1]["top5"], q["gold"])
            m["R@2"].append(recall_at(out[-1]["top2"], q["gold"])); m["R@5"].append(r5); m["AR@5"].append(float(r5 == 1))
            m["uR@5"].append(recall_at(out[-1]["unary_top5"], q["gold"]))
        summary[ds] = {k: float(np.mean(v)) for k, v in m.items()}
        jdump(out, RES / "retrieval" / f"{tag}_{ds}.json")
        (RES / "cache").mkdir(exist_ok=True)
        torch.save(cache, RES / "cache" / f"{tag}_{ds}.pt")
        print(ds, summary[ds], flush=True)
    jdump(summary, RES / "retrieval" / f"{tag}_summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="cdn")
    ap.add_argument("--train", default="train_open.jsonl")
    ap.add_argument("--n_train", type=int, default=0)
    ap.add_argument("--n_test", type=int, default=0)
    ap.add_argument("--n_cand", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--pool_file", default="exp20_s5_k8")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--eval_only", default="")
    ap.add_argument("--unary_only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    (RES / "ckpt").mkdir(parents=True, exist_ok=True)
    if args.eval_only:
        model = AutoModelForSequenceClassification.from_pretrained(ENC).cuda()
        if args.eval_only != "zero":
            model.load_state_dict(torch.load(RES / "ckpt" / f"{args.eval_only}.pt"))
    else:
        model = train(args)
    evaluate(model, args, args.name)


if __name__ == "__main__":
    main()
