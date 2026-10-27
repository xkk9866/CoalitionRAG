"""Coalition Value Network (CVN): a learned 2-additive cooperative game over candidate passages.

v(S) = sum_{i in S} a_i + sum_{i<j in S} b_ij
  a_i  : unary Harsanyi dividend (pretrained relevance prior + set-contextualised term)
  b_ij : pairwise Harsanyi dividend from a neural interaction module (salience-weighted
         cross-passage late interaction over question-conditioned token states + set context)
Trained to match the frozen reader's utilities on sampled coalitions; selection is exact
argmax over all size-k coalitions of the top-m pool.
"""
import argparse
import itertools
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import DATA, DATASETS, RES, jdump, jl_load, jload, load_bench, recall_at

ENC = "BAAI/bge-reranker-v2-m3"


class CVN(nn.Module):
    def __init__(self, d=256, dt=128, layers=2, use_pair=True, use_late=True, use_set=True):
        super().__init__()
        self.enc = AutoModelForSequenceClassification.from_pretrained(ENC, torch_dtype=torch.float32)
        self.enc.config.output_hidden_states = True
        H = self.enc.config.hidden_size
        self.use_pair, self.use_late, self.use_set = use_pair, use_late, use_set
        self.proj_h = nn.Linear(H, d)
        if use_set:
            layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
            self.set_tf = nn.TransformerEncoder(layer, layers)
        self.unary = nn.Linear(d, 1)
        nn.init.zeros_(self.unary.weight); nn.init.zeros_(self.unary.bias)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        if use_pair:
            self.proj_tok = nn.Linear(H, dt)
            self.sal = nn.Linear(H, 1)
            nin = d + (1 if use_late else 0)
            self.pair = nn.Sequential(nn.Linear(nin, d), nn.GELU(), nn.Linear(d, 1))
            nn.init.zeros_(self.pair[-1].weight); nn.init.zeros_(self.pair[-1].bias)

    def forward(self, ids, att, pmask):
        """ids/att: (n, L) question+passage pairs; pmask: (n, L) 1 on passage tokens. Returns a (n,), B (n,n)."""
        out = self.enc(input_ids=ids, attention_mask=att)
        prior = out.logits.squeeze(-1)                      # pretrained cross-encoder relevance
        hs = out.hidden_states[-1]                           # (n, L, H)
        x = self.proj_h(hs[:, 0])
        z = self.set_tf(x.unsqueeze(0)).squeeze(0) if self.use_set else x
        a = self.alpha * prior + self.unary(z).squeeze(-1)
        n = a.shape[0]
        if not self.use_pair:
            return a, torch.zeros(n, n, device=a.device, dtype=a.dtype)
        feats = [z.unsqueeze(1) * z.unsqueeze(0)]                              # coalition-context agreement
        if self.use_late:
            P = F.normalize(self.proj_tok(hs), dim=-1)                       # (n, L, dt)
            sal = self.sal(hs).squeeze(-1).masked_fill(pmask == 0, -1e4)
            w = torch.softmax(sal, dim=-1)                                     # token salience per passage
            sim = torch.einsum("itd,jsd->ijts", P, P)                          # (n, n, L, L)
            sim = sim.masked_fill(pmask[None, :, None, :] == 0, -1e4)
            mx = sim.max(-1).values                                            # (n, n, L): best match of i-token in j
            m = (mx * w[:, None, :]).sum(-1)                                   # m_ij
            feats.append((0.5 * (m + m.T)).unsqueeze(-1))                        # symmetric cross-passage match
        b = self.pair(torch.cat(feats, -1)).squeeze(-1)
        b = 0.5 * (b + b.T)
        b = b * (1 - torch.eye(n, device=b.device, dtype=b.dtype))
        return a, b


def coalition_values(a, B, cols):
    """cols: (K, n) 0/1 float. v = cols@a + 0.5 * diag(cols B cols^T)."""
    return cols @ a + 0.5 * ((cols @ B) * cols).sum(-1)


def shapley(a, B):
    return a + 0.5 * B.sum(-1)


# ------------------------------------------------------------------ encoding
class Encoder:
    def __init__(self, max_len=320, qmax=64):
        self.tok = AutoTokenizer.from_pretrained(ENC)
        self.max_len, self.qmax = max_len, qmax

    def __call__(self, question, passages):
        t = self.tok
        q = t(question, add_special_tokens=False)["input_ids"][: self.qmax]
        rows, pm = [], []
        for p in passages:
            title, _, text = p.partition("\n")
            pi = t(f"{title}: {text}", add_special_tokens=False)["input_ids"][: self.max_len - len(q) - 4]
            ids = [t.cls_token_id] + q + [t.sep_token_id, t.sep_token_id] + pi + [t.sep_token_id]
            rows.append(ids); pm.append([0] * (len(q) + 3) + [1] * len(pi) + [0])
        L = max(map(len, rows))
        ids = torch.full((len(rows), L), t.pad_token_id); att = torch.zeros(len(rows), L, dtype=torch.long)
        pmask = torch.zeros(len(rows), L, dtype=torch.long)
        for i, (r, m) in enumerate(zip(rows, pm)):
            ids[i, :len(r)] = torch.tensor(r); att[i, :len(r)] = 1; pmask[i, :len(m)] = torch.tensor(m)
        return ids, att, pmask


# ------------------------------------------------------------------ training
def load_train(train_file, util_files):
    T = {r["id"]: r for r in jl_load(DATA / "train" / train_file)}
    items = []
    for f in util_files:
        for u in jl_load(DATA / "train" / f):
            r = T.get(u["qid"])
            if r is not None:
                items.append((r, u))
    return items


def loss_fn(a, B, r, u, args):
    n = a.shape[0]
    cols = torch.zeros(len(u["coalitions"]), n, device=a.device)
    for k, c in enumerate(u["coalitions"]):
        if c:
            cols[k, c] = 1.0
    v = coalition_values(a, B, cols)
    y = torch.tensor(u["u"], device=a.device)
    loss = torch.zeros((), device=a.device)
    if args.sup in ("util", "both"):
        p = torch.softmax(y / args.tau, -1)
        loss = loss + F.kl_div(torch.log_softmax(v, -1), p, reduction="sum")
        vc, yc = v - v.mean(), y - y.mean()
        corr = (vc * yc).sum() / (vc.norm() * yc.norm() + 1e-6)
        loss = loss + args.w_corr * (1 - corr)
    if args.sup in ("gold", "both"):
        lab = torch.zeros(n, device=a.device); lab[r["gold"]] = 1.0
        w = args.w_gold if args.sup == "both" else 1.0
        loss = loss + w * F.binary_cross_entropy_with_logits(a, lab)
        if args.use_pair:   # pair supervision: evidence co-membership
            pl = lab[:, None] * lab[None, :]
            iu = torch.triu_indices(n, n, 1, device=a.device)
            loss = loss + w * F.binary_cross_entropy_with_logits(B[iu[0], iu[1]], pl[iu[0], iu[1]])
    return loss


def train(args):
    torch.manual_seed(args.seed); random.seed(args.seed)
    if args.sup == "gold":
        items = [(r, {"coalitions": [], "u": []}) for r in jl_load(DATA / "train" / args.train)]
    else:
        items = load_train(args.train, args.util.split(","))
    if args.n_train:
        items = items[: args.n_train]
    print("train items", len(items), flush=True)
    model = CVN(use_pair=args.use_pair, use_late=args.use_late, use_set=args.use_set).cuda()
    model.enc.gradient_checkpointing_enable()
    enc = Encoder(args.max_len)
    head = [p for n_, p in model.named_parameters() if not n_.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": args.lr},
                             {"params": head, "lr": args.lr_head}], weight_decay=0.01)
    total = args.epochs * len(items) // args.accum
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / max(1, int(0.05 * total))) * max(0.0, 1 - s / max(1, total)))
    step, t0 = 0, time.time()
    model.train()
    for ep in range(args.epochs):
        random.shuffle(items)
        run = 0.0
        for it, (r, u) in enumerate(items):
            ids, att, pm = enc(r["question"], r["pool"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                a, B = model(ids.cuda(), att.cuda(), pm.cuda())
            loss = loss_fn(a.float(), B.float(), r, u, args) / args.accum
            loss.backward(); run += loss.item()
            if (it + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if step % 100 == 0:
                    print(f"ep {ep} step {step}/{total} loss {run / 100:.4f} {time.time() - t0:.0f}s", flush=True); run = 0.0
    torch.save({k: v for k, v in model.state_dict().items()}, RES / "ckpt" / f"{args.name}.pt")
    return model


# ------------------------------------------------------------------ inference
_COMBOS = {}


def _combos(n, k, device):
    if (n, k) not in _COMBOS:
        _COMBOS[(n, k)] = torch.tensor(list(itertools.combinations(range(n), k)), device=device)
    return _COMBOS[(n, k)]


@torch.no_grad()
def best_coalition(a, B, cand, k):
    """Exact argmax of v over all size-k subsets of positions `cand`."""
    k = min(k, len(cand))
    combos = _combos(len(cand), k, a.device)
    aa, BB = a[cand], B[cand][:, cand]
    v = aa[combos].sum(-1)
    for x, y in itertools.combinations(range(k), 2):
        v = v + BB[combos[:, x], combos[:, y]]
    best = combos[v.argmax()].tolist()
    return [cand[i] for i in best]


@torch.no_grad()
def evaluate(model, args, tag):
    model.eval()
    enc = Encoder(args.max_len)
    summary = {}
    for ds in DATASETS:
        docs, qs = load_bench(ds)
        if args.pool_file:
            fs = jload(RES / "pools" / f"{args.pool_file}_{ds}.json")
        else:
            fs = [r[: args.pool] for r in jload(RES / "first_stage" / f"{args.first}_{ds}.json")["ranked"]]
        if args.n_test:
            qs, fs = qs[: args.n_test], fs[: args.n_test]
        out, r2, r5, ar5 = [], [], [], []
        for q, ranked in zip(qs, fs):
            pool = ranked
            ids, att, pm = enc(q["question"], [docs[i] for i in pool])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                a, B = model(ids.cuda(), att.cuda(), pm.cuda())
            a, B = a.float(), B.float()
            cand = torch.argsort(shapley(a, B), descending=True)[: (args.m or len(pool))].tolist()
            s2 = best_coalition(a, B, cand, 2)
            s5 = best_coalition(a, B, cand, 5)
            phi = shapley(a, B)
            s5 = sorted(s5, key=lambda i: -phi[i].item())
            rank_all = torch.argsort(phi, descending=True).tolist()
            out.append({"id": q["id"], "top2": [pool[i] for i in s2], "top5": [pool[i] for i in s5],
                        "rank": [pool[i] for i in rank_all]})
            r2.append(recall_at([pool[i] for i in s2], q["gold"])); r5.append(recall_at([pool[i] for i in s5], q["gold"]))
            ar5.append(float(r5[-1] == 1.0))
        summary[ds] = {"R@2": float(np.mean(r2)), "R@5": float(np.mean(r5)), "AR@5": float(np.mean(ar5)), "n": len(qs)}
        jdump(out, RES / "retrieval" / f"{tag}_{ds}.json")
        print(ds, summary[ds], flush=True)
    jdump(summary, RES / "retrieval" / f"{tag}_summary.json")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="cvn")
    ap.add_argument("--train", default="train_open.jsonl")
    ap.add_argument("--util", default="util_qwen25-7b-awq_train_open_s0.jsonl,util_qwen25-7b-awq_train_open_s1.jsonl")
    ap.add_argument("--n_train", type=int, default=0)
    ap.add_argument("--n_test", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lr_head", type=float, default=2e-4)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--w_corr", type=float, default=1.0)
    ap.add_argument("--w_gold", type=float, default=0.5)
    ap.add_argument("--sup", default="util", choices=["util", "gold", "both"])
    ap.add_argument("--no_pair", dest="use_pair", action="store_false")
    ap.add_argument("--no_late", dest="use_late", action="store_false")
    ap.add_argument("--no_set", dest="use_set", action="store_false")
    ap.add_argument("--max_len", type=int, default=320)
    ap.add_argument("--first", default="bge")
    ap.add_argument("--pool", type=int, default=20)
    ap.add_argument("--pool_file", default="", help="results/pools/<name>_<ds>.json candidate pools")
    ap.add_argument("--m", type=int, default=0, help="restrict exact search to top-m by Shapley (0 = whole pool)")
    ap.add_argument("--zero_shot", action="store_true", help="evaluate untrained prior only")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    (RES / "ckpt").mkdir(parents=True, exist_ok=True)
    if args.zero_shot:
        model = CVN(use_pair=False, use_set=False).cuda()
    else:
        model = train(args)
    evaluate(model, args, args.name)


if __name__ == "__main__":
    main()
