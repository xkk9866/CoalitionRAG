"""First-stage retrieval on the HippoRAG benchmarks: BM25, Contriever, bge-large-en-v1.5.

Writes results/first_stage/{retriever}_{dataset}.json = {"ranked": [[doc ids]], "scores": [[...]]}
"""
import argparse
import re

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from common import DATASETS, RES, jdump, load_bench, recall_at

TOPN = 100


def bm25_rank(docs, qs):
    from rank_bm25 import BM25Okapi
    tok = lambda s: re.findall(r"\w+", s.lower())
    bm = BM25Okapi([tok(d) for d in docs])
    ranked, scores = [], []
    for q in qs:
        s = bm.get_scores(tok(q["question"]))
        top = np.argsort(-s)[:TOPN]
        ranked.append(top.tolist()); scores.append(s[top].tolist())
    return ranked, scores


@torch.no_grad()
def encode(model, tok, texts, pool, bs=128, max_len=512):
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to("cuda")
        h = model(**b).last_hidden_state
        if pool == "cls":
            e = h[:, 0]
        else:
            m = b["attention_mask"].unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1)
        if pool == "cls":
            e = torch.nn.functional.normalize(e, dim=-1)
        out.append(e.float().cpu())
    return torch.cat(out)


def dense_rank(docs, qs, name):
    if name == "contriever":
        path, pool, qpref = "facebook/contriever", "mean", ""
    else:
        path, pool, qpref = "BAAI/bge-large-en-v1.5", "cls", "Represent this sentence for searching relevant passages: "
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path, torch_dtype=torch.float16).cuda().eval()
    D = encode(model, tok, docs, pool)
    Q = encode(model, tok, [qpref + q["question"] for q in qs], pool)
    S = Q @ D.T
    sc, ix = S.topk(TOPN, dim=1)
    return ix.tolist(), sc.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrievers", default="bm25,contriever,bge")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    args = ap.parse_args()
    for ds in args.datasets.split(","):
        docs, qs = load_bench(ds)
        for r in args.retrievers.split(","):
            ranked, scores = bm25_rank(docs, qs) if r == "bm25" else dense_rank(docs, qs, r)
            jdump({"ranked": ranked, "scores": scores}, RES / "first_stage" / f"{r}_{ds}.json")
            msg = " ".join(f"R@{k}={np.mean([recall_at(rk, q['gold'], k) for rk, q in zip(ranked, qs)]):.3f}"
                           for k in (2, 5, 10, 20, 30, 50, 100))
            print(f"{ds:16s} {r:10s} {msg}", flush=True)


if __name__ == "__main__":
    main()
