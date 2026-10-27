"""Training pools that match test-time candidate generation (fixes shortcut learning on distractor pools).

Per dataset: corpus = union of all candidate passages of its training questions (HippoRAG protocol on the
train side); pool = bge-large top-20 + coalition recruitment ([q; seed] retrieval, 5 seeds x 8).
For training we keep the gold passages plus 18 negatives (12 dense, 6 recruited) so n <= ~21.
"""
import random

import torch
from transformers import AutoModel, AutoTokenizer

from common import DATA, jl_dump, jl_load, recall_at
from first_stage import encode

QP = "Represent this sentence for searching relevant passages: "


def main():
    rows = jl_load(DATA / "train" / "train_4000.jsonl")
    tok = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5", torch_dtype=torch.float16).cuda().eval()
    out = []
    for ds in ["hotpotqa", "2wikimultihopqa", "musique"]:
        R = [r for r in rows if r["ds"] == ds]
        docs = list(dict.fromkeys(p for r in R for p in r["pool"]))
        index = {d: i for i, d in enumerate(docs)}
        D = encode(model, tok, docs, "cls").cuda()
        Q = encode(model, tok, [QP + r["question"] for r in R], "cls").cuda()
        top = (Q @ D.T).topk(20, dim=1).indices.tolist()
        seeds_q, owners = [], []
        for qi, (r, t) in enumerate(zip(R, top)):
            for s in t[:5]:
                title, _, text = docs[s].partition("\n")
                seeds_q.append(f"{QP}{r['question']} {title}: {text}"); owners.append((qi, s))
        rec = [dict() for _ in R]
        for b in range(0, len(seeds_q), 4096):
            E = encode(model, tok, seeds_q[b:b + 4096], "cls").cuda()
            tt = (E @ D.T).topk(9, dim=1).indices.tolist()
            for (qi, s), cand in zip(owners[b:b + 4096], tt):
                for c in cand:
                    if c != s and c not in top[qi]:
                        rec[qi][c] = 1
        hit_d, hit_p = [], []
        for qi, r in enumerate(R):
            gold = [index[r["pool"][g]] for g in r["gold"]]
            dense, part = top[qi], list(rec[qi])
            hit_d.append(recall_at(dense, gold)); hit_p.append(recall_at(dense + part, gold))
            rng = random.Random(qi)
            negd = [i for i in dense if i not in gold]; negp = [i for i in part if i not in gold]
            keep = gold + negd[:12] + rng.sample(negp, min(6, len(negp)))
            rng.shuffle(keep)
            out.append({"ds": ds, "id": r["id"], "question": r["question"], "answers": r["answers"],
                        "pool": [docs[i] for i in keep], "gold": [keep.index(g) for g in gold], "type": r.get("type")})
        print(f"{ds}: corpus {len(docs)} dense@20 {sum(hit_d)/len(R):.3f} +recruit {sum(hit_p)/len(R):.3f}", flush=True)
    random.Random(0).shuffle(out)
    jl_dump(out, DATA / "train" / "train_open.jsonl")
    print("wrote", len(out))


if __name__ == "__main__":
    main()
