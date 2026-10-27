"""Build training questions from the official train splits (disjoint from the 1,000 dev-derived test questions).

Each item: {ds, id, question, answers, pool: ['title\\ntext', ...], gold: [pool idx]}.
Pools are the official candidate sets (HotpotQA/2Wiki: 10 paragraphs, MuSiQue: 20).
"""
import argparse
import json
import random
from pathlib import Path

import pandas as pd

from common import DATA, jl_dump, jload

RAW = DATA / "raw"


def from_ctx_parquet(paths, sep, name, test_ids):
    out = []
    for p in paths:
        d = pd.read_parquet(p)
        for r in d.itertuples():
            if r.id in test_ids:
                continue
            titles = list(r.context["title"]); sents = list(r.context["sentences"])
            pool = [t + "\n" + sep.join(list(s)) for t, s in zip(titles, sents)]
            gold_t = set(r.supporting_facts["title"])
            gold = [i for i, t in enumerate(titles) if t in gold_t]
            if len(gold) < 2:
                continue
            out.append({"ds": name, "id": r.id, "question": r.question, "answers": [r.answer], "pool": pool, "gold": gold,
                        "type": r.type})
    return out


def from_musique(path, test_ids):
    out = []
    with open(path, encoding="utf-8") as f:
        for l in f:
            j = json.loads(l)
            if j["id"] in test_ids or not j.get("answerable", True):
                continue
            pool = [p["title"] + "\n" + p["paragraph_text"] for p in j["paragraphs"]]
            gold = [i for i, p in enumerate(j["paragraphs"]) if p["is_supporting"]]
            out.append({"ds": "musique", "id": j["id"], "question": j["question"],
                        "answers": [j["answer"]] + j.get("answer_aliases", []), "pool": pool, "gold": gold,
                        "type": j["id"].split("__")[0]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="questions per dataset")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    test_ids = {ds: {s.get("id", s.get("_id")) for s in jload(DATA / f"{ds}.json")}
                for ds in ["musique", "2wikimultihopqa", "hotpotqa"]}
    sets = {
        "hotpotqa": from_ctx_parquet(sorted((RAW / "hotpot_qa/distractor").glob("train-*.parquet")), "", "hotpotqa", test_ids["hotpotqa"]),
        "2wikimultihopqa": from_ctx_parquet(sorted((RAW / "2WikiMultihopQA/data").glob("train-*.parquet")), " ", "2wikimultihopqa", test_ids["2wikimultihopqa"]),
        "musique": from_musique(RAW / "MuSiQue/musique_ans_v1.0_train.jsonl", test_ids["musique"]),
    }
    rng = random.Random(args.seed)
    allrows = []
    for ds, rows in sets.items():
        if ds == "hotpotqa":  # hard + medium bridge/comparison questions are the informative ones
            rows = [r for r in rows if r["type"] in ("bridge", "comparison")]
        rows = [r for r in rows if r["gold"] and len(r["pool"]) - len(r["gold"]) >= 3]
        rng.shuffle(rows)
        print(ds, "available", len(rows))
        allrows += rows[:args.n]
    rng.shuffle(allrows)
    jl_dump(allrows, DATA / "train" / f"train_{args.n}.jsonl")
    print("wrote", len(allrows))


if __name__ == "__main__":
    main()
