"""Shared paths, data loading, metrics and reader prompt for CoalitionRAG v2."""
import json
import os
import re
import string
from collections import Counter
from pathlib import Path

ROOT = Path(os.environ.get("CRAG_ROOT", Path(__file__).resolve().parents[1]))
DATA = ROOT / "data"
RES = ROOT / "results"
MODELS = {
    "qwen25-7b-awq": str(ROOT / "models" / "Qwen2.5-7B-Instruct-AWQ"),
    "qwen3-8b-awq": str(ROOT / "models" / "Qwen3-8B-AWQ"),
}
DATASETS = os.environ.get("CRAG_DATASETS", "musique,2wikimultihopqa,hotpotqa").split(",")
SHORT = {"musique": "MuSiQue", "2wikimultihopqa": "2Wiki", "hotpotqa": "HotpotQA", "multihoprag": "MultiHop-RAG"}


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def jdump(obj, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def jmerge(obj, p):
    """Writes a per-dataset summary dict, keeping datasets written by earlier runs of the same tag."""
    old = jload(p) if Path(p).exists() else {}
    for k, v in obj.items():
        if isinstance(v, dict) and isinstance(old.get(k), dict):
            old[k].update(v)
        else:
            old[k] = v
    jdump(old, p)


def jl_load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def jl_dump(rows, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- test benchmark (HippoRAG protocol)
def _gold_docs(sample, name):
    """Gold passages as 'title\\ntext' strings, identical to HippoRAG's main.get_gold_docs."""
    if "supporting_facts" in sample:
        titles = {t for t, _ in sample["supporting_facts"]}
        ctx = [c for c in sample["context"] if c[0] in titles]
        sep = "" if name.startswith("hotpotqa") else " "
        gold = [c[0] + "\n" + sep.join(c[1]) for c in ctx]
    else:
        paras = [p for p in sample["paragraphs"] if p.get("is_supporting", True)]
        gold = [p["title"] + "\n" + (p["text"] if "text" in p else p["paragraph_text"]) for p in paras]
    return list(dict.fromkeys(gold))


def _answers(sample):
    a = sample["answer"]
    ans = [a] if isinstance(a, str) else list(a)
    ans.extend(sample.get("answer_aliases", []))
    return list(dict.fromkeys(ans))


def load_bench(name):
    """Returns (docs, questions). docs: list of 'title\\ntext'. Each question has gold doc ids."""
    corpus = jload(DATA / f"{name}_corpus.json")
    docs = [f"{d['title']}\n{d['text']}" for d in corpus]
    index = {d: i for i, d in enumerate(docs)}
    qs = []
    for s in jload(DATA / f"{name}.json"):
        gold = _gold_docs(s, name)
        gid = [index[g] for g in gold if g in index]
        qs.append({
            "id": s.get("id", s.get("_id")),
            "question": s["question"],
            "answers": _answers(s),
            "gold": gid,
            "n_gold_text": len(gold),
            "hops": int(s["id"][0]) if name == "musique" else None,
            "type": s.get("type"),
        })
    return docs, qs


# ---------------------------------------------------------------- metrics (MuSiQue/HippoRAG official style)
def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em_f1(pred, golds):
    p = normalize_answer(pred)
    em, f1 = 0.0, 0.0
    for g in golds:
        g = normalize_answer(g)
        em = max(em, float(p == g))
        pt, gt = p.split(), g.split()
        common = Counter(pt) & Counter(gt)
        ns = sum(common.values())
        if ns == 0:
            continue
        pr, rc = ns / len(pt), ns / len(gt)
        f1 = max(f1, 2 * pr * rc / (pr + rc))
    return em, f1


def recall_at(ranked_or_set, gold, k=None):
    sel = ranked_or_set[:k] if k is not None else ranked_or_set
    if not gold:
        return 0.0
    return len(set(sel) & set(gold)) / len(set(gold))


# ---------------------------------------------------------------- reader prompt
SYSTEM = ("You answer multi-hop questions using the given Wikipedia passages. "
          "Reason over the passages silently and reply with only the final short answer "
          "(an entity, date, number, or yes/no). Do not explain.")


def passage_block(docs, ids):
    lines = []
    for r, i in enumerate(ids, 1):
        title, _, text = docs[i].partition("\n")
        lines.append(f"Passage {r} (Title: {title}): {text}")
    return "\n\n".join(lines)


def reader_messages(question, docs, ids):
    ctx = passage_block(docs, ids) if ids else "(no passages)"
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{ctx}\n\nQuestion: {question}\nAnswer:"},
    ]


def clean_pred(text):
    t = text.strip().split("\n")[0]
    t = re.sub(r"^(the )?answer( is)?\s*[:：]?\s*", "", t, flags=re.I)
    return t.strip().strip(".").strip()
