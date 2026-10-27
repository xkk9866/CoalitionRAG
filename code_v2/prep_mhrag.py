"""Out-of-domain benchmark: MultiHop-RAG (Tang & Yang, 2024; news articles from 2023, after the training data).

Articles are split into passages of whole sentences (at most 120 words). A gold passage is the passage that contains
an evidence fact of the query (exact match after whitespace normalisation, otherwise the passage with the largest
share of the fact's words). Null queries (no answer in the corpus) are dropped, and the first 1,000 remaining
queries form the test set. Writes data/multihoprag_corpus.json and data/multihoprag.json in the MuSiQue layout.
"""
import json
import re

from common import DATA, jdump

MAXW = 120


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def chunks(body):
    out, cur = [], []
    for s in sentences(body):
        if cur and len(" ".join(cur + [s]).split()) > MAXW:
            out.append(" ".join(cur)); cur = []
        cur.append(s.strip())
    if cur:
        out.append(" ".join(cur))
    return out


def norm(s):
    return " ".join(re.findall(r"\w+", s.lower()))


def main():
    files = {n: DATA / "mhrag_raw" / n for n in ("corpus.json", "MultiHopRAG.json")}   # the released files
    corpus_raw = json.load(open(files["corpus.json"], encoding="utf-8"))
    queries = json.load(open(files["MultiHopRAG.json"], encoding="utf-8"))
    corpus, by_title = [], {}
    for a in corpus_raw:
        for c in chunks(a["body"]):
            by_title.setdefault(a["title"], []).append(len(corpus))
            corpus.append({"title": a["title"], "text": c})
    normed = [norm(d["text"]) for d in corpus]
    samples, dropped = [], 0
    for qi, q in enumerate(queries):
        if q.get("question_type") == "null_query" or not q.get("evidence_list"):
            continue
        gold, ok = [], True
        for ev in q["evidence_list"]:
            ids = by_title.get(ev["title"], [])
            f = norm(ev["fact"])
            hit = [i for i in ids if f and f in normed[i]]
            if not hit and ids:
                fw = set(f.split())
                share = [(len(fw & set(normed[i].split())) / max(1, len(fw)), i) for i in ids]
                best = max(share)
                hit = [best[1]] if best[0] >= 0.5 else []
            if not hit:
                ok = False; break
            if hit[0] not in gold:
                gold.append(hit[0])
        if not ok:
            dropped += 1; continue
        samples.append({"id": f"mhrag_{qi}", "question": q["query"], "answer": q["answer"], "type": q["question_type"],
                        "paragraphs": [{"title": corpus[i]["title"], "paragraph_text": corpus[i]["text"], "is_supporting": True}
                                       for i in gold]})
    print(f"passages {len(corpus)}  usable queries {len(samples)}  dropped {dropped}")
    samples = samples[:1000]
    jdump(corpus, DATA / "multihoprag_corpus.json")
    jdump(samples, DATA / "multihoprag.json")
    from collections import Counter
    print("types", Counter(s["type"] for s in samples), "gold per query",
          sum(len(s["paragraphs"]) for s in samples) / len(samples))


if __name__ == "__main__":
    main()
