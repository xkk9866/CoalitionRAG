"""How many 2Wiki and MuSiQue gold titles exist in the 2017 Hotpot abstract index."""

from __future__ import annotations

import pickle

from datasets import load_dataset

import open_retrieve as ore


def main() -> None:
    with ore.TITLE_PATH.open("rb") as handle:
        titles = pickle.load(handle)
    print("index titles", len(titles), flush=True)

    wiki = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    n = hit = q = qfull = 0
    for row in wiki:
        gold = list(dict.fromkeys(row["supporting_facts"]["title"]))
        q += 1
        ok = 0
        for title in gold:
            key = ore.title_key(title)
            n += 1
            if key and key in titles:
                hit += 1
                ok += 1
        if gold and ok == len(gold):
            qfull += 1
    print(f"2wiki gold_pages {hit}/{n} questions_complete {qfull}/{q}", flush=True)

    musique = load_dataset("dgslibisey/MuSiQue", split="validation")
    n = hit = q = qfull = 0
    for row in musique:
        gold = [p["title"] for p in row["paragraphs"] if p["is_supporting"]]
        gold = list(dict.fromkeys(gold))
        q += 1
        ok = 0
        for title in gold:
            key = ore.title_key(title)
            n += 1
            if key and key in titles:
                hit += 1
                ok += 1
        if gold and ok == len(gold):
            qfull += 1
    print(f"musique gold_pages {hit}/{n} questions_complete {qfull}/{q}", flush=True)


if __name__ == "__main__":
    main()
