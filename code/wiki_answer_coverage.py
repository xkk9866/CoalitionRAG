"""Whether 2Wiki and MuSiQue answers occur in the 2017 introductory abstracts."""

from __future__ import annotations

import random
import sqlite3

from datasets import load_dataset

import open_retrieve as ore


def bodies_for(con: sqlite3.Connection, titles: list[str]) -> str:
    chunks = []
    for title in titles:
        key = ore.title_key(title)
        row = con.execute("SELECT rowid FROM meta WHERE title_key = ?", (key,)).fetchone()
        if row is None:
            continue
        body = con.execute("SELECT body FROM pages WHERE rowid = ?", (int(row[0]),)).fetchone()
        if body is not None:
            chunks.append(body[0] or "")
    return "\n".join(chunks).lower()


def contains_answer(text: str, answer: str) -> bool:
    answer = " ".join(answer.lower().split())
    return bool(answer) and answer in " ".join(text.split())


def main() -> None:
    con = sqlite3.connect(ore.DB_PATH)
    con.execute("PRAGMA query_only=ON")
    rng = random.Random(0)

    wiki = list(load_dataset("framolfese/2WikiMultihopQA", split="validation"))
    rng.shuffle(wiki)
    hit = n = 0
    for row in wiki[:400]:
        gold = list(dict.fromkeys(row["supporting_facts"]["title"]))
        n += 1
        if contains_answer(bodies_for(con, gold), str(row["answer"])):
            hit += 1
    print(f"2wiki answer_in_2017_abstracts {hit}/{n}", flush=True)

    musique = list(load_dataset("dgslibisey/MuSiQue", split="validation"))
    rng.shuffle(musique)
    hit = n = 0
    for row in musique[:400]:
        gold = [p["title"] for p in row["paragraphs"] if p["is_supporting"]]
        n += 1
        if contains_answer(bodies_for(con, gold), str(row["answer"])):
            hit += 1
    print(f"musique answer_in_2017_abstracts {hit}/{n}", flush=True)
    con.close()


if __name__ == "__main__":
    main()
