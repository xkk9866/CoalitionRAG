"""Time FTS phrase lookups used by a unique-definition bind."""

import time

import open_retrieve as ore

db = ore.WikiDB()
question = "What was the third studio album released by Richard Melville Hall?"
tokens = ore.title_tokens(question)
t0 = time.perf_counter()
hits = 0
queries = 0
try:
    for _ in range(20):
        for n in range(min(6, len(tokens)), 1, -1):
            found = False
            for i in range(len(tokens) - n + 1):
                span = tokens[i : i + n]
                if not any(len(tok) >= 4 for tok in span):
                    continue
                query = '"' + " ".join(span) + '"'
                queries += 1
                rows = db.con.execute(
                    "SELECT rowid FROM pages WHERE pages MATCH ? LIMIT 2",
                    (query,),
                ).fetchall()
                if len(rows) == 1:
                    hits += 1
                    found = True
                    break
            if found:
                break
finally:
    db.close()
elapsed = time.perf_counter() - t0
print(f"queries {queries} hits {hits} sec {elapsed:.2f} per_q {elapsed / 20:.3f}")
