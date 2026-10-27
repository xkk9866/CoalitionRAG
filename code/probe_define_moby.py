import open_retrieve as ore

db = ore.WikiDB()
question = "What was the third studio album released by Richard Melville Hall?"
tokens = ore.title_tokens(question)
try:
    for n in range(min(6, len(tokens)), 1, -1):
        for i in range(len(tokens) - n + 1):
            span = tokens[i : i + n]
            long_toks = [tok for tok in span if len(tok) >= 5]
            if len(long_toks) < 2:
                continue
            query = '"' + " ".join(span) + '"'
            rows = db.con.execute(
                "SELECT rowid FROM pages WHERE pages MATCH ? LIMIT 2",
                (query,),
            ).fetchall()
            if len(rows) != 1:
                continue
            page = db.fetch([int(rows[0]["rowid"])])[int(rows[0]["rowid"])]
            print(n, " ".join(span), "->", page.title)
finally:
    db.close()
