import open_retrieve as ore
import run_open as ro

rows = {row["id"]: row for row in ro.load_hotpot()}
db = ore.WikiDB()
try:
    for qid in ("5adf37a95542995ec70e8f97", "5ab3e45655429976abd1bcd4"):
        row = rows[qid]
        print("Q", row["question"])
        print("GOLD", row["gold_titles"])
        found = ore.maximal_title_matches(ore.title_tokens(row["question"]), db.titles)
        print("MATCHES")
        pages = db.fetch([m.rowid for m in found])
        for mention in found:
            page = pages.get(mention.rowid)
            print(" ", mention.phrase, "->", page.title if page else mention.rowid)
        print()
finally:
    db.close()
