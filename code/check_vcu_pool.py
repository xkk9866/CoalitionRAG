import open_retrieve as ore

q = (
    "The 2011\u201312 VCU Rams men's basketball team, led by third year head coach "
    "Shaka Smart, represented Virginia Commonwealth University which was founded in what year?"
)
db = ore.WikiDB()
try:
    ids = db.search(q, 100)
    pages = db.fetch(ids)
    keys = {pages[i].key: pages[i].title for i in ids if i in pages}
    for key in (
        "2011 12 vcu rams men s basketball team",
        "virginia commonwealth university",
    ):
        print(key, "in top100" if key in keys else "OUTSIDE", keys.get(key, ""))
    print("top5", [pages[i].title for i in ids[:5] if i in pages])
finally:
    db.close()
