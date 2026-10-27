"""Probe whether an unbound proper name retrieves the page BM25 missed."""

from __future__ import annotations

import sqlite3

import open_retrieve as ore

db = ore.WikiDB()
phrases = [
    "richard melville hall",
    "amil grove",
    "amii grove",
    "great antonine altar",
    "1991 ms olympia",
    "ew",
]
try:
    for phrase in phrases:
        query = '"' + phrase + '"'
        try:
            rows = db.con.execute(
                "SELECT rowid FROM pages WHERE pages MATCH ? ORDER BY rank LIMIT 5",
                (query,),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            print(phrase, "ERR", exc)
            continue
        pages = db.fetch([int(row["rowid"]) for row in rows])
        print(phrase, [pages[int(row["rowid"])].title for row in rows if int(row["rowid"]) in pages])
finally:
    db.close()
