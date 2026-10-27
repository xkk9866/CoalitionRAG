"""Build a SQLite FTS index of the HotpotQA Wikipedia introductory paragraphs.

The source archive is the official 1.55GB abstract release, not the 10-page
distractor pools. Run this only after that archive has finished downloading.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import pickle
import sqlite3
import tarfile
import time
from pathlib import Path

from open_retrieve import (
    DB_PATH,
    LOCAL_DIR,
    PAGE_WORDS,
    TITLE_PATH,
    WIKI_DIR,
    clip_words,
    find_span,
    parse_links,
    title_key,
    title_tokens,
)

ARCHIVE = WIKI_DIR / "enwiki-abstracts.tar.bz2"
EXPECTED_MD5 = "01edf64cd120ecc03a2745352779514c"
EXPECTED_BYTES = 1553565403


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def iter_part_objects(directory: Path):
    paths = sorted(directory.glob("part_*.jsonl"))
    if not paths:
        raise SystemExit(f"no part files in {directory}")
    buf = b""
    for path in paths:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1 << 20)
                if not chunk:
                    break
                buf += chunk
                while True:
                    cut = buf.find(b"\n")
                    if cut < 0:
                        break
                    line = buf[:cut].decode("utf-8", "replace").strip()
                    buf = buf[cut + 1 :]
                    if not line.startswith("{"):
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
    tail = buf.decode("utf-8", "replace").strip()
    if tail.startswith("{"):
        try:
            yield json.loads(tail)
        except json.JSONDecodeError:
            return


def iter_objects(path: Path):
    try:
        tar = tarfile.open(path, "r:bz2")
    except tarfile.ReadError:
        tar = None
    if tar is None:
        with bz2.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        return
    with tar:
        for member in tar:
            if not member.isfile() or member.size <= 0:
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            name = member.name.lower()
            stream = bz2.BZ2File(extracted) if name.endswith(".bz2") else extracted
            for raw in stream:
                if isinstance(raw, bytes):
                    line = raw.decode("utf-8", "replace").strip()
                else:
                    line = str(raw).strip()
                if not line.startswith("{"):
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def page_record(obj: dict) -> tuple[str, str, str, str] | None:
    title = str(obj.get("title") or "").strip()
    key = title_key(title)
    if not title or not key:
        return None
    linked = obj.get("text_with_links")
    if linked is None:
        linked = obj.get("text")
    plain, links = parse_links(linked if isinstance(linked, str) else _flatten(linked))
    if not plain:
        plain = _flatten(obj.get("text"))
        links = []
    plain = clip_words(plain, PAGE_WORDS)
    if not plain:
        return None
    tokens = title_tokens(plain)
    kept = []
    seen = set()
    for link in links:
        target = link.get("t") or ""
        if not target or target in seen:
            continue
        anchor = title_tokens(link.get("a") or "")
        if anchor and find_span(tokens, anchor) is None:
            continue
        seen.add(target)
        kept.append({"a": link.get("a") or "", "t": target})
        if len(kept) >= 30:
            break
    return title, key, plain, json.dumps(kept, ensure_ascii=False)


def _flatten(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_flatten(part) for part in value)
    return str(value)


def build(limit: int | None, parts: Path | None) -> None:
    if parts is not None:
        source = iter_part_objects(parts)
    else:
        if not ARCHIVE.exists():
            raise SystemExit(f"missing archive: {ARCHIVE}")
        size = ARCHIVE.stat().st_size
        if size != EXPECTED_BYTES:
            raise SystemExit(f"archive size {size} != {EXPECTED_BYTES}; download is incomplete")
        stamp = WIKI_DIR / "abstracts.md5.ok"
        if stamp.exists() and stamp.read_text(encoding="utf-8").strip() == EXPECTED_MD5:
            print("md5 already verified", flush=True)
        else:
            print("checking md5", flush=True)
            got = md5_file(ARCHIVE)
            if got != EXPECTED_MD5:
                raise SystemExit(f"md5 {got} != {EXPECTED_MD5}")
            stamp.write_text(EXPECTED_MD5 + "\n", encoding="utf-8")
            print("md5 ok", flush=True)
        source = iter_objects(ARCHIVE)

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = DB_PATH.with_suffix(".sqlite.tmp")
    if tmp.exists():
        tmp.unlink()
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(tmp)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA locking_mode=EXCLUSIVE")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("PRAGMA cache_size=-400000")
    con.execute(
        "CREATE VIRTUAL TABLE pages USING fts5(title, body, tokenize='unicode61 remove_diacritics 0')"
    )
    con.execute("CREATE TABLE meta (rowid INTEGER PRIMARY KEY, title_key TEXT, links TEXT)")
    first_keys = None
    count = 0
    with_links = 0
    sample_written = False
    started = time.perf_counter()
    batch = 0
    con.execute("BEGIN")
    for obj in source:
        if first_keys is None:
            first_keys = sorted(obj.keys())
            print("first object keys:", first_keys, flush=True)
        rec = page_record(obj)
        if rec is None:
            continue
        title, key, body, links = rec
        count += 1
        if links != "[]":
            with_links += 1
            if not sample_written:
                (WIKI_DIR / "sample_page.json").write_text(
                    json.dumps({"title": title, "body": body, "links": json.loads(links)}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                sample_written = True
        con.execute("INSERT INTO pages(rowid, title, body) VALUES (?,?,?)", (count, title, body))
        con.execute("INSERT INTO meta(rowid, title_key, links) VALUES (?,?,?)", (count, key, links))
        batch += 1
        if batch >= 5000:
            con.execute("COMMIT")
            con.execute("BEGIN")
            batch = 0
            elapsed = time.perf_counter() - started
            print(f"indexed {count} pages  {count / elapsed:.0f}/s", flush=True)
        if limit is not None and count >= limit:
            break
    con.execute("COMMIT")
    print("building title index", flush=True)
    con.execute("CREATE INDEX idx_title_key ON meta(title_key)")
    titles: dict[str, int] = {}
    for rowid, key in con.execute("SELECT rowid, title_key FROM meta"):
        if key and key not in titles:
            titles[key] = int(rowid)
    with TITLE_PATH.open("wb") as handle:
        pickle.dump(titles, handle, protocol=4)
    con.commit()
    con.close()
    if DB_PATH.exists():
        DB_PATH.unlink()
    tmp.replace(DB_PATH)
    elapsed = time.perf_counter() - started
    print(
        f"done pages={count} titles={len(titles)} with_links={with_links} seconds={elapsed:.0f}",
        flush=True,
    )
    if count > 1000 and with_links / count < 0.05:
        raise SystemExit("hyperlinks were parsed from fewer than 5% of pages; link format is wrong")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--parts", type=Path, default=None)
    args = parser.parse_args()
    build(args.limit or None, args.parts)


if __name__ == "__main__":
    main()
