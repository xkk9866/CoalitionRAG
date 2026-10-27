"""Open-corpus mention resolution over the HotpotQA Wikipedia abstracts.

The first hop searches the whole abstract corpus. A later page is fetched by
title only when the pages already kept contain a resolvable mention of it.
A mention is either a Wikipedia hyperlink or a maximal token-boundary title.
Unlinked single-token titles are not mentions. Nested shorter titles are not
mentions. The fetched page must add a question term that the locked pages do
not already contain.

These constraints are fixed before test scoring. Calibration chooses only
whether a second resolution round is kept.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import sqlite3
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

from run_experiment import content_tokens

ROOT = Path(__file__).resolve().parents[1]
WIKI_DIR = ROOT / "data" / "wiki"
# The sqlite index is built on a local disk. The corpus archive stays with the project.
LOCAL_DIR = Path(r"local/hotpot_wiki")
DB_PATH = LOCAL_DIR / "hotpot_abstracts.sqlite"
TITLE_PATH = LOCAL_DIR / "title_keys.pkl"

# Evidence unit is the introductory paragraph, clipped so every method sees
# the same bounded page. Chosen before any open-corpus score.
PAGE_WORDS = 160
FIRST_HOP_K = 100
BUDGET = 4
ANCHOR_K = 2
RRF_K = 60
MAX_TITLE_TOKENS = 10
SECOND_ROUND_MIN_GAIN = 0.01

TOKEN_RE = re.compile(r"[0-9A-Za-z\u00C0-\u024F]+")
LINK_RE = re.compile(r'<a\s+href="([^"]+)"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")


def title_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def title_key(text: str) -> str:
    return " ".join(title_tokens(text))


def href_to_key(href: str) -> str:
    href = urllib.parse.unquote(href or "")
    href = href.split("#", 1)[0]
    lower = href.lower()
    if "wiki/" in lower:
        href = href[lower.rfind("wiki/") + 5 :]
    href = href.replace("_", " ").strip()
    return title_key(href)


def is_disambiguation(key: str) -> bool:
    return key.endswith(" disambiguation")


def flatten_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(flatten_text(part) for part in value)
    return str(value)


def strip_tags(text: str) -> str:
    text = TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_links(linked: str) -> tuple[str, list[dict]]:
    """Return plaintext and hyperlinks whose anchors sit in that plaintext."""
    parts: list[str] = []
    links: list[dict] = []
    last = 0
    for match in LINK_RE.finditer(linked):
        parts.append(strip_tags(linked[last : match.start()]))
        anchor = strip_tags(match.group(2))
        parts.append(anchor)
        links.append({"a": anchor, "t": href_to_key(match.group(1))})
        last = match.end()
    parts.append(strip_tags(linked[last:]))
    plain = re.sub(r"\s+", " ", " ".join(part for part in parts if part)).strip()
    return plain, links


def clip_words(text: str, n: int = PAGE_WORDS) -> str:
    parts = text.split()
    if len(parts) <= n:
        return text.strip()
    return " ".join(parts[:n]).strip()


def find_span(tokens: list[str], needle: list[str]) -> tuple[int, int] | None:
    if not needle or len(needle) > len(tokens):
        return None
    width = len(needle)
    for i in range(len(tokens) - width + 1):
        if tokens[i : i + width] == needle:
            return i, i + width
    return None


def fts_query(question: str) -> str:
    toks = []
    for tok in content_tokens(question):
        if tok.isalnum():
            toks.append(tok)
        if len(toks) >= 32:
            break
    if not toks:
        toks = ["history"]
    return " OR ".join(dict.fromkeys(toks))


def _fts_search_worker(task: tuple) -> list[list[int]]:
    """Read-only FTS5 search. The query strings are already built by the parent."""
    db_path, queries, k = task
    con = sqlite3.connect(db_path, timeout=120)
    try:
        con.execute("PRAGMA query_only=ON")
        con.execute("PRAGMA cache_size=-200000")
        con.execute("PRAGMA temp_store=MEMORY")
        out: list[list[int]] = []
        for query in queries:
            try:
                rows = con.execute(
                    "SELECT rowid FROM pages WHERE pages MATCH ? ORDER BY rank LIMIT ?",
                    (query, k),
                ).fetchall()
                out.append([int(row[0]) for row in rows])
            except sqlite3.OperationalError:
                out.append([])
        return out
    finally:
        con.close()


@dataclass
class Page:
    rowid: int
    title: str
    text: str
    key: str
    links: list[dict] = field(default_factory=list)


@dataclass
class Mention:
    rowid: int
    start: int
    end: int
    kind: str  # "link", "title", or "raw"
    phrase: str
    key: str


def link_mentions(tokens: list[str], links: list[dict], titles: dict[str, int]) -> tuple[list[Mention], list[bool]]:
    occupied = [False] * len(tokens)
    found: list[tuple[int, Mention]] = []
    ordered = sorted(links, key=lambda item: len(title_tokens(item.get("a", ""))), reverse=True)
    for link in ordered:
        key = link.get("t") or ""
        if not key or is_disambiguation(key):
            continue
        rowid = titles.get(key)
        if rowid is None:
            continue
        anchor = title_tokens(link.get("a", ""))
        if not anchor:
            anchor = title_tokens(key)
        span = find_span(tokens, anchor)
        if span is None:
            continue
        start, end = span
        if any(occupied[start:end]):
            continue
        for i in range(start, end):
            occupied[i] = True
        found.append((start, Mention(rowid, start, end, "link", " ".join(anchor), key)))
    found.sort(key=lambda item: item[0])
    return [item[1] for item in found], occupied


def maximal_title_matches(
    tokens: list[str],
    titles: dict[str, int],
    occupied: list[bool] | None = None,
) -> list[Mention]:
    n = len(tokens)
    if occupied is None:
        occupied = [False] * n
    else:
        occupied = list(occupied)
    found: list[Mention] = []
    upper = min(MAX_TITLE_TOKENS, n)
    for length in range(upper, 1, -1):
        for i in range(0, n - length + 1):
            if any(occupied[i : i + length]):
                continue
            phrase = " ".join(tokens[i : i + length])
            rowid = titles.get(phrase)
            if rowid is None or is_disambiguation(phrase):
                continue
            for j in range(i, i + length):
                occupied[j] = True
            found.append(Mention(rowid, i, i + length, "title", phrase, phrase))
    found.sort(key=lambda item: item.start)
    return found


def raw_title_matches(tokens: list[str], titles: dict[str, int]) -> list[Mention]:
    """Token-boundary title hits, including nested titles and long unigrams.

    This is the closed-pool substring rule lifted onto the open title list.
    """
    n = len(tokens)
    found: list[Mention] = []
    seen: set[int] = set()
    upper = min(MAX_TITLE_TOKENS, n)
    for length in range(upper, 0, -1):
        for i in range(0, n - length + 1):
            phrase = " ".join(tokens[i : i + length])
            if len(phrase) <= 3:
                continue
            rowid = titles.get(phrase)
            if rowid is None or rowid in seen or is_disambiguation(phrase):
                continue
            seen.add(rowid)
            found.append(Mention(rowid, i, i + length, "raw", phrase, phrase))
            if len(found) >= 40:
                return found
    return found


def nested_collision(tokens: list[str], titles: dict[str, int], admitted_phrase: str) -> bool:
    """True when the admitted title is a proper span of a longer title in the same text."""
    if not admitted_phrase:
        return False
    width = len(admitted_phrase.split())
    n = len(tokens)
    for length in range(width + 1, min(MAX_TITLE_TOKENS, n) + 1):
        for i in range(0, n - length + 1):
            phrase = " ".join(tokens[i : i + length])
            if admitted_phrase in phrase and phrase in titles and phrase != admitted_phrase:
                return True
    return False


def page_mentions(page: Page, titles: dict[str, int], mode: str) -> list[Mention]:
    tokens = title_tokens(page.title + " " + page.text)
    if mode == "raw":
        mentions = raw_title_matches(tokens, titles)
    else:
        linked, occupied = link_mentions(tokens, page.links, titles)
        lexical = maximal_title_matches(tokens, titles, occupied)
        if mode == "links":
            mentions = linked
        elif mode == "lexicon":
            mentions = lexical
        else:
            mentions = linked + lexical
    own = titles.get(page.key)
    return [m for m in mentions if m.rowid != own and m.rowid != page.rowid]


def new_term_count(question: str, kept: list[Page], cand: Page) -> int:
    question_terms = set(content_tokens(question))
    covered: set[str] = set()
    for page in kept:
        covered.update(content_tokens(page.title + " " + page.text))
    gained = set(content_tokens(cand.title + " " + cand.text)) & (question_terms - covered)
    return len(gained)


def render_pages(question: str, pages: list[Page]) -> str:
    """Same page presentation for every retriever. No method name and no span."""
    lines = [
        "Answer the question using the pages.",
        "Reply with only the short answer.",
        "",
    ]
    for i, page in enumerate(pages, start=1):
        lines.append(f"[{i}] {page.title}")
        lines.append(page.text)
        lines.append("")
    lines.append(f"Question: {question}")
    lines.append("Answer:")
    return "\n".join(lines)


class WikiDB:
    def __init__(self, db_path: Path = DB_PATH, title_path: Path = TITLE_PATH):
        if not db_path.exists():
            raise FileNotFoundError(f"missing wikipedia index: {db_path}")
        self.con = sqlite3.connect(db_path)
        self.con.execute("PRAGMA query_only=ON")
        self.con.row_factory = sqlite3.Row
        if title_path.exists():
            with title_path.open("rb") as f:
                self.titles: dict[str, int] = pickle.load(f)
        else:
            self.titles = {
                row["title_key"]: int(row["rowid"])
                for row in self.con.execute("SELECT rowid, title_key FROM meta")
            }

    def close(self) -> None:
        self.con.close()

    def search(self, question: str, k: int = FIRST_HOP_K) -> list[int]:
        query = fts_query(question)
        try:
            rows = self.con.execute(
                "SELECT rowid FROM pages WHERE pages MATCH ? ORDER BY rank LIMIT ?",
                (query, k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [int(row["rowid"]) for row in rows]

    def search_many(self, questions: list[str], k: int = FIRST_HOP_K, workers: int = 8) -> list[list[int]]:
        """Same FTS5 top-k as search(), issued from several read-only connections."""
        queries = [fts_query(question) for question in questions]
        if len(queries) < 24 or workers <= 1:
            return [self.search(question, k) for question in questions]
        import concurrent.futures

        workers = max(1, min(workers, len(queries)))
        chunks: list[list[str]] = [[] for _ in range(workers)]
        positions: list[list[int]] = [[] for _ in range(workers)]
        for i, query in enumerate(queries):
            chunks[i % workers].append(query)
            positions[i % workers].append(i)
        tasks = [(str(DB_PATH), chunk, k) for chunk in chunks]
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_fts_search_worker, tasks))
        merged: list[list[int]] = [[] for _ in queries]
        for part, idxs in zip(parts, positions):
            for rowids, idx in zip(part, idxs):
                merged[idx] = rowids
        return merged

    def bm25_scores(self, question: str, rowids: list[int]) -> dict[int, float]:
        if not rowids:
            return {}
        query = fts_query(question)
        marks = ",".join("?" for _ in rowids)
        try:
            rows = self.con.execute(
                f"SELECT rowid, bm25(pages) AS s FROM pages WHERE pages MATCH ? AND rowid IN ({marks})",
                (query, *rowids),
            ).fetchall()
        except sqlite3.OperationalError:
            return {}
        # FTS5 bm25 is negative when a row is more relevant.
        return {int(row["rowid"]): float(row["s"]) for row in rows}

    def fetch(self, rowids: list[int]) -> dict[int, Page]:
        if not rowids:
            return {}
        uniq = list(dict.fromkeys(int(i) for i in rowids))
        out: dict[int, Page] = {}
        for start in range(0, len(uniq), 400):
            chunk = uniq[start : start + 400]
            marks = ",".join("?" for _ in chunk)
            rows = self.con.execute(
                f"""
                SELECT p.rowid AS rowid, p.title AS title, p.body AS body, m.links AS links, m.title_key AS title_key
                FROM pages p JOIN meta m ON p.rowid = m.rowid
                WHERE p.rowid IN ({marks})
                """,
                chunk,
            ).fetchall()
            out.update(self._pages_from_rows(rows))
        return out

    def _pages_from_rows(self, rows) -> dict[int, Page]:
        out: dict[int, Page] = {}
        for row in rows:
            try:
                links = json.loads(row["links"] or "[]")
            except json.JSONDecodeError:
                links = []
            out[int(row["rowid"])] = Page(
                rowid=int(row["rowid"]),
                title=row["title"],
                text=clip_words(row["body"] or "", PAGE_WORDS),
                key=row["title_key"],
                links=links,
            )
        return out

    def gold_keys_present(self, gold_titles: list[str]) -> tuple[int, int]:
        keys = [title_key(title) for title in gold_titles if title_key(title)]
        hit = sum(1 for key in keys if key in self.titles)
        return hit, len(keys)


def rrf_fuse(left: list[int], right: list[int], k: int = RRF_K) -> list[int]:
    scores: dict[int, float] = {}
    for rank, rowid in enumerate(left):
        scores[rowid] = scores.get(rowid, 0.0) + 1.0 / (k + rank + 1)
    for rank, rowid in enumerate(right):
        scores[rowid] = scores.get(rowid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda rowid: scores[rowid], reverse=True)


def _take(order: list[int], have: list[int], n: int) -> list[int]:
    out = list(have)
    seen = set(out)
    for rowid in order:
        if rowid in seen:
            continue
        out.append(rowid)
        seen.add(rowid)
        if len(out) >= n:
            break
    return out


def order_by_dense(rowids: list[int], q_vec, vectors: dict[int, object]) -> list[int]:
    scored = []
    for rowid in rowids:
        vec = vectors.get(rowid)
        if vec is None:
            continue
        scored.append((float(q_vec @ vec), rowid))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [rowid for _, rowid in scored]


def _collect_mentions(kind: str, sources: list[Page], titles: dict[str, int], blocked: set[int], pages: dict[int, Page]):
    mode = "raw" if kind == "rawtitle" else "links" if kind == "links" else "lexicon" if kind == "lexicon" else "resolve"
    grouped: dict[int, Mention] = {}
    for page in sources:
        for mention in page_mentions(page, titles, mode):
            if mention.rowid in blocked or mention.rowid not in pages:
                continue
            prev = grouped.get(mention.rowid)
            if prev is None or (mention.kind == "link" and prev.kind != "link"):
                grouped[mention.rowid] = mention
    return list(grouped.values())


def _filter_terms(
    kind: str,
    question: str,
    kept: list[Page],
    cands: list[Mention],
    pages: dict[int, Page],
    require_term: bool = True,
) -> list[Mention]:
    # require_term is the frozen switch. When calibration drops it, every
    # mention ablation uses the same candidate pool as the shipped rule.
    if kind in {"rawtitle", "noterm"} or not require_term:
        return cands
    kept_ok = []
    for mention in cands:
        if new_term_count(question, kept, pages[mention.rowid]) > 0:
            kept_ok.append(mention)
    return kept_ok


def select_from_orders(
    kind: str,
    question: str,
    bm25_ids: list[int],
    dense_ids: list[int],
    pages: dict[int, Page],
    titles: dict[str, int],
    vectors: dict[int, object],
    q_vec,
    rounds: int = 1,
    rng_seed: int = 0,
    bm25_hint: dict[int, float] | None = None,
    require_term: bool = True,
) -> dict:
    """Return up to four rowids. Mention targets must already be in `pages`."""
    if kind == "bm25":
        return _pack(_take(bm25_ids, [], BUDGET), pages, {})
    if kind == "dense":
        return _pack(_take(dense_ids, [], BUDGET), pages, {})
    if kind == "rrf":
        return _pack(_take(rrf_fuse(bm25_ids, dense_ids), [], BUDGET), pages, {})
    if kind == "d2b2":
        return _pack(_take(dense_ids[:ANCHOR_K] + bm25_ids, [], BUDGET), pages, {})
    if kind == "rerank":
        return _pack(_take(dense_ids, [], BUDGET), pages, {})

    anchor = _take(dense_ids, [], ANCHOR_K)
    rank_mode = "dense"
    if kind == "namedrand":
        rank_mode = "random"
    elif kind == "namedbm25":
        rank_mode = "bm25"
    chosen = list(anchor)
    origin: dict[int, str] = {}
    collisions = 0
    token_blob = []
    for rowid in anchor:
        if rowid in pages:
            token_blob.extend(title_tokens(pages[rowid].title + " " + pages[rowid].text))
    for round_i in range(max(1, rounds)):
        if round_i == 0:
            sources = [pages[i] for i in anchor if i in pages]
        else:
            sources = [pages[i] for i in chosen if i in origin and i in pages]
        kept_pages = [pages[i] for i in chosen if i in pages]
        cands = _filter_terms(
            kind,
            question,
            kept_pages,
            _collect_mentions(kind, sources, titles, set(chosen), pages),
            pages,
            require_term=require_term,
        )
        cands = _rank_mentions(
            cands, question, kept_pages, pages, vectors, q_vec, rank_mode, rng_seed, bm25_hint or {}
        )
        if round_i == 0:
            added = 0
            for mention in cands:
                if len(chosen) >= BUDGET or added >= ANCHOR_K:
                    break
                chosen.append(mention.rowid)
                origin[mention.rowid] = mention.kind
                added += 1
                if kind == "rawtitle" and nested_collision(token_blob, titles, mention.phrase):
                    collisions += 1
            if added == 0:
                break
        else:
            if not cands:
                break
            mention = cands[0]
            if len(chosen) < BUDGET:
                chosen.append(mention.rowid)
                origin[mention.rowid] = mention.kind
            else:
                last = chosen[-1]
                prefix = kept_pages[:-1]
                gain = new_term_count(question, prefix, pages[mention.rowid])
                last_gain = new_term_count(question, prefix, pages[last]) if last in pages else 0
                if gain > last_gain and mention.rowid not in chosen:
                    origin.pop(last, None)
                    chosen[-1] = mention.rowid
                    origin[mention.rowid] = mention.kind
    if len(chosen) < BUDGET:
        chosen = _take(dense_ids, chosen, BUDGET)
    packed = _pack(chosen, pages, origin)
    packed["collisions"] = collisions
    return packed


def fill_expand(anchor: list[int], second_ids: list[int], pages: dict[int, Page]) -> dict:
    anchor = list(anchor)
    chosen = _take(second_ids, anchor, BUDGET)
    origin = {rowid: "expand" for rowid in chosen if rowid not in anchor}
    return _pack(chosen, pages, origin)


def _rank_mentions(cands, question, kept, pages, vectors, q_vec, rank_mode, rng_seed, bm25_hint):
    if rank_mode == "random":
        import random

        rng = random.Random(rng_seed)
        order = list(cands)
        rng.shuffle(order)
        return order
    scored = []
    for mention in cands:
        page = pages[mention.rowid]
        gain = new_term_count(question, kept, page)
        vec = vectors.get(mention.rowid)
        dense = float(q_vec @ vec) if vec is not None else -1.0
        if rank_mode == "bm25":
            # More negative FTS5 bm25 means a closer lexical match.
            bm25 = float(bm25_hint.get(mention.rowid, 0.0))
            scored.append((bm25, -gain, -dense, mention))
        else:
            scored.append((gain, dense, mention.end - mention.start, mention))
    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=rank_mode != "bm25")
    return [item[-1] for item in scored]


def _pack(chosen: list[int], pages: dict[int, Page], origin: dict[int, str]) -> dict:
    ids = [rowid for rowid in chosen if rowid in pages][:BUDGET]
    return {
        "rowids": ids,
        "titles": [pages[rowid].title for rowid in ids],
        "keys": [pages[rowid].key for rowid in ids],
        "origin": {str(rowid): origin[rowid] for rowid in ids if rowid in origin},
        "collisions": 0,
    }


def mention_rowids(kind: str, question: str, anchor_ids: list[int], pages: dict[int, Page], titles: dict[str, int]) -> list[int]:
    """Rowids the mention methods may need encoded before selection."""
    del question
    if kind not in {"resolve", "links", "lexicon", "rawtitle", "namedrand", "namedbm25", "noterm"}:
        return []
    mode = "raw" if kind == "rawtitle" else "links" if kind == "links" else "lexicon" if kind == "lexicon" else "resolve"
    out = []
    seen = set(anchor_ids)
    for rowid in anchor_ids:
        page = pages.get(rowid)
        if page is None:
            continue
        for mention in page_mentions(page, titles, mode):
            if mention.rowid in seen:
                continue
            seen.add(mention.rowid)
            out.append(mention.rowid)
    return out


def stable_seed(qid: str) -> int:
    return int(hashlib.sha1(qid.encode("utf-8")).hexdigest()[:8], 16)


def self_check() -> None:
    titles = {
        "independent spirit awards": 1,
        "18th independent spirit awards": 2,
        "georges franju": 3,
        "washington": 4,
        "washington state": 5,
        "city": 6,
        "fougeres": 7,
    }
    tokens = title_tokens("The 18th Independent Spirit Awards were held in Washington.")
    maximal = maximal_title_matches(tokens, titles)
    phrases = [item.phrase for item in maximal]
    assert "18th independent spirit awards" in phrases, phrases
    assert "independent spirit awards" not in phrases, phrases
    assert "washington" not in phrases, phrases
    raw = raw_title_matches(tokens, titles)
    raw_phrases = [item.phrase for item in raw]
    assert "independent spirit awards" in raw_phrases, raw_phrases
    assert "washington" in raw_phrases, raw_phrases
    assert nested_collision(tokens, titles, "independent spirit awards")
    assert not nested_collision(tokens, titles, "18th independent spirit awards")
    page = Page(
        rowid=9,
        title="Spotlight on a Murderer",
        text="It was directed by Georges Franju in France.",
        key="spotlight on a murderer",
        links=[{"a": "Georges Franju", "t": "georges franju"}],
    )
    mentions = page_mentions(page, titles, "resolve")
    assert any(item.rowid == 3 and item.kind == "link" for item in mentions), mentions
    short = Page(
        rowid=8,
        title="Awards",
        text="Independent Spirit Awards honored the film.",
        key="awards",
        links=[],
    )
    lex = page_mentions(short, titles, "lexicon")
    assert any(item.phrase == "independent spirit awards" for item in lex), lex
    print("self_check ok")


if __name__ == "__main__":
    self_check()
