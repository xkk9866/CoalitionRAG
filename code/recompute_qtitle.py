"""Exact title lookup with longer spans packed first.

This does not rescore the cross-encoder arms. It only rebuilds the
question-side title baseline: maximal corpus titles in the question,
longest span first, then the BM25 order fills any open slots.
"""

from __future__ import annotations

import json

import numpy as np

import open_retrieve as ore
import run_action as act
import run_fair
import run_fair_wiki
import run_open as ro


def longest_ids(question: str, titles: dict[str, int]) -> list[int]:
    found = ore.maximal_title_matches(ore.title_tokens(question), titles)
    found.sort(key=lambda mention: (-(mention.end - mention.start), mention.start))
    out = []
    seen = set()
    for mention in found:
        if mention.rowid in seen:
            continue
        seen.add(mention.rowid)
        out.append(mention.rowid)
    return out


def score_rows(rows: list[dict], db: ore.WikiDB, dest) -> list[dict]:
    done = {}
    if dest.exists():
        with dest.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    done[row["id"]] = row
    pending = [row for row in rows if row["id"] not in done]
    print(f"qtitle pending {len(pending)} done {len(done)} -> {dest.name}", flush=True)
    saved = list(done.values())
    with dest.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), 40):
            batch = pending[start : start + 40]
            print(f"qtitle {start}:{start + len(batch)} / {len(pending)}", flush=True)
            lists = db.search_many([row["question"] for row in batch], ore.FIRST_HOP_K, workers=8)
            wanted = [rowid for ids in lists for rowid in ids]
            for row in batch:
                wanted.extend(longest_ids(row["question"], db.titles))
            pages = db.fetch(list(dict.fromkeys(wanted)))
            for row, ids in zip(batch, lists):
                gold = set(row["gold_keys"])
                picked = longest_ids(row["question"], db.titles)
                packed = act.pack_ids(picked, [rowid for rowid in ids if rowid not in set(picked)], pages)
                keys = {pages[rowid].key for rowid in packed if rowid in pages}
                pool = {pages[rowid].key for rowid in ids if rowid in pages}
                outside = [key for key in gold if key not in pool]
                record = {
                    "id": row["id"],
                    "type": row["type"],
                    "qtitle_long": act.support(keys, gold),
                    "qtitle_long_hit": sum(1 for key in outside if key in keys),
                    "qtitle_long_ids": packed,
                    "outside": len(outside),
                }
                saved.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
    return saved


def boot(left: np.ndarray, right: np.ndarray) -> dict:
    diff = left - right
    se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(diff), size=(10000, len(diff)))
    means = diff[draws].mean(axis=1)
    return {
        "delta": float(diff.mean()),
        "se": se,
        "boot_low": float(np.quantile(means, 0.025)),
        "boot_high": float(np.quantile(means, 0.975)),
    }


def attach(path, records: list[dict]) -> dict:
    base = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                base[row["id"]] = row
    joined = []
    for row in records:
        src = base[row["id"]]
        row["ace"] = src["ace"]
        row["flat"] = src["flat"]
        row["qtitle"] = src["qtitle"]
        joined.append(row)
    ace = np.asarray([row["ace"] for row in joined], dtype=np.float64)
    flat = np.asarray([row["flat"] for row in joined], dtype=np.float64)
    old = np.asarray([row["qtitle"] for row in joined], dtype=np.float64)
    new = np.asarray([row["qtitle_long"] for row in joined], dtype=np.float64)
    return {
        "n": len(joined),
        "qtitle_long": float(new.mean()),
        "qtitle_appearance": float(old.mean()),
        "versus_flat": boot(new, flat),
        "versus_ace": boot(ace, new),
        "both": int(np.sum(new == 1.0)),
        "rows": joined,
    }


def main() -> None:
    db = ore.WikiDB()
    try:
        hotpot = run_fair.hotpot_rows("test")
        hotpot_path = ro.RESULT_DIR / "fair_hotpot_qtitle_long.jsonl"
        hotpot_rows = score_rows(hotpot, db, hotpot_path)
        hotpot_report = attach(ro.RESULT_DIR / "fair_retrieve_predictions.jsonl", hotpot_rows)
        hotpot_rows_out = hotpot_report.pop("rows")
        (ro.RESULT_DIR / "fair_hotpot_qtitle_long_summary.json").write_text(
            json.dumps(hotpot_report, indent=2),
            encoding="utf-8",
        )
        print("hotpot", json.dumps(hotpot_report), flush=True)
        wiki_rows = score_rows(
            run_fair_wiki.covered_rows("2wiki", db.titles),
            db,
            ro.RESULT_DIR / "fair_2wiki_qtitle_long_partial.jsonl",
        )
    finally:
        db.close()
    wiki_report = attach(ro.RESULT_DIR / "fair_2wiki_predictions.jsonl", wiki_rows)
    wiki_rows_out = wiki_report.pop("rows")
    (ro.RESULT_DIR / "fair_2wiki_qtitle_long.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in wiki_rows_out),
        encoding="utf-8",
    )
    (ro.RESULT_DIR / "fair_2wiki_qtitle_long_summary.json").write_text(
        json.dumps(wiki_report, indent=2),
        encoding="utf-8",
    )
    print("2wiki", json.dumps(wiki_report), flush=True)


if __name__ == "__main__":
    main()
