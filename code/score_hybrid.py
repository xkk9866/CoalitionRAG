"""One-shot test of the named-entity chain.

A Wikipedia title is resolved from the question when a majority of its
surface tokens are capitalized. Up to two such pages, longest first, stay
in the dossier. The remaining slots keep the greedy hyperlink dossier.
The rule is not refit on these questions.
"""

from __future__ import annotations

import json
import re

import numpy as np

import open_retrieve as ore
import run_action as act
import run_fair
import run_fair_wiki
import run_open as ro

WORD = re.compile(r"[0-9A-Za-z\u00C0-\u024F]+(?:'[0-9A-Za-z\u00C0-\u024F]+)?")


def cap_anchors(question: str, titles: dict[str, int]) -> list[int]:
    tokens = [(m.group(0), m.group(0)[:1].isupper()) for m in WORD.finditer(question)]
    mentions = ore.maximal_title_matches([tok.lower() for tok, _cap in tokens], titles)
    kept = []
    for mention in mentions:
        caps = [tokens[i][1] for i in range(mention.start, mention.end)]
        if caps and sum(caps) * 2 > len(caps):
            kept.append(mention)
    kept.sort(key=lambda item: (-(item.end - item.start), item.start))
    out, seen = [], set()
    for mention in kept:
        if mention.rowid in seen:
            continue
        seen.add(mention.rowid)
        out.append(mention.rowid)
        if len(out) >= 2:
            break
    return out


def paired(left: list[float], right: list[float]) -> dict:
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(diff), size=(10000, len(diff)))
    means = diff[draws].mean(axis=1)
    return {
        "n": int(len(diff)),
        "delta": float(diff.mean()),
        "se": se,
        "boot_low": float(np.quantile(means, 0.025)),
        "boot_high": float(np.quantile(means, 0.975)),
    }


def score(name: str, rows: list[dict], predictions: dict[str, dict], db: ore.WikiDB) -> dict:
    hybrid_scores = []
    ace_scores = []
    flat_scores = []
    lex_scores = []
    types: list[str] = []
    changed = 0
    for row in rows:
        record = predictions.get(row["id"])
        if record is None:
            continue
        gold = set(row["gold_keys"])
        anchors = cap_anchors(row["question"], db.titles)
        order = anchors + [rowid for rowid in record["ace_ids"] if rowid not in set(anchors)]
        pages = db.fetch(order[: ore.BUDGET])
        keys = {pages[rowid].key for rowid in order[: ore.BUDGET] if rowid in pages}
        hybrid_scores.append(act.support(keys, gold))
        ace_scores.append(float(record["ace"]))
        flat_scores.append(float(record["flat"]))
        lex_scores.append(float(record["lexlink"]))
        types.append(row["type"])
        if order[: ore.BUDGET] != list(record["ace_ids"])[: ore.BUDGET]:
            changed += 1
        if len(hybrid_scores) % 1000 == 0:
            print(f"{name} {len(hybrid_scores)}", flush=True)
    report = {
        "dataset": name,
        "n": len(hybrid_scores),
        "ace": float(np.mean(ace_scores)),
        "hybrid": float(np.mean(hybrid_scores)),
        "complete": float(np.mean([score == 1.0 for score in hybrid_scores])),
        "changed": changed,
        "versus_ace": paired(hybrid_scores, ace_scores),
        "versus_flat": paired(hybrid_scores, flat_scores),
        "versus_lexlink": paired(hybrid_scores, lex_scores),
        "by_type": {},
    }
    for kind in sorted(set(types)):
        idx = [i for i, value in enumerate(types) if value == kind]
        report["by_type"][kind] = {
            "n": len(idx),
            "ace": float(np.mean([ace_scores[i] for i in idx])),
            "hybrid": float(np.mean([hybrid_scores[i] for i in idx])),
            "versus_ace": paired([hybrid_scores[i] for i in idx], [ace_scores[i] for i in idx]),
        }
    return report


def load_predictions(path) -> dict[str, dict]:
    out = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                out[row["id"]] = row
    return out


def main() -> None:
    db = ore.WikiDB()
    try:
        hotpot = run_fair.hotpot_rows("test")
        hotpot_pred = load_predictions(ro.RESULT_DIR / "fair_retrieve_predictions.jsonl")
        hotpot_report = score("hotpot", hotpot, hotpot_pred, db)
        (ro.RESULT_DIR / "hybrid_hotpot_summary.json").write_text(json.dumps(hotpot_report, indent=2), encoding="utf-8")
        print("hotpot", json.dumps(hotpot_report), flush=True)
        wiki_rows = run_fair_wiki.covered_rows("2wiki", db.titles)
        wiki_pred = load_predictions(ro.RESULT_DIR / "fair_2wiki_predictions.jsonl")
        wiki_report = score("2wiki", wiki_rows, wiki_pred, db)
        (ro.RESULT_DIR / "hybrid_2wiki_summary.json").write_text(json.dumps(wiki_report, indent=2), encoding="utf-8")
        print("2wiki", json.dumps(wiki_report), flush=True)
        musique_rows = run_fair_wiki.covered_rows("musique", db.titles)
        musique_pred = load_predictions(ro.RESULT_DIR / "fair_musique_predictions.jsonl")
        musique_report = score("musique", musique_rows, musique_pred, db)
        (ro.RESULT_DIR / "hybrid_musique_summary.json").write_text(json.dumps(musique_report, indent=2), encoding="utf-8")
        print("musique", json.dumps(musique_report), flush=True)
    finally:
        db.close()


if __name__ == "__main__":
    main()
