"""Paired support, bootstrap intervals, and gold pages the question does not name."""

from __future__ import annotations

import json

import numpy as np

import open_retrieve as ore
import run_fair
import run_open as ro


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


def title_named(question: str, title: str) -> bool:
    q = ore.title_tokens(question)
    t = ore.title_tokens(title)
    if not t or len(t) > len(q):
        return False
    if len(t) == 1:
        return t[0] in q
    for i in range(len(q) - len(t) + 1):
        if q[i : i + len(t)] == t:
            return True
    return False


def main() -> None:
    rows = ro.load_hotpot()
    by_id = {row["id"]: row for row in rows}
    saved = []
    with (ro.RESULT_DIR / "fair_retrieve_predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                saved.append(json.loads(line))
    arms = {name: np.asarray([row[name] for row in saved], dtype=np.float64) for name in run_fair.ARMS}
    report = {
        "n": len(saved),
        "mean": {name: float(values.mean()) for name, values in arms.items()},
        "versus_flat": {name: boot(arms[name], arms["flat"]) for name in run_fair.ARMS if name != "flat"},
        "versus_qtitle": {name: boot(arms[name], arms["qtitle"]) for name in ("ace", "flat", "lexlink", "surface", "title")},
        "versus_lexlink": {"ace": boot(arms["ace"], arms["lexlink"])},
        "versus_surface": {"ace": boot(arms["ace"], arms["surface"])},
        "versus_title": {"ace": boot(arms["ace"], arms["title"])},
        "mean_ce_abstracts": float(np.mean([row["ce_abstracts"] for row in saved])),
        "mean_ce_surface": float(np.mean([row["ce_surface"] for row in saved])),
        "mean_ce_titles": float(np.mean([row["ce_titles"] for row in saved])),
        "both": {name: int(np.sum(values == 1.0)) for name, values in arms.items()},
    }
    by_type: dict[str, dict[str, list[float]]] = {}
    for row in saved:
        bucket = by_type.setdefault(row["type"], {name: [] for name in run_fair.ARMS})
        for name in run_fair.ARMS:
            bucket[name].append(row[name])
    report["by_type_n"] = {name: len(next(iter(bucket.values()))) for name, bucket in by_type.items()}
    report["by_type"] = {
        name: {arm: float(np.mean(values)) for arm, values in bucket.items()}
        for name, bucket in by_type.items()
    }
    focus = ("ace", "flat", "qtitle", "lexlink", "surface", "title")
    db = ore.WikiDB()
    try:
        wanted = [rowid for row in saved for name in focus for rowid in row[name + "_ids"]]
        pages = db.fetch(list(dict.fromkeys(wanted)))
    finally:
        db.close()
    unnamed_scores = {name: [] for name in focus}
    examples = []
    n_unnamed_questions = 0
    n_unnamed_pages = 0
    for row in saved:
        src = by_id[row["id"]]
        unnamed = []
        for title in src["gold_titles"]:
            key = ore.title_key(title)
            if key and not title_named(src["question"], title):
                unnamed.append(key)
        if not unnamed:
            continue
        n_unnamed_questions += 1
        n_unnamed_pages += len(unnamed)
        gold = set(unnamed)
        got = {}
        for name in focus:
            keys = {pages[rowid].key for rowid in row[name + "_ids"] if rowid in pages}
            score = len(keys & gold) / len(gold)
            unnamed_scores[name].append(score)
            got[name] = score
        if got["ace"] == 1.0 and got["qtitle"] < 1.0 and got["flat"] < 1.0 and len(examples) < 12:
            examples.append(
                {
                    "id": row["id"],
                    "type": src["type"],
                    "question": src["question"],
                    "unnamed_keys": unnamed,
                    "ace": got["ace"],
                    "flat": got["flat"],
                    "qtitle": got["qtitle"],
                    "lexlink": got["lexlink"],
                }
            )
    report["unnamed"] = {
        "questions": n_unnamed_questions,
        "pages": n_unnamed_pages,
        "mean": {name: float(np.mean(values)) if values else None for name, values in unnamed_scores.items()},
    }
    if unnamed_scores["ace"]:
        report["unnamed"]["ace_versus_flat"] = boot(
            np.asarray(unnamed_scores["ace"]), np.asarray(unnamed_scores["flat"])
        )
        report["unnamed"]["ace_versus_qtitle"] = boot(
            np.asarray(unnamed_scores["ace"]), np.asarray(unnamed_scores["qtitle"])
        )
    report["examples"] = examples
    (ro.RESULT_DIR / "fair_stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: report[key] for key in ("n", "mean", "versus_flat", "versus_lexlink", "unnamed")},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
