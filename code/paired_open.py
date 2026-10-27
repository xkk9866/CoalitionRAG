"""Paired support contrasts on the held-out open retrieval split."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

path = Path(__file__).resolve().parent / "results" / "open_retrieve_predictions.jsonl"
by: dict[str, dict[str, float]] = defaultdict(dict)
with path.open(encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        by[row["id"]][row["method"]] = float(row["support"])


def paired(left: str, right: str) -> None:
    diff = np.asarray([by[qid][left] - by[qid][right] for qid in by], dtype=np.float64)
    se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    changed = diff != 0
    print(
        f"{left} - {right}: delta {diff.mean():.4f} se {se:.4f} "
        f"changed {int(changed.sum())} n {len(diff)}"
    )


def both(method: str) -> None:
    vals = np.asarray([by[qid][method] for qid in by], dtype=np.float64)
    print(f"{method} both-gold {float((vals == 1).mean()):.4f} any-miss {float((vals < 1).mean()):.4f}")


for name in ["dense", "links", "resolve", "namedbm25", "rawtitle", "expand"]:
    both(name)
paired("links", "resolve")
paired("links", "namedbm25")
paired("resolve", "rawtitle")
paired("links", "rawtitle")
paired("namedbm25", "rawtitle")
