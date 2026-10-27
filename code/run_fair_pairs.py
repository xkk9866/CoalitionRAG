"""Paired bootstrap for a finished fair retrieval file."""

from __future__ import annotations

import json
import sys

import numpy as np

import run_fair


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


def main() -> None:
    path = sys.argv[1]
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    arms = {name: np.asarray([row[name] for row in rows], dtype=np.float64) for name in run_fair.ARMS}
    report = {
        "n": len(rows),
        "mean": {name: float(values.mean()) for name, values in arms.items()},
        "versus_flat": {name: boot(arms[name], arms["flat"]) for name in run_fair.ARMS if name != "flat"},
        "versus_lexlink": {"ace": boot(arms["ace"], arms["lexlink"])},
    }
    types = sorted({row["type"] for row in rows})
    by_type = {}
    for name in types:
        subset = [row for row in rows if row["type"] == name]
        local = {arm: np.asarray([row[arm] for row in subset], dtype=np.float64) for arm in ("flat", "ace", "lexlink", "qtitle")}
        by_type[name] = {
            "n": len(subset),
            "mean": {arm: float(values.mean()) for arm, values in local.items()},
            "ace_versus_flat": boot(local["ace"], local["flat"]),
            "lex_versus_flat": boot(local["lexlink"], local["flat"]),
        }
    report["by_type"] = by_type
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
