"""Print cover-experiment means. Does not rescore or retune."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np

from run_experiment import load_examples

ROOT = Path(__file__).resolve().parent / "results"
summary = json.loads((ROOT / "cover_summary.json").read_text(encoding="utf-8"))
rows = [json.loads(line) for line in (ROOT / "cover_predictions.jsonl").open(encoding="utf-8")]


def r3(x: float) -> str:
    return str(Decimal(str(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def ri(x: float) -> str:
    return str(Decimal(str(x)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def show(block: dict, keys: list[str]) -> str:
    parts = []
    for key in keys:
        val = block[key]
        if key in {"tokens", "n"}:
            parts.append(f"{key}={ri(val) if key != 'n' else val}")
        else:
            parts.append(f"{key}={r3(val)}")
    return " ".join(parts)


KEYS = ["n", "em", "f1", "support", "tokens", "utility", "size", "f1_se"]

print("=== MAIN ===")
for ds in ["hotpot", "2wiki", "musique"]:
    print(ds)
    for method in ["norag", "single", "router", "fixed", "debate", "residual", "cover"]:
        block = summary["summary"][ds]["main"][method]
        extra = ""
        if "selection" in block:
            sel = {k: round(v, 3) for k, v in block["selection"].items() if v > 0}
            extra = " sel=" + str(sel)
        print(" ", method, show(block, KEYS), extra)

print("=== ABLATION ===")
for ds in ["hotpot", "2wiki", "musique"]:
    print(ds)
    for method in ["cover", "no_chain", "no_term", "no_cite"]:
        block = summary["summary"][ds]["ablation"][method] if method != "cover" else summary["summary"][ds]["main"]["cover"]
        sel = block.get("selection", {})
        opened = {k: round(v, 3) for k, v in sel.items() if v > 0 and k != "dense"}
        print(" ", method, show(block, KEYS), "partners", opened)

print("=== ROBUST ===")
for n in [0, 1, 2, 4]:
    print("n_weak", n)
    for method in ["fixed", "cover", "router"]:
        block = summary["summary"]["hotpot"][f"robust{n}"][method]
        sel = block.get("selection", {})
        weak = {k: round(v, 3) for k, v in sel.items() if str(k).startswith("weak") and v > 0}
        print(" ", method, show(block, ["f1", "tokens", "utility", "support", "size"]), "weak", weak or "none")

print("=== TYPES ===")
id_type = {}
for name in ["hotpot", "2wiki", "musique"]:
    _cal, test = load_examples(name, 200, 300, seed=7)
    key = "qtype" if "qtype" in test[0] else "type"
    for ex in test:
        id_type[(name, ex["id"])] = ex[key]

main = [r for r in rows if r["split"] == "main"]
for ds in ["hotpot", "2wiki", "musique"]:
    types = sorted({id_type[(ds, r["id"])] for r in main if r["dataset"] == ds})
    print("==", ds, types)
    for method in ["single", "cover", "residual", "fixed", "debate"]:
        sub = [r for r in main if r["dataset"] == ds and r["method"] == method]
        print(" ", method)
        for t in types:
            vals = [r["f1"] for r in sub if id_type.get((ds, r["id"])) == t]
            sup = [r["support"] for r in sub if id_type.get((ds, r["id"])) == t]
            print(f"    {t:22} n={len(vals):3d} F1={np.mean(vals):.3f} sup={np.mean(sup):.3f}")

print("=== SWAP EFFECT ===")
for ds in ["hotpot", "2wiki", "musique"]:
    by_id = defaultdict(dict)
    for r in main:
        if r["dataset"] == ds and r["method"] in {"single", "cover"}:
            by_id[r["id"]][r["method"]] = r
    changed = same = 0
    f1_c = f1_s = sup_c = sup_s = 0.0
    n_ch = 0
    for pair in by_id.values():
        if "cover" not in pair or "single" not in pair:
            continue
        # Agent set differs from dense-only when a partner is present.
        swapped = pair["cover"]["agents"] != ["dense"]
        if swapped:
            changed += 1
            n_ch += 1
            f1_c += pair["cover"]["f1"]
            f1_s += pair["single"]["f1"]
            sup_c += pair["cover"]["support"]
            sup_s += pair["single"]["support"]
        else:
            same += 1
    print(
        ds,
        "swapped",
        changed,
        "same",
        same,
        "on_swap F1",
        round(f1_c / max(n_ch, 1), 3),
        round(f1_s / max(n_ch, 1), 3),
        "sup",
        round(sup_c / max(n_ch, 1), 3),
        round(sup_s / max(n_ch, 1), 3),
    )

print("=== PARTNER GIVEN OPEN ===")
for ds in ["hotpot", "2wiki", "musique"]:
    sub = [r for r in main if r["dataset"] == ds and r["method"] == "cover" and r["agents"] != ["dense"]]
    cnt = Counter()
    for r in sub:
        for name in r["agents"]:
            if name != "dense":
                cnt[name] += 1
    n = len(sub)
    print(ds, "n_open", n, {k: round(v / n, 3) for k, v in cnt.most_common()})
