"""Find one real swap where cover support exceeds dense support."""

from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "results"
summary = json.loads((ROOT / "cover_summary.json").read_text(encoding="utf-8"))
jobs = [json.loads(line) for line in (ROOT / "cover_jobs.jsonl").open(encoding="utf-8")]
preds = [json.loads(line) for line in (ROOT / "cover_predictions.jsonl").open(encoding="utf-8")]


def half_up(x: float, places: str) -> str:
    return str(Decimal(repr(x)).quantize(Decimal(places), rounding=ROUND_HALF_UP))


print("RAW")
for ds in ["hotpot", "2wiki", "musique"]:
    for method in ["norag", "single", "router", "fixed", "debate", "residual", "cover"]:
        b = summary["summary"][ds]["main"][method]
        print(
            ds,
            method,
            "em",
            half_up(b["em"], "0.001"),
            "f1",
            half_up(b["f1"], "0.001"),
            "sup",
            half_up(b["support"], "0.001"),
            "tok",
            half_up(b["tokens"], "1"),
            "util",
            half_up(b["utility"], "0.001"),
            "size",
            half_up(b["size"], "0.001"),
            "f1se",
            half_up(b["f1_se"], "0.001"),
        )

pred = {(r["dataset"], r["split"], r["method"], r["id"]): r for r in preds}
main_jobs = [j for j in jobs if j["split"] == "main" and j["method"] in {"cover", "single", "residual"}]
by = {}
for j in main_jobs:
    by.setdefault((j["dataset"], j["id"]), {})[j["method"]] = j

print("\nCASES")
shown = 0
for key, pack in by.items():
    if "cover" not in pack or "single" not in pack:
        continue
    c, s = pack["cover"], pack["single"]
    if c["agents"] == ["dense"]:
        continue
    if c["support"] <= s["support"]:
        continue
    pc = pred[(c["dataset"], "main", "cover", c["id"])]
    ps = pred[(s["dataset"], "main", "single", s["id"])]
    if pc["f1"] <= ps["f1"]:
        continue
    if c["dataset"] != "2wiki":
        continue
    print("---", c["dataset"], c["id"])
    print("gold", c["answers"][:2])
    print("cover agents", c["agents"], "sup", round(c["support"], 3), "pred", pc["pred"], "f1", round(pc["f1"], 3))
    print("dense pred", ps["pred"], "f1", round(ps["f1"], 3), "sup", round(s["support"], 3))
    # Prompt titles are the lines starting with [
    titles = [line for line in c["prompt"].splitlines() if line.startswith("[")]
    print("cover pages:")
    for line in titles:
        print(" ", line[:180])
    stitles = [line for line in s["prompt"].splitlines() if line.startswith("[")]
    print("dense pages:")
    for line in stitles:
        print(" ", line[:180])
    shown += 1
    if shown >= 3:
        break
