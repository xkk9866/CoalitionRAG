"""Precise half-up numbers for the manuscript. No retuning."""

from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "results"
summary = json.loads((ROOT / "cover_summary.json").read_text(encoding="utf-8"))
preds = [json.loads(line) for line in (ROOT / "cover_predictions.jsonl").open(encoding="utf-8")]
jobs = [json.loads(line) for line in (ROOT / "cover_jobs.jsonl").open(encoding="utf-8")]


def h(x: float, places: str = "0.001") -> str:
    return str(Decimal(repr(x)).quantize(Decimal(places), rounding=ROUND_HALF_UP))


print("ABLATION")
for ds in ["hotpot", "2wiki", "musique"]:
    for method, split in [("cover", "main"), ("no_chain", "ablation"), ("no_term", "ablation"), ("no_cite", "ablation")]:
        b = summary["summary"][ds][split][method]
        print(ds, method, "f1", h(b["f1"]), "sup", h(b["support"]), "tok", h(b["tokens"], "1"), "size", h(b["size"]))

print("ROBUST")
for n in [0, 1, 2, 4]:
    for method in ["fixed", "cover", "router"]:
        b = summary["summary"]["hotpot"][f"robust{n}"][method]
        print(n, method, "f1", h(b["f1"]), "tok", h(b["tokens"], "1"), "util", h(b["utility"]), "sup", h(b["support"]))

print("SUBSET")
main = [r for r in preds if r["split"] == "main"]
for ds in ["hotpot", "2wiki", "musique"]:
    ids = {r["id"] for r in main if r["dataset"] == ds and r["method"] == "cover" and r["agents"] != ["dense"]}
    for method in ["cover", "single"]:
        vals = [r for r in main if r["dataset"] == ds and r["method"] == method and r["id"] in ids]
        f1 = sum(r["f1"] for r in vals) / len(vals)
        sup = sum(r["support"] for r in vals) / len(vals)
        print(ds, method, "n", len(vals), "f1", h(f1), "sup", h(sup))

wanted = {
    "57e690c7087f11ebbd6bac1f6bf848b6",
    "6d7c51640bdb11eba7f7acde48001122",
}
for j in jobs:
    if j["id"] in wanted and j["method"] == "cover" and j["split"] == "main":
        q = [line for line in j["prompt"].splitlines() if line.startswith("Question:")]
        print("Q", j["id"], q[0][:300] if q else "none")
        print("answers", j["answers"][:3], "agents", j["agents"])
