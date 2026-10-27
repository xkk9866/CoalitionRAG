import json
from pathlib import Path

jobs = [
    json.loads(line)
    for line in (Path(__file__).resolve().parent / "results" / "full_jobs.jsonl").open(encoding="utf-8")
    if '"split": "main"' in line or True
]
# filter in python
main = [j for j in jobs if j["split"] == "main"]
by = {}
for j in main:
    by[(j["dataset"], j["method"], j["id"])] = j

# Find Derrickson if present
for j in main:
    if "Derrickson" in j["prompt"] or "Derrickson" in j.get("prompt", ""):
        print("FOUND", j["dataset"], j["method"], j["id"], j["agents"])
        break
else:
    print("derrickson not in prompts via first scan")

# print a hotpot coalition that includes dense+bridge and a 2wiki bridge-only
shown = 0
for j in main:
    if j["dataset"] == "hotpot" and j["method"] == "coalition" and j["agents"] == ["dense", "bridge"]:
        q = j["prompt"].split("Question:")[-1][:180].replace("\n", " ")
        titles = [ln for ln in j["prompt"].splitlines() if ln.startswith("[")]
        print("HOTPOT", j["id"], titles)
        print(" Q", q)
        sib = by[("hotpot", "single", j["id"])]
        print(" SINGLE", [ln for ln in sib["prompt"].splitlines() if ln.startswith("[")])
        print(" support c", j["support"], "s", sib["support"])
        shown += 1
        if shown >= 1:
            break

shown = 0
for j in main:
    if j["dataset"] == "2wiki" and j["method"] == "coalition" and j["agents"] == ["bridge"]:
        q = j["prompt"].split("Question:")[-1][:200].replace("\n", " ")
        titles = [ln for ln in j["prompt"].splitlines() if ln.startswith("[")]
        sib = by[("2wiki", "single", j["id"])]
        print("WIKI", j["id"])
        print(" Q", q)
        print(" COAL", titles, "sup", round(j["support"], 3))
        print(" SINGLE", [ln for ln in sib["prompt"].splitlines() if ln.startswith("[")], "sup", round(sib["support"], 3))
        shown += 1
        if shown >= 1:
            break
