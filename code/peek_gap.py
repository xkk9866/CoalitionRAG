import json
from pathlib import Path

root = Path(__file__).resolve().parent / "results"
preds = [json.loads(l) for l in (root / "full_predictions.jsonl").open(encoding="utf-8")]
jobs = [json.loads(l) for l in (root / "full_jobs.jsonl").open(encoding="utf-8")]
job = {(j["dataset"], j["method"], j["id"]): j for j in jobs if j["split"] == "main"}
main = [r for r in preds if r["split"] == "main"]
by = {(r["dataset"], r["method"], r["id"]): r for r in main}

n = 0
for r in main:
    if r["dataset"] != "2wiki" or r["method"] != "coalition":
        continue
    s = by[(r["dataset"], "single", r["id"])]
    if r["f1"] < 0.2 and s["f1"] > 0.8 and r["agents"] == ["bridge"]:
        c = job[("2wiki", "coalition", r["id"])]
        d = job[("2wiki", "single", r["id"])]
        q = c["prompt"].split("Question:")[-1].split("Answer:")[0].strip().replace("\n", " ")
        ct = [ln for ln in c["prompt"].splitlines() if ln.startswith("[")]
        dt = [ln for ln in d["prompt"].splitlines() if ln.startswith("[")]
        print("Q", q)
        print(" gold", r["gold"], "coal", r["pred"], "dense", s["pred"])
        print(" COAL", ct, "sup", round(c["support"], 2))
        print(" DENSE", dt, "sup", round(d["support"], 2))
        print("---")
        n += 1
        if n >= 2:
            break
print("shown", n)
