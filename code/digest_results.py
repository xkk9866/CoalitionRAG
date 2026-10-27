import json
from pathlib import Path

p = Path(__file__).resolve().parent / "results" / "full_summary.json"
d = json.loads(p.read_text(encoding="utf-8"))
s = d["summary"]


def line(name, r):
    print(
        f"{name:12} EM {r['em']:.3f} F1 {r['f1']:.3f} sup {r['support']:.3f} "
        f"tok {r['tokens']:.1f} util {r['utility']:.3f} size {r['size']:.2f} "
        f"rev {r['revise_rate']:.3f} f1se {r['f1_se']:.3f} tokse {r['tokens_se']:.1f} "
        f"lat {r['latency']:.3f} n {r['n']}"
    )


for ds in ["hotpot", "2wiki", "musique"]:
    print("==", ds, "MAIN")
    for m in ["norag", "single", "router", "fixed", "debate", "coalition"]:
        line(m, s[ds]["main"][m])
    print("-- ablation")
    for m in ["no_rev", "no_cost", "no_profile", "no_credit"]:
        line(m, s[ds]["ablation"][m])
    print("selection", {k: round(v, 3) for k, v in s[ds]["main"]["coalition"]["selection"].items()})

print("== ROBUST")
for n in [0, 1, 2, 4]:
    print("n", n)
    for m in ["fixed", "coalition", "router"]:
        line("  " + m, s["hotpot"][f"robust{n}"][m])
