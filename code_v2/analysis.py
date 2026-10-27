"""Analyses: (1) counterfactual dividends by pair type, (2) MuSiQue breakdown by hop count, (3) retrieval latency."""
import time

import numpy as np
import torch

from common import DATASETS, RES, jdump, jload, load_bench, recall_at


def dividends():
    out = {}
    for ds in DATASETS:
        _, qs = load_bench(ds)
        pools = jload(RES / "pools" / f"exp20_s5_k8_{ds}.json")
        cache = torch.load(RES / "cache" / f"f_cdn_{ds}.pt")
        gg, gn, ng, nn_ = [], [], [], []
        for q, pool, c in zip(qs, pools, cache):
            pa, pc = torch.sigmoid(c["a"]), torch.sigmoid(c["C"])
            G = set(q["gold"])
            for i, pi in enumerate(pool):
                for j, pj in enumerate(pool):
                    if i == j:
                        continue
                    d = float(pc[i, j] - pa[j])           # gain of j when i is known
                    (gg if pi in G and pj in G else gn if pi in G else ng if pj in G else nn_).append(d)
        out[ds] = {k: [float(np.mean(v)), float(np.mean(np.array(v) > 0.1)), len(v)]
                   for k, v in {"gold->gold": gg, "gold->other": gn, "other->gold": ng, "other->other": nn_}.items()}
        print(ds, {k: [round(x, 3) for x in v[:2]] for k, v in out[ds].items()}, flush=True)
    jdump(out, RES / "tables" / "dividends.json")


def hops():
    docs, qs = load_bench("musique")
    res = {}
    for m in ["first:bge", "o_point_gold", "ircot", "setr", "f_cdn-cover"]:
        rows = (jload(RES / "first_stage" / "bge_musique.json")["ranked"] if m.startswith("first")
                else jload(RES / "retrieval" / f"{m}_musique.json"))
        qa = jload(RES / "qa" / f"qwen25-7b-awq_{m.replace(':', '-')}_musique.json")["em_f1"]
        for h in (2, 3, 4):
            idx = [i for i, q in enumerate(qs) if q["hops"] == h]
            r5 = [recall_at((rows[i] if m.startswith("first") else (rows[i].get("top5") or rows[i]["rank"][:5]))[:5], qs[i]["gold"]) for i in idx]
            res.setdefault(m, {})[h] = {"n": len(idx), "R@5": float(np.mean(r5)), "F1": float(np.mean([qa[i][1] for i in idx]))}
        print(m, {h: (v["n"], round(100 * v["R@5"], 1), round(100 * v["F1"], 1)) for h, v in res[m].items()}, flush=True)
    jdump(res, RES / "tables" / "hops.json")


def latency(n=50):
    from transformers import AutoModelForSequenceClassification
    from cdn import ENC, Pairer, game
    from select_games import _combos
    model = AutoModelForSequenceClassification.from_pretrained(ENC).cuda().eval()
    model.load_state_dict(torch.load(RES / "ckpt" / "p_cdn.pt"))
    pr = Pairer()
    docs, qs = load_bench("musique")
    pools = jload(RES / "pools" / "exp20_s5_k8_musique.json")
    torch.cuda.synchronize(); t0 = time.time(); ts = []
    for q, pool in zip(qs[:n], pools[:n]):
        t1 = time.time()
        a, B, C = game(model, pr, q["question"], [docs[i] for i in pool])
        _combos(len(pool), 5, a.device)
        torch.cuda.synchronize(); ts.append(time.time() - t1)
    print("CDN latency per query: mean %.2fs, pool size %.1f" % (np.mean(ts), np.mean([len(p) for p in pools[:n]])))
    jdump({"cdn_sec_per_query": float(np.mean(ts))}, RES / "tables" / "latency.json")


if __name__ == "__main__":
    dividends(); hops(); latency()
