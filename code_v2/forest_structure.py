"""Does the forest recover the reasoning structure it was never shown?

Gold reason of every gold passage, from the datasets' own annotations:
  MuSiQue : the decomposition step that a sub-question refers to (#k); steps without a reference hang from q.
  2Wiki   : the passage whose evidence triple has this passage's title as its object; otherwise q.
The forest's posterior P(parent_j = x | E = G) is computed exactly (matrix-tree edge marginals) on test questions
whose whole gold set is in the pool. Also reports the calibration of the marginals m_j.
usage: forest_structure.py <forest cache tag> [<cdn cache tag>]
"""
import re
import sys

import numpy as np
import torch

from common import DATA, RES, jdump, jload, load_bench
from forest import marginals, reason_posteriors

Q = -1


def musique_parents(raw, index):
    out = []
    for s in raw:
        steps = s["question_decomposition"]
        ids = []
        for st in steps:
            p = s["paragraphs"][st["paragraph_support_idx"]]
            ids.append(index.get(p["title"] + "\n" + p["paragraph_text"]))
        par = {}
        for k, st in enumerate(steps):
            refs = [int(x) - 1 for x in re.findall(r"#(\d)", st["question"])]
            par[ids[k]] = {ids[r] for r in refs if 0 <= r < len(ids)} or {Q}
        out.append(par)
    return out


def wiki_parents(raw, docs, qs):
    title_of = {}
    out = []
    for s, q in zip(raw, qs):
        tit = {docs[g].partition("\n")[0]: g for g in q["gold"]}
        par = {g: set() for g in q["gold"]}
        for subj, _, obj in s.get("evidences", []):
            if obj in tit and subj in tit and tit[obj] != tit[subj]:
                par[tit[obj]].add(tit[subj])
        out.append({g: (p or {Q}) for g, p in par.items()})
    return out


def structure(ds, tag, cdn_tag):
    docs, qs = load_bench(ds)
    raw = jload(DATA / f"{ds}.json")
    index = {d: i for i, d in enumerate(docs)}
    parents = musique_parents(raw, index) if ds == "musique" else wiki_parents(raw, docs, qs)
    pools = jload(RES / "pools" / f"exp20_s5_k8_{ds}.json")
    cache = torch.load(RES / "cache" / f"{tag}_{ds}.pt")
    cdn = torch.load(RES / "cache" / f"{cdn_tag}_{ds}.pt") if cdn_tag else None
    rng = np.random.default_rng(0)
    hit = {"forest": [], "cdn_gain": [], "unary_chain": [], "random": [], "always_q": []}
    hit_bridge = {k: [] for k in hit}
    for qi, (q, pool, c) in enumerate(zip(qs[: len(cache)], pools, cache)):
        par = parents[qi]
        if len(q["gold"]) < 2 or None in par or not all(g in pool for g in q["gold"]):
            continue
        pos = {d: i for i, d in enumerate(pool)}
        G = [pos[g] for g in q["gold"]]
        gq, gS = reason_posteriors(c["sq"], c["S"], nodes=G)
        sq = c["sq"].double()
        chain = sorted(G, key=lambda j: -float(sq[j]))
        for g in q["gold"]:
            j = pos[g]
            truth = {Q if t == Q else pos[t] for t in par[g]}
            opts = [Q] + [i for i in G if i != j]
            post = [float(gq[j])] + [float(gS[i, j]) for i in G if i != j]
            pred = {"forest": opts[int(np.argmax(post))],
                    "unary_chain": Q if chain.index(j) == 0 else chain[chain.index(j) - 1],
                    "random": opts[rng.integers(len(opts))], "always_q": Q}
            if cdn is not None:
                a, C = cdn[qi]["a"].double(), cdn[qi]["C"].double()
                gain = [0.0] + [float(torch.sigmoid(C[i, j]) - torch.sigmoid(a[j])) for i in G if i != j]
                pred["cdn_gain"] = opts[int(np.argmax(gain))]
            for k, p in pred.items():
                hit[k].append(float(p in truth))
                if Q not in truth:
                    hit_bridge[k].append(float(p in truth))
    res = {k: {"all": float(np.mean(v)), "bridge": float(np.mean(hit_bridge[k])), "n": len(v), "n_bridge": len(hit_bridge[k])}
           for k, v in hit.items() if v}
    print(ds, {k: (round(100 * v["all"], 1), round(100 * v["bridge"], 1), v["n"], v["n_bridge"]) for k, v in res.items()}, flush=True)
    return res


def calibration(ds, tag, bins=10):
    _, qs = load_bench(ds)
    pools = jload(RES / "pools" / f"exp20_s5_k8_{ds}.json")
    cache = torch.load(RES / "cache" / f"{tag}_{ds}.pt")
    out = {}
    for name in ("forest", "unary"):
        P, Y = [], []
        for q, pool, c in zip(qs, pools, cache):
            m = marginals(c["sq"], c["S"]) if name == "forest" else torch.sigmoid(c["sq"].double())
            P += m.tolist(); Y += [float(d in set(q["gold"])) for d in pool]
        P, Y = np.array(P), np.array(Y)
        edges = np.linspace(0, 1, bins + 1)
        ece, rel = 0.0, []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (P >= lo) & (P < hi if hi < 1 else P <= hi)
            if s.any():
                ece += s.mean() * abs(P[s].mean() - Y[s].mean())
                rel.append([float(P[s].mean()), float(Y[s].mean()), int(s.sum())])
        out[name] = {"ece": float(ece), "reliability": rel, "expected_gold": float(P.sum() / len(cache)),
                     "true_gold_in_pool": float(Y.sum() / len(cache))}
    print(ds, "ECE forest %.3f unary %.3f | expected evidence per pool %.2f vs gold in pool %.2f" % (
        out["forest"]["ece"], out["unary"]["ece"], out["forest"]["expected_gold"], out["forest"]["true_gold_in_pool"]), flush=True)
    return out


def reason_mass(ds, tag):
    """Unconditional posterior P(reason of p_j = x) on test pools, by kind of child and kind of reason.

    Children: gold passages whose annotated reason is the question ('gold, first hop'), gold passages whose
    annotated reason is another passage ('gold, later hop'), and non-gold passages.
    """
    docs, qs = load_bench(ds)
    raw = jload(DATA / f"{ds}.json")
    index = {d: i for i, d in enumerate(docs)}
    parents = musique_parents(raw, index) if ds == "musique" else wiki_parents(raw, docs, qs)
    pools = jload(RES / "pools" / f"exp20_s5_k8_{ds}.json")
    cache = torch.load(RES / "cache" / f"{tag}_{ds}.pt")
    acc = {k: [] for k in ("gold, first hop", "gold, later hop", "non-gold")}
    for qi, (q, pool, c) in enumerate(zip(qs[: len(cache)], pools, cache)):
        gq, gS = reason_posteriors(c["sq"], c["S"])
        gold = {d for d in q["gold"]}
        isg = torch.tensor([d in gold for d in pool])
        for j, d in enumerate(pool):
            if d in gold:
                kind = "gold, first hop" if parents[qi].get(d) == {Q} else "gold, later hop"
                if d not in parents[qi] or None in parents[qi]:
                    continue
            else:
                kind = "non-gold"
            col = gS[:, j]
            pg = float(col[isg].sum() - (col[j] if isg[j] else 0))
            po = float(col[~isg].sum() - (col[j] if not isg[j] else 0))
            acc[kind].append([float(gq[j]), pg, po, 1 - float(gq[j]) - pg - po])
    out = {k: {"q": float(np.mean([v[0] for v in a])), "gold": float(np.mean([v[1] for v in a])),
               "other": float(np.mean([v[2] for v in a])), "null": float(np.mean([v[3] for v in a])), "n": len(a)}
           for k, a in acc.items() if a}
    print(ds, "reason mass", {k: [round(v[x], 3) for x in ("q", "gold", "other", "null")] + [v["n"]] for k, v in out.items()}, flush=True)
    return out


if __name__ == "__main__":
    tag = sys.argv[1]
    cdn_tag = sys.argv[2] if len(sys.argv) > 2 else ""
    res = {"structure": {ds: structure(ds, tag, cdn_tag) for ds in ("musique", "2wikimultihopqa")},
           "reason_mass": {ds: reason_mass(ds, tag) for ds in ("musique", "2wikimultihopqa")},
           "calibration": {ds: calibration(ds, tag) for ds in ("musique", "2wikimultihopqa", "hotpotqa")}}
    jdump(res, RES / "tables" / f"{tag}_structure.json")
