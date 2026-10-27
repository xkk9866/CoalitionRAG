"""Brute-force check of the forest algebra: enumerate all reason assignments of a small pool."""
import itertools

import torch

from forest import log_prob, marginals, reason_posteriors


def brute(sq, S):
    n = sq.shape[0]
    probs = {}
    edge = torch.zeros(n + 1, n, dtype=torch.float64)   # row n = q
    Z = 0.0
    for par in itertools.product(range(-2, n), repeat=n):  # -2 null, -1 q, i passage
        if any(p == j for j, p in enumerate(par)):
            continue
        ok, E = True, set()
        for j in range(n):            # follow reasons to a root; a cycle never reaches one
            seen, x = set(), j
            while x >= 0 and x not in seen:
                seen.add(x); x = par[x]
            if x >= 0:
                ok = False; break
            if x == -1:
                E.add(j)
        if not ok:
            continue
        w = 1.0
        for j, p in enumerate(par):
            w *= 1.0 if p == -2 else float(torch.exp(sq[j] if p == -1 else S[p, j]))
        Z += w
        key = tuple(sorted(E))
        probs[key] = probs.get(key, 0.0) + w
        for j, p in enumerate(par):
            if p == -1:
                edge[n, j] += w
            elif p >= 0:
                edge[p, j] += w
    return {k: v / Z for k, v in probs.items()}, edge / Z


torch.manual_seed(0)
for n in (1, 2, 3, 4):
    sq = torch.randn(n, dtype=torch.float64) * 2
    S = torch.randn(n, n, dtype=torch.float64) * 2
    P, edge = brute(sq, S)
    assert abs(sum(P.values()) - 1) < 1e-9
    for G, p in P.items():
        if G:
            lp = float(log_prob(sq, S, list(G)).exp())
            assert abs(lp - p) < 1e-9, (n, G, lp, p)
    m = marginals(sq, S)
    mb = torch.tensor([sum(p for G, p in P.items() if j in G) for j in range(n)], dtype=torch.float64)
    assert torch.allclose(m, mb, atol=1e-9), (m, mb)
    gq, gS = reason_posteriors(sq, S)
    assert torch.allclose(gq, edge[n], atol=1e-9) and torch.allclose(gS, edge[:n], atol=1e-9)
    print(n, "ok", m.numpy().round(3))
# no reason edges: the forest reduces to the pointwise cross-encoder
sq = torch.randn(5, dtype=torch.float64)
S = torch.full((5, 5), -1e9, dtype=torch.float64)
assert torch.allclose(marginals(sq, S), torch.sigmoid(sq))
print("pointwise reduction ok")
