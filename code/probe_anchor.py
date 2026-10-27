"""Calibration-only probes. Test questions are loaded and then discarded."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from run_experiment import (
    COVER_SLOTS,
    build_offers,
    cached_agent_scores,
    content_tokens,
    load_examples,
    support_recall,
    top_indices,
    attach_embeddings,
)


def term_hits(question: str, pages: list[dict]) -> set[str]:
    blob = " ".join((p["title"] + " " + p["text"]).lower() for p in pages)
    return {t for t in content_tokens(question) if t in blob}


def cite_blob(pages: list[dict]) -> str:
    return " ".join((p["title"] + " " + p["text"]).lower() for p in pages)


def agents_of(pages: list[dict]) -> list[str]:
    out = []
    for page in pages:
        if page["agent"] not in out:
            out.append(page["agent"])
    return out


def dense_pack(offers: list[dict]) -> list[dict]:
    selected = []
    held = set()
    for offer in offers:
        if offer["agent"] != "dense" or offer["title"] in held:
            continue
        selected.append(offer)
        held.add(offer["title"])
        if len(selected) == COVER_SLOTS:
            break
    return selected


def swap_slot4(ex: dict, offers: list[dict]) -> list[dict]:
    """Keep the dense top-3. Replace the fourth page only with a cited page that adds a query term."""
    selected = dense_pack(offers)
    if len(selected) < COVER_SLOTS:
        return selected
    scores = cached_agent_scores(ex, "dense")
    core, marginal = selected[:3], selected[3]
    base = term_hits(ex["question"], selected)
    blob = cite_blob(core)
    best = marginal
    best_gain = 0
    held = {p["title"] for p in selected}
    for cand in offers:
        if cand["agent"] in {"dense", "redundant"} or cand["title"] in held:
            continue
        if float(scores[cand["idx"]]) < 0:
            continue
        key = cand["title"].lower().strip()
        if len(key) <= 3 or key not in blob:
            continue
        trial_terms = term_hits(ex["question"], core + [cand])
        if not base <= trial_terms:
            continue
        gain = len(trial_terms) - len(base)
        if gain > best_gain:
            best = cand
            best_gain = gain
    if best is marginal:
        return selected
    return core + [best]


def swap_if_value(ex: dict, offers: list[dict], beta: float = 0.2) -> list[dict]:
    """Keep dense pages that uniquely cover a query term. Swap a redundant one for a cited page if value rises."""
    selected = dense_pack(offers)
    if len(selected) < 2:
        return selected
    scores = cached_agent_scores(ex, "dense")
    terms = content_tokens(ex["question"])

    def value(pages: list[dict]) -> float:
        if not pages:
            return 0.0
        tau = (len(term_hits(ex["question"], pages)) / len(terms)) if terms else 0.0
        blob = cite_blob(pages)
        closed = 0
        for i, page in enumerate(pages):
            key = page["title"].lower().strip()
            if len(key) <= 3:
                continue
            for j, other in enumerate(pages):
                if i != j and key in (other["title"] + " " + other["text"]).lower():
                    closed += 1
                    break
        return tau + beta * (closed / len(pages))

    base_v = value(selected)
    held = {p["title"] for p in selected}
    # Index of a dense page with no unique query term; else the last slot.
    drop_at = len(selected) - 1
    for i, page in enumerate(selected):
        others = selected[:i] + selected[i + 1 :]
        mine = term_hits(ex["question"], [page]) - term_hits(ex["question"], others)
        if not mine:
            drop_at = i
            break
    core = selected[:drop_at] + selected[drop_at + 1 :]
    blob = cite_blob(core)
    best_pages = selected
    best_v = base_v
    for cand in offers:
        if cand["agent"] in {"dense", "redundant"} or cand["title"] in held:
            continue
        if float(scores[cand["idx"]]) < 0:
            continue
        key = cand["title"].lower().strip()
        if len(key) <= 3 or key not in blob:
            continue
        trial = core + [cand]
        trial_v = value(trial)
        if trial_v > best_v + 1e-6:
            best_v = trial_v
            best_pages = trial
    return best_pages


def mandatory_pair(ex: dict, offers: list[dict], beta: float, lam: float) -> list[dict]:
    """Dense is mandatory. Choose the best 4-page subset of dense top-4 plus one partner's top-2."""
    dense_pages = [o for o in offers if o["agent"] == "dense"]
    by_agent: dict[str, list[dict]] = {}
    for offer in offers:
        by_agent.setdefault(offer["agent"], []).append(offer)
    terms = content_tokens(ex["question"])

    def value(pages: list[dict]) -> float:
        if not pages or not terms:
            tau = 0.0
        else:
            tau = len(term_hits(ex["question"], pages)) / len(terms)
        return tau

    def pack(cands: list[dict]) -> list[dict]:
        # Greedy by marginal term gain, then stop at 4. Dense offers are listed first.
        chosen = []
        held = set()
        pool = list(cands)
        while len(chosen) < COVER_SLOTS and pool:
            base = term_hits(ex["question"], chosen)
            best_i = None
            best_gain = -1
            for i, cand in enumerate(pool):
                if cand["title"] in held:
                    continue
                gain = len(term_hits(ex["question"], chosen + [cand]) - base)
                # slight preference for closing a citation
                key = cand["title"].lower().strip()
                bonus = 0.25 if chosen and len(key) > 3 and key in cite_blob(chosen) else 0.0
                score = gain + bonus
                if score > best_gain:
                    best_gain = score
                    best_i = i
            if best_i is None or (chosen and best_gain <= 0):
                break
            pick = pool.pop(best_i)
            chosen.append(pick)
            held.add(pick["title"])
        return chosen

    base_pages = pack(dense_pages)
    base_v = value(base_pages)
    best_pages = base_pages
    best_score = base_v
    for name, pages in by_agent.items():
        if name in {"dense", "redundant"}:
            continue
        trial = pack(dense_pages + pages[:2])
        score = value(trial) - lam
        if score > best_score:
            best_score = score
            best_pages = trial
    return best_pages


def report(name: str, examples: list[dict], fn) -> None:
    sup_c, sup_d, sizes = [], [], []
    bridge = 0
    copy = 0
    changed = 0
    for ex in examples:
        offers = build_offers(ex, ["dense", "bm25", "title", "bridge", "compare", "numeric", "lead", "redundant"])
        pages = fn(ex, offers)
        dense_idx = top_indices(cached_agent_scores(ex, "dense"), COVER_SLOTS)
        dense_titles = [ex["titles"][i] for i in dense_idx]
        got = [p["title"] for p in pages]
        sup_c.append(support_recall(got, ex["gold_titles"]))
        sup_d.append(support_recall(dense_titles, ex["gold_titles"]))
        ag = agents_of(pages)
        sizes.append(len(ag))
        bridge += int("bridge" in ag)
        copy += int("redundant" in ag)
        changed += int(got != dense_titles)
    n = len(examples)
    print(
        f"{name:18} sup {np.mean(sup_c):.3f} dense {np.mean(sup_d):.3f} "
        f"gain {np.mean(sup_c) - np.mean(sup_d):+.3f} size {np.mean(sizes):.2f} "
        f"bridge {bridge / n:.2f} copy {copy / n:.3f} changed {changed / n:.2f}",
        flush=True,
    )


def main() -> None:
    encoder = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda:0")
    prepared = {}
    for dataset in ["hotpot", "2wiki", "musique"]:
        cal, _test = load_examples(dataset, 200, 300, 7)
        attach_embeddings(cal, encoder)
        prepared[dataset] = cal
        print("ready", dataset, len(cal), flush=True)
    del encoder
    for dataset, cal in prepared.items():
        print("==", dataset, flush=True)
        report("slot4", cal, lambda ex, offers: swap_slot4(ex, offers))
        report("value", cal, lambda ex, offers: swap_if_value(ex, offers))
        report("pair", cal, lambda ex, offers: mandatory_pair(ex, offers, 0.2, 0.0))


if __name__ == "__main__":
    main()
