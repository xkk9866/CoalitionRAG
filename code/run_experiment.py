"""CoalitionRAG experiments on HotpotQA, 2WikiMultihopQA, and MuSiQue.

Profiles are estimated on a calibration split. Test labels are used only for scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import string
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "code" / "results"
CACHE_PATH = RESULT_DIR / "gen_cache.jsonl"

SPECIALISTS = ["dense", "bm25", "title", "bridge", "compare", "numeric", "lead"]
MAIN_AGENTS = SPECIALISTS + ["redundant"]

# Frozen before test evaluation.
LAM_R = 0.75
LAM_C = 0.04
LAM_COMM = 0.015
K_MAX = 3
# Evidence-cover coalition. Numeric values are replaced by cover_hparams.json,
# which is chosen on the calibration split only.
COVER_ALPHA = 0.70
COVER_LAM = 0.03
COVER_COMM = 0.015
COVER_BETA = 0.20
COVER_SLOTS = 4
COVER_METHODS = {"cover", "no_chain", "no_term", "no_cite"}
REVISION_UNC = 0.40
CREDIT_BLEND = 0.55
UTILITY_TOKEN_SCALE = 4000.0
PASSAGE_WORDS = 80
SINGLE_K = 4
FIXED_CAP = 8
COALITION_CAP = 5

STOP = set(
    """a an the of to and in on for with by from as at or is are was were be been being
    it this that those these who whom whose which what when where why how many much
    same both than then into over after before about between among during without
    within not no yes do does did done having have has had can could should would
    will just than their his her its they them he she we you i""".split()
)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOP and len(t) > 2]


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(pred: str, gold: str) -> float:
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    common = sum((Counter(pred_toks) & Counter(gold_toks)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_toks)
    recall = common / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def best_em_f1(pred: str, answers: list[str]) -> tuple[float, float]:
    em = 0.0
    f1 = 0.0
    for gold in answers:
        em = max(em, float(normalize_answer(pred) == normalize_answer(gold)))
        f1 = max(f1, token_f1(pred, gold))
    return em, f1


def clip_words(text: str, n: int = PASSAGE_WORDS) -> str:
    parts = text.split()
    if len(parts) <= n:
        return text.strip()
    return " ".join(parts[:n]).strip()


def support_recall(titles: list[str], gold: set[str]) -> float:
    if not gold:
        return 0.0
    return len(set(titles) & gold) / len(gold)


def answer_hit(text: str, answers: list[str]) -> float:
    blob = normalize_answer(text)
    for gold in answers:
        g = normalize_answer(gold)
        if g and g in blob:
            return 1.0
    return 0.0


def stratified_indices(types: list[str], n_cal: int, n_test: int, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(types):
        buckets[t].append(i)
    total = len(types)
    cal: list[int] = []
    test: list[int] = []
    for t in sorted(buckets):
        idxs = buckets[t][:]
        rng.shuffle(idxs)
        nc = int(round(n_cal * len(idxs) / total))
        nt = int(round(n_test * len(idxs) / total))
        cal.extend(idxs[:nc])
        test.extend(idxs[nc : nc + nt])
    rng.shuffle(cal)
    rng.shuffle(test)
    return cal[:n_cal], test[:n_test]


def _join_hotpot_like(row: dict) -> tuple[list[str], list[str], set[str]]:
    ctx = row["context"]
    titles = list(ctx["title"])
    texts = [" ".join(sents).strip() for sents in ctx["sentences"]]
    gold = set(row["supporting_facts"]["title"])
    return titles, texts, gold


def load_examples(dataset: str, n_cal: int, n_test: int, seed: int) -> tuple[list[dict], list[dict]]:
    from datasets import load_dataset

    if dataset == "hotpot":
        raw = load_dataset("hotpot_qa", "distractor", split="validation")
        types = list(raw["type"])
        cal_i, test_i = stratified_indices(types, n_cal, n_test, seed)
        picked = cal_i + test_i

        def convert(row):
            titles, texts, gold = _join_hotpot_like(row)
            return {
                "id": row["id"],
                "question": row["question"],
                "answers": [row["answer"]],
                "type": row["type"],
                "titles": titles,
                "texts": texts,
                "gold_titles": gold,
            }

        rows = [convert(raw[i]) for i in picked]
    elif dataset == "2wiki":
        raw = load_dataset("framolfese/2WikiMultihopQA", split="validation")
        types = list(raw["type"])
        cal_i, test_i = stratified_indices(types, n_cal, n_test, seed)
        picked = cal_i + test_i

        def convert(row):
            titles, texts, gold = _join_hotpot_like(row)
            return {
                "id": str(row["id"]),
                "question": row["question"],
                "answers": [row["answer"]],
                "type": row["type"],
                "titles": titles,
                "texts": texts,
                "gold_titles": gold,
            }

        rows = [convert(raw[i]) for i in picked]
    elif dataset == "musique":
        raw = load_dataset("dgslibisey/MuSiQue", split="validation")
        types = []
        for row in raw:
            types.append(f"{len(row['question_decomposition'])}hop")
        cal_i, test_i = stratified_indices(types, n_cal, n_test, seed)
        picked = cal_i + test_i

        def convert(row, hop):
            titles = [p["title"] for p in row["paragraphs"]]
            texts = [p["paragraph_text"] for p in row["paragraphs"]]
            gold = {p["title"] for p in row["paragraphs"] if p["is_supporting"]}
            aliases = [row["answer"]] + list(row["answer_aliases"] or [])
            return {
                "id": row["id"],
                "question": row["question"],
                "answers": aliases,
                "type": hop,
                "titles": titles,
                "texts": texts,
                "gold_titles": gold,
            }

        rows = [convert(raw[i], types[i]) for i in picked]
    else:
        raise ValueError(dataset)

    cal = rows[: len(cal_i)]
    test = rows[len(cal_i) :]
    return cal, test


def attach_embeddings(examples: list[dict], encoder) -> None:
    queries = [ex["question"] for ex in examples]
    q_emb = np.asarray(encoder.encode(queries, normalize_embeddings=True, batch_size=64, show_progress_bar=False))
    flat = []
    spans = []
    for ex in examples:
        start = len(flat)
        flat.extend(ex["texts"])
        spans.append((start, len(ex["texts"])))
    p_emb = np.asarray(encoder.encode(flat, normalize_embeddings=True, batch_size=64, show_progress_bar=False))
    for ex, q, (start, n) in zip(examples, q_emb, spans):
        ex["q_emb"] = q.astype(np.float32)
        ex["P"] = p_emb[start : start + n].astype(np.float32)
        tokenized = [tokenize(f"{title} {text}") for title, text in zip(ex["titles"], ex["texts"])]
        ex["bm25_tokens"] = tokenized


def agent_scores(agent: str, ex: dict, weak_id: int = 0) -> np.ndarray:
    question = ex["question"]
    dense = ex["P"] @ ex["q_emb"]
    n = len(dense)
    if agent == "dense" or agent.startswith("redundant"):
        return dense.copy()
    if agent.startswith("weak"):
        rng = np.random.default_rng(1000 + weak_id)
        return -dense + 0.04 * rng.standard_normal(n)
    if agent == "bm25":
        from rank_bm25 import BM25Okapi

        bm25 = BM25Okapi(ex["bm25_tokens"])
        return np.asarray(bm25.get_scores(tokenize(question)), dtype=np.float32)
    if agent == "title":
        q_toks = set(content_tokens(question))
        scores = np.zeros(n, dtype=np.float32)
        for i, (title, text) in enumerate(zip(ex["titles"], ex["texts"])):
            first = text.split(".")[0]
            pool = set(tokenize(title)) | set(content_tokens(first))
            scores[i] = len(q_toks & pool) / (len(q_toks) + 1e-6)
        return scores
    if agent == "bridge":
        top = int(np.argmax(dense))
        scores = 0.25 * dense
        top_title = ex["titles"][top].lower()
        top_text = ex["texts"][top].lower()
        for i in range(n):
            if i == top:
                scores[i] = 0.05
                continue
            title_i = ex["titles"][i].lower()
            if title_i and title_i in top_text:
                scores[i] += 1.0
            if top_title and top_title in ex["texts"][i].lower():
                scores[i] += 0.45
        return scores
    if agent == "compare":
        lexical = _lexical(question, ex)
        cue = bool(re.search(r"\b(same|both|which|older|younger|more|less|or)\b", question, re.I))
        if cue:
            return lexical + 0.15
        return 0.25 * lexical
    if agent == "numeric":
        lexical = _lexical(question, ex)
        cue = bool(re.search(r"\b(when|how many|year|date|age|population|number)\b", question, re.I))
        bonus = np.zeros(n, dtype=np.float32)
        for i, text in enumerate(ex["texts"]):
            years = len(re.findall(r"\b(?:1[5-9]\d{2}|20\d{2})\b", text))
            nums = len(re.findall(r"\b\d+(?:\.\d+)?\b", text))
            bonus[i] = 0.12 * years + 0.03 * min(nums, 8)
        if cue:
            return lexical + bonus
        return 0.2 * lexical
    if agent == "lead":
        scores = np.zeros(n, dtype=np.float32)
        q_toks = set(content_tokens(question))
        for i, text in enumerate(ex["texts"]):
            first = set(content_tokens(" ".join(text.split()[:40])))
            scores[i] = len(q_toks & first) / (len(q_toks) + 1e-6)
        return scores
    raise KeyError(agent)


def _lexical(question: str, ex: dict) -> np.ndarray:
    q_toks = set(content_tokens(question))
    scores = np.zeros(len(ex["texts"]), dtype=np.float32)
    for i, text in enumerate(ex["texts"]):
        pool = set(content_tokens(ex["titles"][i] + " " + text[:400]))
        scores[i] = len(q_toks & pool) / (len(q_toks) + 1e-6)
    return scores


def top_indices(scores: np.ndarray, k: int) -> list[int]:
    k = min(k, len(scores))
    order = np.argsort(-scores)
    return [int(i) for i in order[:k]]


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


class Profile:
    def __init__(self, names: list[str], dim: int):
        self.names = names
        n = len(names)
        self.rho = np.full(n, 0.25, dtype=np.float64)
        self.eta = np.full(n, 0.25, dtype=np.float64)
        self.cost = np.ones(n, dtype=np.float64)
        self.R = np.eye(n, dtype=np.float64)
        self.E = np.zeros((n, dim), dtype=np.float64)
        self.phi = np.zeros(n, dtype=np.float64)
        self.rho_raw = self.rho.copy()
        self.eta_raw = self.eta.copy()

    def index(self, name: str) -> int:
        return self.names.index(name)


def estimate_profiles(cal: list[dict], names: list[str], span_f1: dict[str, np.ndarray] | None) -> Profile:
    dim = cal[0]["q_emb"].shape[0]
    prof = Profile(names, dim)
    n = len(names)
    recall = np.zeros((len(cal), n))
    contain = np.zeros((len(cal), n))
    costs = np.zeros((len(cal), n))
    title_sets: list[list[set[str]]] = []
    hit_queries = [[] for _ in range(n)]

    for qi, ex in enumerate(cal):
        sets = []
        for ai, name in enumerate(names):
            weak_id = int(name.split("weak")[-1]) if name.startswith("weak") else 0
            scores = agent_scores(name, ex, weak_id=weak_id)
            idxs = top_indices(scores, 3)
            titles = [ex["titles"][j] for j in idxs]
            text = " ".join(ex["texts"][j] for j in idxs)
            recall[qi, ai] = support_recall(titles, ex["gold_titles"])
            contain[qi, ai] = answer_hit(text, ex["answers"])
            costs[qi, ai] = sum(len(ex["texts"][j].split()) for j in idxs[:2]) + 12
            sets.append(set(titles))
            if recall[qi, ai] > 0:
                hit_queries[ai].append(ex["q_emb"])
        title_sets.append(sets)

    prof.rho_raw = recall.mean(axis=0)
    if span_f1 is not None:
        prof.eta_raw = np.array([span_f1[name].mean() for name in names])
    else:
        prof.eta_raw = contain.mean(axis=0)
    prof.cost = costs.mean(axis=0)
    prof.cost = prof.cost / max(prof.cost.mean(), 1e-6)

    R = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            vals = [jaccard(title_sets[q][i], title_sets[q][j]) for q in range(len(cal))]
            R[i, j] = float(np.mean(vals))
    prof.R = R

    global_q = np.mean([ex["q_emb"] for ex in cal], axis=0)
    for i in range(n):
        if hit_queries[i]:
            proto = np.mean(hit_queries[i], axis=0)
        else:
            proto = global_q
        norm = np.linalg.norm(proto) + 1e-8
        prof.E[i] = proto / norm

    phi = shapley_credit(cal, names)
    prof.phi = phi
    phi_n = _minmax(phi)
    prof.rho = (1 - CREDIT_BLEND) * prof.rho_raw + CREDIT_BLEND * (0.15 + 0.85 * phi_n) * max(prof.rho_raw.mean(), 0.05)
    # Keep raw reasoning quality, then tilt it by marginal credit.
    prof.eta = (1 - CREDIT_BLEND) * prof.eta_raw + CREDIT_BLEND * prof.eta_raw.mean() * (0.35 + 0.9 * phi_n)
    prof.rho = np.clip(prof.rho, 0.02, 0.98)
    prof.eta = np.clip(prof.eta, 0.02, 0.98)
    return prof


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return np.full_like(x, 0.5)
    return (x - lo) / (hi - lo)


def shapley_credit(cal: list[dict], names: list[str]) -> np.ndarray:
    n = len(names)
    acc = np.zeros(n)
    # Weights depend only on coalition size, so build them once per roster.
    size_weight = [1.0 / (n * math.comb(n - 1, r)) for r in range(n)]
    for qi, ex in enumerate(cal):
        cached_titles = []
        cached_norm = []
        for name in names:
            weak_id = int(name.split("weak")[-1]) if name.startswith("weak") else 0
            scores = agent_scores(name, ex, weak_id=weak_id)
            idxs = top_indices(scores, 3)
            cached_titles.append({ex["titles"][j] for j in idxs})
            cached_norm.append(normalize_answer(" ".join(ex["texts"][j] for j in idxs)))
        golds = [normalize_answer(a) for a in ex["answers"]]
        golds = [g for g in golds if g]
        values = np.zeros(1 << n)
        for mask in range(1 << n):
            if mask == 0:
                continue
            titles: set[str] = set()
            parts = []
            bit = 0
            m = mask
            while m:
                if m & 1:
                    titles |= cached_titles[bit]
                    parts.append(cached_norm[bit])
                m >>= 1
                bit += 1
            rec = support_recall(list(titles), ex["gold_titles"])
            blob = " ".join(parts)
            hit = 1.0 if any(g in blob for g in golds) else 0.0
            values[mask] = 0.5 * rec + 0.5 * hit
        for i in range(n):
            bit = 1 << i
            total = 0.0
            for mask in range(1 << n):
                if mask & bit:
                    continue
                total += size_weight[int(mask.bit_count())] * (values[mask | bit] - values[mask])
            acc[i] += total
        if (qi + 1) % 50 == 0:
            print(f"  shapley {qi + 1}/{len(cal)} n={n}", flush=True)
    return acc / max(len(cal), 1)


def match_scores(prof: Profile, q_emb: np.ndarray) -> np.ndarray:
    cos = prof.E @ q_emb
    return (cos + 1.0) / 2.0


def competence(prof: Profile, q_emb: np.ndarray, use_profile: bool, use_credit: bool) -> np.ndarray:
    if not use_profile:
        return np.full(len(prof.names), 0.55)
    rho = prof.rho if use_credit else prof.rho_raw
    eta = prof.eta if use_credit else prof.eta_raw
    rho = np.clip(rho, 0.02, 0.98)
    eta = np.clip(eta, 0.02, 0.98)
    return match_scores(prof, q_emb) * rho * eta


def set_utility(S: list[int], p: np.ndarray, R: np.ndarray) -> float:
    if not S:
        return 0.0
    order = sorted(S, key=lambda i: float(p[i]), reverse=True)
    residual = 1.0
    utility = 0.0
    chosen: list[int] = []
    for i in order:
        red = 1.0
        for j in chosen:
            red *= max(0.0, 1.0 - LAM_R * float(R[i, j]))
        gain = residual * float(p[i]) * red
        utility += gain
        residual = max(0.0, residual - gain)
        chosen.append(i)
    return utility


def form_coalition(p: np.ndarray, R: np.ndarray, cost: np.ndarray, use_cost: bool, kmax: int = K_MAX) -> list[int]:
    lam_c = LAM_C if use_cost else 0.0
    lam_comm = LAM_COMM if use_cost else 0.0
    selected: list[int] = []
    while len(selected) < kmax:
        best_i = None
        best_score = -1e18
        base = set_utility(selected, p, R)
        for i in range(len(p)):
            if i in selected:
                continue
            delta = set_utility(selected + [i], p, R) - base
            score = delta - lam_c * float(cost[i]) - (lam_comm if selected else 0.0)
            if score > best_score:
                best_score = score
                best_i = i
        if best_i is None:
            break
        if selected and best_score <= 0:
            break
        selected.append(best_i)
    if not selected:
        score = p - lam_c * cost
        selected = [int(np.argmax(score))]
    return selected


def cached_agent_scores(ex: dict, name: str) -> np.ndarray:
    cache = ex.get("_score_cache")
    if cache is None:
        cache = {}
        ex["_score_cache"] = cache
    if name not in cache:
        weak_id = int(name.split("weak")[-1]) if name.startswith("weak") else 0
        cache[name] = agent_scores(name, ex, weak_id=weak_id)
    return cache[name]


def build_offers(ex: dict, names: list[str]) -> list[dict]:
    offers = []
    for name in names:
        scores = cached_agent_scores(ex, name)
        for idx in top_indices(scores, COVER_SLOTS):
            offers.append(
                {
                    "agent": name,
                    "idx": int(idx),
                    "title": ex["titles"][idx],
                    "text": clip_words(ex["texts"][idx]),
                }
            )
    return offers


def cover_components(pages: list[dict], question: str) -> tuple[float, float]:
    """Term coverage and cross-page title closure of the pages the reader will see.

    Closure counts a selected title only when a different selected page names it.
    Adding the page that a dossier already cites therefore raises closure.
    A second copy of the same title does not.
    """
    if not pages:
        return 0.0, 0.0
    blob = " ".join((page["title"] + " " + page["text"]).lower() for page in pages)
    terms = content_tokens(question)
    if terms:
        tau = sum(term in blob for term in terms) / len(terms)
    else:
        tau = 0.0
    closed = 0
    for i, page in enumerate(pages):
        key = page["title"].lower().strip()
        if len(key) <= 3:
            continue
        for j, other in enumerate(pages):
            if i == j:
                continue
            other_blob = (other["title"] + " " + other["text"]).lower()
            if key in other_blob:
                closed += 1
                break
    return tau, closed / len(pages)


def mix_cover(tau: float, chi: float, alpha: float, use_term: bool, use_chain: bool) -> float:
    if use_term and use_chain:
        return alpha * tau + (1.0 - alpha) * chi
    if use_term:
        return tau
    if use_chain:
        return chi
    return 0.0


def _page_value(question: str, pages: list[dict], use_term: bool, use_chain: bool) -> float:
    tau, chi = cover_components(pages, question)
    if use_term and use_chain:
        return tau + COVER_BETA * chi
    if use_term:
        return tau
    if use_chain:
        return chi
    return 0.0


def fill_cover(
    ex: dict,
    offers: list[dict],
    alpha: float,
    lam: float,
    comm: float,
    use_term: bool,
    use_chain: bool,
    use_cost: bool,
    require_cite: bool = True,
) -> tuple[list[str], list[dict]]:
    """Anchor-locked complement swap inside a four-page budget.

    Slots start as the dense top four. A page that covers no query term the others
    miss may be replaced by one page from another specialist. That page has to be
    named by the pages that stay, and the swap has to raise term coverage plus a
    closure bonus. The dense copy is not a candidate. alpha, lam, and comm are
    retained so older call sites keep working; the swap rule does not use them.
    """
    del alpha, lam, comm, use_cost
    selected: list[dict] = []
    held = set()
    for offer in offers:
        if offer["agent"] != "dense" or offer["title"] in held:
            continue
        selected.append(offer)
        held.add(offer["title"])
        if len(selected) == COVER_SLOTS:
            break
    if len(selected) < 2:
        return ["dense"] if selected else [], selected

    scores = cached_agent_scores(ex, "dense")
    base_v = _page_value(ex["question"], selected, use_term, use_chain)
    drop_at = len(selected) - 1
    q_terms = content_tokens(ex["question"])
    for i, page in enumerate(selected):
        others = selected[:i] + selected[i + 1 :]
        page_blob = (page["title"] + " " + page["text"]).lower()
        other_blob = " ".join((p["title"] + " " + p["text"]).lower() for p in others)
        mine = {t for t in q_terms if t in page_blob} - {t for t in q_terms if t in other_blob}
        if not mine:
            drop_at = i
            break
    core = selected[:drop_at] + selected[drop_at + 1 :]
    blob = " ".join((p["title"] + " " + p["text"]).lower() for p in core)
    best_pages = selected
    best_v = base_v
    for cand in offers:
        if cand["agent"] in {"dense", "redundant"} or cand["title"] in held:
            continue
        if float(scores[cand["idx"]]) < 0.0:
            continue
        key = cand["title"].lower().strip()
        if require_cite and (len(key) <= 3 or key not in blob):
            continue
        trial = core + [cand]
        trial_v = _page_value(ex["question"], trial, use_term, use_chain)
        if trial_v > best_v + 1e-6:
            best_v = trial_v
            best_pages = trial
    opened: list[str] = []
    for page in best_pages:
        if page["agent"] not in opened:
            opened.append(page["agent"])
    return opened, best_pages


def _dense_pack(offers: list[dict]) -> list[dict]:
    selected = []
    held = set()
    for offer in offers:
        if offer["agent"] != "dense" or offer["title"] in held:
            continue
        selected.append(dict(offer))
        held.add(offer["title"])
        if len(selected) == COVER_SLOTS:
            break
    return selected


def _redundant_index(question: str, selected: list[dict]) -> int | None:
    q_terms = content_tokens(question)
    for i, page in enumerate(selected):
        others = selected[:i] + selected[i + 1 :]
        page_blob = (page["title"] + " " + page["text"]).lower()
        other_blob = " ".join((p["title"] + " " + p["text"]).lower() for p in others)
        mine = {t for t in q_terms if t in page_blob} - {t for t in q_terms if t in other_blob}
        if not mine:
            return i
    return None


def _cite_blob(pages: list[dict]) -> str:
    return " ".join((p["title"] + " " + p["text"]).lower() for p in pages)


def select_hop(
    ex: dict,
    offers: list[dict],
    rounds: int = 2,
    message_only: bool = True,
    require_cite: bool = True,
) -> tuple[list[str], list[dict]]:
    """Sequential coalition on a public dossier.

    Each round, a specialist may send one page. The page is feasible only when
    its title is already written on the dossier. The coalition accepts the message
    with the largest increase in evidence value and uses it to replace the first
    term-redundant slot. With message_only, later rounds read titles only from
    pages the coalition has already accepted, so a second hop is a consequence of
    the first message. The page budget stays at four.
    """
    selected = _dense_pack(offers)
    if len(selected) < 2:
        opened = ["dense"] if selected else []
        return opened, selected
    scores = cached_agent_scores(ex, "dense")
    for rnd in range(rounds):
        drop_at = _redundant_index(ex["question"], selected)
        if drop_at is None:
            break
        core = selected[:drop_at] + selected[drop_at + 1 :]
        if message_only and rnd > 0:
            citers = [page for page in core if page.get("admitted")]
            if not citers:
                break
            blob = _cite_blob(citers)
        else:
            blob = _cite_blob(core)
        held = {page["title"] for page in selected}
        base_v = _page_value(ex["question"], selected, True, True)
        best = None
        best_v = base_v
        for cand in offers:
            if cand["agent"] in {"dense", "redundant"} or cand["title"] in held:
                continue
            if float(scores[cand["idx"]]) < 0.0:
                continue
            key = cand["title"].lower().strip()
            if require_cite and (len(key) <= 3 or key not in blob):
                continue
            trial = core + [dict(cand, admitted=True)]
            trial_v = _page_value(ex["question"], trial, True, True)
            if trial_v > best_v + 1e-6:
                best_v = trial_v
                best = trial
        if best is None:
            break
        selected = best
    opened: list[str] = []
    for page in selected:
        if page["agent"] not in opened:
            opened.append(page["agent"])
    return opened, selected


def rrf_pages(ex: dict, k: int = 60) -> tuple[list[str], list[dict]]:
    """Reciprocal rank fusion of dense retrieval and BM25, four pages."""
    dense = cached_agent_scores(ex, "dense")
    bm25 = cached_agent_scores(ex, "bm25")
    n = len(dense)
    dense_rank = np.empty(n, dtype=np.int32)
    bm25_rank = np.empty(n, dtype=np.int32)
    dense_rank[np.argsort(-dense)] = np.arange(n)
    bm25_rank[np.argsort(-bm25)] = np.arange(n)
    fused = 1.0 / (k + dense_rank) + 1.0 / (k + bm25)
    pages = []
    held = set()
    for idx in top_indices(fused, COVER_SLOTS * 2):
        title = ex["titles"][idx]
        if title in held:
            continue
        held.add(title)
        agent = "dense" if dense_rank[idx] <= bm25_rank[idx] else "bm25"
        pages.append(
            {
                "agent": agent,
                "idx": int(idx),
                "title": title,
                "text": clip_words(ex["texts"][idx]),
            }
        )
        if len(pages) == COVER_SLOTS:
            break
    opened: list[str] = []
    for page in pages:
        if page["agent"] not in opened:
            opened.append(page["agent"])
    return opened, pages


def _page_of(ex: dict, idx: int, agent: str, admitted: bool = False) -> dict:
    return {
        "agent": agent,
        "idx": int(idx),
        "title": ex["titles"][idx],
        "text": clip_words(ex["texts"][idx]),
        "admitted": admitted,
    }


def _dense_order(ex: dict) -> list[int]:
    scores = cached_agent_scores(ex, "dense")
    order = []
    held: set[str] = set()
    for idx in top_indices(scores, len(scores)):
        title = ex["titles"][idx]
        if title in held:
            continue
        held.add(title)
        order.append(int(idx))
        if len(order) == COVER_SLOTS:
            break
    return order


def select_anchor(
    ex: dict,
    offers: list[dict],
    require_cite: bool = True,
    require_term: bool = False,
) -> tuple[list[str], list[dict]]:
    """Keep the dense agent's first two pages, then fill two slots.

    A specialist page is feasible when the current dossier names its title.
    It is accepted only when it raises evidence value above the next dense page.
    Otherwise the slot stays with dense retrieval, so the four-page dense set is
    the fallback. The redundant copy cannot send a message.
    """
    order = _dense_order(ex)
    if len(order) < 2:
        pages = [_page_of(ex, idx, "dense") for idx in order]
        return (["dense"] if pages else []), pages
    pages = [_page_of(ex, order[0], "dense"), _page_of(ex, order[1], "dense")]
    dense_pos = 2
    scores = cached_agent_scores(ex, "dense")
    for _ in range(COVER_SLOTS - 2):
        held = {page["title"] for page in pages}
        blob = _cite_blob(pages)
        nxt = None
        if dense_pos < len(order):
            nxt = _page_of(ex, order[dense_pos], "dense")
        best = nxt
        best_v = _page_value(ex["question"], pages + ([nxt] if nxt else []), True, True) if nxt else -1.0
        for cand in offers:
            if cand["agent"] in {"dense", "redundant"} or cand["title"] in held:
                continue
            if float(scores[cand["idx"]]) < 0.0:
                continue
            key = cand["title"].lower().strip()
            if require_cite and (len(key) <= 3 or key not in blob):
                continue
            if require_term:
                terms = content_tokens(ex["question"])
                cand_blob = (cand["title"] + " " + cand["text"]).lower()
                have = _cite_blob(pages)
                if not any(term in cand_blob and term not in have for term in terms):
                    continue
            trial = dict(cand)
            trial["admitted"] = True
            trial_v = _page_value(ex["question"], pages + [trial], True, True)
            if trial_v > best_v + 1e-6:
                best_v = trial_v
                best = trial
        if best is None:
            break
        pages.append(best)
        if nxt is not None and best["title"] == nxt["title"] and best["agent"] == "dense":
            dense_pos += 1
    opened: list[str] = []
    for page in pages:
        if page["agent"] not in opened:
            opened.append(page["agent"])
    return opened, pages


def mention_pages(ex: dict) -> tuple[list[str], list[dict]]:
    """Closed-pool iterative retrieval: follow a title the current pages already name.

    This is the link-following step of multi-hop retrieval inside the released
    candidate pool. It does not call the reader between hops.
    """
    scores = cached_agent_scores(ex, "dense")
    order = top_indices(scores, len(scores))
    selected_idx: list[int] = []
    held: set[str] = set()
    for idx in order:
        title = ex["titles"][idx]
        if title in held:
            continue
        selected_idx.append(int(idx))
        held.add(title)
        if len(selected_idx) == 2:
            break
    while len(selected_idx) < COVER_SLOTS:
        blob = " ".join(ex["texts"][i].lower() + " " + ex["titles"][i].lower() for i in selected_idx)
        best_i = None
        best_score = -1e18
        for idx, title in enumerate(ex["titles"]):
            if title in held:
                continue
            key = title.lower().strip()
            if len(key) > 3 and key in blob and float(scores[idx]) > best_score:
                best_score = float(scores[idx])
                best_i = idx
        if best_i is None:
            for idx in order:
                if ex["titles"][idx] not in held:
                    best_i = int(idx)
                    break
        if best_i is None:
            break
        selected_idx.append(int(best_i))
        held.add(ex["titles"][best_i])
    pages = [
        {
            "agent": "mention",
            "idx": idx,
            "title": ex["titles"][idx],
            "text": clip_words(ex["texts"][idx]),
        }
        for idx in selected_idx
    ]
    return ["mention"], pages


def select_follow(ex: dict, policy: str = "dense") -> tuple[list[str], list[dict]]:
    """Fill the non-anchor slots with a pool title the dossier already names.

    policy chooses which named title is sent:
    dense, bm25, and bridge use that agent's score;
    term prefers a title that adds an uncovered question word;
    vote prefers a title that more than one specialist ranked in its top four.
    """
    order = _dense_order(ex)
    if not order:
        return [], []
    protect = 1 if policy == "from1" else min(2, len(order))
    pages = [_page_of(ex, idx, "dense") for idx in order[:protect]]
    score_map = {
        "dense": cached_agent_scores(ex, "dense"),
        "bm25": cached_agent_scores(ex, "bm25"),
        "bridge": cached_agent_scores(ex, "bridge"),
        "from1": cached_agent_scores(ex, "dense"),
        "term": cached_agent_scores(ex, "dense"),
        "vote": cached_agent_scores(ex, "dense"),
    }
    rank_scores = score_map[policy]
    offers = build_offers(ex, SPECIALISTS) if policy == "vote" else []
    votes: dict[int, int] = {}
    for offer in offers:
        votes[offer["idx"]] = votes.get(offer["idx"], 0) + 1
    while len(pages) < COVER_SLOTS:
        held = {page["title"] for page in pages}
        blob = " ".join((ex["titles"][p["idx"]] + " " + ex["texts"][p["idx"]]).lower() for p in pages)
        named = []
        for idx, title in enumerate(ex["titles"]):
            if title in held:
                continue
            key = title.lower().strip()
            if len(key) > 3 and key in blob:
                named.append(idx)
        if not named:
            nxt = next((idx for idx in order if ex["titles"][idx] not in held), None)
            if nxt is None:
                break
            pages.append(_page_of(ex, nxt, "dense"))
            continue
        have = blob

        def sort_key(idx: int) -> tuple:
            if policy == "term":
                cand = (ex["titles"][idx] + " " + ex["texts"][idx]).lower()
                extra = sum(term in cand and term not in have for term in content_tokens(ex["question"]))
                return (extra, float(rank_scores[idx]))
            if policy == "vote":
                return (votes.get(idx, 0), float(rank_scores[idx]))
            return (float(rank_scores[idx]),)

        best = max(named, key=sort_key)
        sender = "dense" if policy in {"dense", "term", "from1"} else policy
        pages.append(_page_of(ex, best, sender, admitted=True))
    opened: list[str] = []
    for page in pages:
        if page["agent"] not in opened:
            opened.append(page["agent"])
    return opened, pages


def missing_bridge_titles(ex: dict, passage_ids: list[int]) -> list[str]:
    retrieved = {ex["titles"][i] for i in passage_ids}
    blob = " ".join(ex["texts"][i] for i in passage_ids).lower()
    missing = []
    for title in ex["titles"]:
        if title in retrieved:
            continue
        key = title.lower().strip()
        if len(key) > 3 and key in blob:
            missing.append(title)
    return missing


def uncertainty(question: str, texts: list[str], bridge_missing: bool) -> float:
    terms = content_tokens(question)
    if not terms:
        term_gap = 0.0
    else:
        blob = " ".join(texts).lower()
        covered = sum(t in blob for t in terms) / len(terms)
        term_gap = 1.0 - covered
    if bridge_missing:
        return max(term_gap, 0.72)
    return term_gap


def maybe_revise(selected: list[int], prof: Profile, ex: dict, p: np.ndarray, passage_ids: list[int], use_cost: bool) -> tuple[list[int], bool]:
    texts = [ex["texts"][i] for i in passage_ids]
    missing = missing_bridge_titles(ex, passage_ids)
    if uncertainty(ex["question"], texts, bool(missing)) < REVISION_UNC:
        return selected, False
    if len(selected) >= K_MAX + 1:
        return selected, False
    if passage_ids:
        mean_p = ex["P"][passage_ids].mean(axis=0)
        mean_p = mean_p / (np.linalg.norm(mean_p) + 1e-8)
        resid = ex["q_emb"] - np.dot(ex["q_emb"], mean_p) * mean_p
        resid = resid / (np.linalg.norm(resid) + 1e-8)
    else:
        resid = ex["q_emb"]
    p_res = match_scores(prof, resid) * (p / np.maximum(match_scores(prof, ex["q_emb"]), 1e-3))
    best_i = None
    best_score = 0.02
    base = set_utility(selected, p_res, prof.R)
    lam_c = LAM_C if use_cost else 0.0
    for i in range(len(p_res)):
        if i in selected:
            continue
        delta = set_utility(selected + [i], p_res, prof.R) - base
        name = prof.names[i]
        weak_id = int(name.split("weak")[-1]) if name.startswith("weak") else 0
        top_title = ex["titles"][int(np.argmax(agent_scores(name, ex, weak_id=weak_id)))]
        bonus = 0.18 if top_title in missing else 0.0
        score = delta + bonus - lam_c * float(prof.cost[i]) - LAM_COMM
        if score > best_score:
            best_score = score
            best_i = i
    if best_i is None:
        return selected, False
    return selected + [best_i], True


def evidence_blocks(ex: dict, names: list[str], chosen: list[str], per_agent: int, cap: int, spans: dict[str, str]) -> list[dict]:
    ranked = {}
    for name in chosen:
        weak_id = int(name.split("weak")[-1]) if name.startswith("weak") else 0
        scores = agent_scores(name, ex, weak_id=weak_id)
        ranked[name] = top_indices(scores, per_agent)
    blocks = []
    seen = set()
    round_id = 0
    while len(blocks) < cap and round_id < per_agent:
        progress = False
        for name in chosen:
            if round_id >= len(ranked[name]):
                continue
            idx = ranked[name][round_id]
            title = ex["titles"][idx]
            if title in seen:
                continue
            seen.add(title)
            blocks.append(
                {
                    "agent": name,
                    "title": title,
                    "text": clip_words(ex["texts"][idx]),
                    "span": spans.get(f"{name}:{idx}", ""),
                }
            )
            progress = True
            if len(blocks) >= cap:
                break
        if not progress:
            break
        round_id += 1
    return blocks


def render_prompt(question: str, blocks: list[dict], style: str) -> str:
    if style == "norag":
        return (
            "Answer the question with a short fact. "
            "Reply with only the answer.\n\n"
            f"Question: {question}\nAnswer:"
        )
    if style == "debate":
        lines = [
            "Two specialists retrieved evidence for the same question.",
            "Read both dossiers and reply with only the short answer.",
            "",
        ]
        for block in blocks:
            lines.append(f"Specialist {block['agent']}:")
            lines.append(f"Title: {block['title']}")
            lines.append(block["text"])
            if block["span"]:
                lines.append(f"Proposed span: {block['span']}")
            lines.append("")
        lines.append(f"Question: {question}")
        lines.append("Answer:")
        return "\n".join(lines)
    lines = [
        "Use the evidence dossiers to answer the question.",
        "Reply with only the short answer.",
        "",
    ]
    for i, block in enumerate(blocks, start=1):
        lines.append(f"[{i}] {block['agent']} | {block['title']}")
        lines.append(block["text"])
        if block["span"]:
            lines.append(f"Proposed span: {block['span']}")
        lines.append("")
    lines.append(f"Question: {question}")
    lines.append("Answer:")
    return "\n".join(lines)


def choose_method(method: str, prof: Profile, ex: dict) -> tuple[list[str], str, bool, list[dict] | None]:
    names = prof.names
    use_profile = method != "no_profile"
    use_credit = method != "no_credit"
    use_cost = method != "no_cost"
    p = competence(prof, ex["q_emb"], use_profile=use_profile, use_credit=use_credit)
    if method == "norag":
        return [], "norag", False, None
    if method == "single":
        return ["dense"], "rag", False, None
    if method == "router":
        score = p - (LAM_C if use_cost else 0.0) * prof.cost
        return [names[int(np.argmax(score))]], "rag", False, None
    if method == "fixed":
        return list(names), "rag", False, None
    if method == "debate":
        order = list(np.argsort(-(p - 0.02 * prof.cost)))
        picked = [names[int(order[0])]]
        for idx in order[1:]:
            j = int(idx)
            if prof.R[names.index(picked[0]), j] < 0.85:
                picked.append(names[j])
                break
        if len(picked) == 1 and len(order) > 1:
            picked.append(names[int(order[1])])
        return picked, "debate", False, None
    if method in COVER_METHODS:
        offers = build_offers(ex, names)
        agents, pages = fill_cover(
            ex,
            offers,
            alpha=COVER_ALPHA,
            lam=COVER_LAM,
            comm=COVER_COMM,
            use_term=method != "no_term",
            use_chain=method != "no_chain",
            use_cost=True,
            require_cite=method != "no_cite",
        )
        return agents, "rag", False, pages
    selected = form_coalition(p, prof.R, prof.cost, use_cost=use_cost)
    chosen = [names[i] for i in selected]
    revised = False
    if method != "no_rev":
        preview = evidence_blocks(ex, names, chosen, per_agent=2, cap=COALITION_CAP, spans={})
        ids = [ex["titles"].index(b["title"]) for b in preview]
        new_sel, revised = maybe_revise(selected, prof, ex, p, ids, use_cost=use_cost)
        chosen = [names[i] for i in new_sel]
    return chosen, "rag", revised, None


def method_budget(method: str, n_agents: int) -> tuple[int, int]:
    if method == "single" or method == "router":
        return 4, SINGLE_K
    if method == "fixed":
        return 2, FIXED_CAP
    if method == "debate":
        return 3, 6
    if method == "norag":
        return 0, 0
    if n_agents <= 1:
        return 4, SINGLE_K
    return 2, min(6, max(4, 2 * n_agents))


class SpanReader:
    def __init__(self):
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.torch = torch
        name = "deepset/roberta-base-squad2"
        self.tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
        self.model = AutoModelForQuestionAnswering.from_pretrained(name, local_files_only=True)
        self.model.to("cuda:0")
        self.model.eval()

    def answer_batch(self, questions: list[str], contexts: list[str], batch_size: int = 16) -> list[tuple[str, float]]:
        results = []
        torch = self.torch
        for start in range(0, len(questions), batch_size):
            qs = questions[start : start + batch_size]
            cs = contexts[start : start + batch_size]
            batch = self.tokenizer(
                qs,
                cs,
                padding=True,
                truncation="only_second",
                max_length=384,
                return_tensors="pt",
                return_offsets_mapping=True,
            )
            offsets = batch["offset_mapping"]
            encodings = batch.encodings
            model_inputs = {k: v.to(self.model.device) for k, v in batch.items() if k != "offset_mapping"}
            with torch.inference_mode():
                out = self.model(**model_inputs)
            start_logits = out.start_logits.detach().float().cpu()
            end_logits = out.end_logits.detach().float().cpu()
            for b in range(len(qs)):
                seq_ids = encodings[b].sequence_ids
                allowed = [i for i, sid in enumerate(seq_ids) if sid == 1]
                if not allowed:
                    results.append(("", 0.0))
                    continue
                s_logit = start_logits[b].clone()
                e_logit = end_logits[b].clone()
                mask = torch.full_like(s_logit, -1e4)
                mask[allowed] = 0.0
                s_logit = s_logit + mask
                e_logit = e_logit + mask
                s_idx = int(torch.argmax(s_logit))
                e_idx = int(torch.argmax(e_logit[s_idx:])) + s_idx
                score = float(torch.softmax(s_logit, dim=0)[s_idx] * torch.softmax(e_logit, dim=0)[e_idx])
                s_char, _ = offsets[b, s_idx].tolist()
                _, e_char = offsets[b, e_idx].tolist()
                text = cs[b][int(s_char) : int(e_char)].strip()
                results.append((text, score))
        return results


def enc_sequence_ids(tokenizer, question: str, context: str) -> list[int | None]:
    encoded = tokenizer(question, context, truncation="only_second", max_length=384)
    return encoded.sequence_ids()


def build_spans(examples: list[dict], names: list[str], qa: SpanReader) -> dict[str, dict[str, str]]:
    questions = []
    contexts = []
    keys = []
    for ex in examples:
        for name in names:
            weak_id = int(name.split("weak")[-1]) if name.startswith("weak") else 0
            scores = agent_scores(name, ex, weak_id=weak_id)
            for idx in top_indices(scores, 2):
                questions.append(ex["question"])
                contexts.append(clip_words(ex["texts"][idx], 160))
                keys.append((ex["id"], name, idx))
    spans: dict[str, dict[str, str]] = {ex["id"]: {} for ex in examples}
    if not questions:
        return spans
    outputs = qa.answer_batch(questions, contexts, batch_size=16)
    for (eid, name, idx), (answer, score) in zip(keys, outputs):
        if score < 0.05 or not answer:
            continue
        spans[eid][f"{name}:{idx}"] = answer
    return spans


def span_f1_matrix(cal: list[dict], names: list[str], spans: dict[str, dict[str, str]]) -> dict[str, np.ndarray]:
    out = {name: np.zeros(len(cal)) for name in names}
    for qi, ex in enumerate(cal):
        bag = spans.get(ex["id"], {})
        for name in names:
            vals = []
            for key, span in bag.items():
                if key.startswith(name + ":"):
                    _em, f1 = best_em_f1(span, ex["answers"])
                    vals.append(f1)
            out[name][qi] = float(np.mean(vals)) if vals else 0.0
    return out


def extract_answer(text: str) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    line = re.sub(r"^(answer\s*:\s*)", "", line, flags=re.I).strip().strip("\"'` ")
    low = line.lower()
    if low.startswith("yes"):
        return "yes"
    if low.startswith("no"):
        return "no"
    line = re.split(r"(?<=[.])\s", line)[0]
    words = line.split()
    if len(words) > 16:
        line = " ".join(words[:16])
    return line.strip(" .")


def load_cache(path: Path | None = None) -> dict[str, dict]:
    dest = path or CACHE_PATH
    cache = {}
    if dest.exists():
        with dest.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                cache[row["key"]] = row
    return cache


def append_cache(rows: list[dict], path: Path | None = None) -> None:
    dest = path or CACHE_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prompt_key(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()


def generate_missing(
    prompts: list[str],
    cache: dict[str, dict],
    batch_size: int,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    cache_path: Path | None = None,
) -> None:
    missing = [p for p in prompts if prompt_key(p) not in cache]
    # Preserve order uniqueness
    uniq = list(dict.fromkeys(missing))
    if not uniq:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # device_map crashes with an access violation on this Windows driver.
    # Load on CPU, then move the weights onto the free GPU.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        torch.cuda.init()
        print(f"moving {model_name} to cuda:0", flush=True)
        model.to("cuda:0")
    except OSError as exc:
        print(f"cpu load failed ({exc}); streaming weights to cuda:0", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
            device_map={"": "cuda:0"},
        )
    model.eval()
    print(f"generating {len(uniq)} prompts", flush=True)
    new_rows = []
    for start in range(0, len(uniq), batch_size):
        chunk = uniq[start : start + batch_size]
        chats = []
        for prompt in chunk:
            messages = [{"role": "user", "content": prompt}]
            chats.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        enc = tokenizer(chats, return_tensors="pt", padding=True, truncation=True, max_length=1536)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - t0
        input_len = enc["input_ids"].shape[1]
        per_time = elapsed / len(chunk)
        for i, prompt in enumerate(chunk):
            gen_ids = out[i, input_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            prompt_tokens = int(enc["attention_mask"][i].sum().item())
            ids = gen_ids.tolist()
            eos_id = tokenizer.eos_token_id
            if eos_id is not None and eos_id in ids:
                gen_tokens = ids.index(eos_id) + 1
            else:
                gen_tokens = len(ids)
            row = {
                "key": prompt_key(prompt),
                "text": text,
                "prompt_tokens": prompt_tokens,
                "gen_tokens": gen_tokens,
                "latency": per_time,
            }
            cache[row["key"]] = row
            new_rows.append(row)
        print(f"  batch {start + len(chunk)}/{len(uniq)} {elapsed:.1f}s", flush=True)
        append_cache(new_rows, cache_path)
        new_rows = []
    if new_rows:
        append_cache(new_rows, cache_path)
    del model
    torch.cuda.empty_cache()


def summarize(rows: list[dict]) -> dict:
    def mean(key):
        return float(np.mean([r[key] for r in rows])) if rows else 0.0

    def sem(key):
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        if len(vals) <= 1:
            return 0.0
        return float(vals.std(ddof=1) / math.sqrt(len(vals)))

    f1 = mean("f1")
    tokens = mean("tokens")
    return {
        "n": len(rows),
        "em": mean("em"),
        "f1": f1,
        "support": mean("support"),
        "tokens": tokens,
        "latency": mean("latency"),
        "size": mean("size"),
        "revise_rate": mean("revised"),
        "utility": f1 - tokens / UTILITY_TOKEN_SCALE,
        "em_se": sem("em"),
        "f1_se": sem("f1"),
        "tokens_se": sem("tokens"),
    }


def selection_freq(rows: list[dict], names: list[str]) -> dict[str, float]:
    counts = {name: 0 for name in names}
    for row in rows:
        for name in row["agents"]:
            if name in counts:
                counts[name] += 1
    n = max(len(rows), 1)
    return {k: v / n for k, v in counts.items()}


def prepare_dataset(dataset: str, n_cal: int, n_test: int, seed: int, encoder, qa):
    print(f"loading {dataset}", flush=True)
    cal, test = load_examples(dataset, n_cal, n_test, seed)
    attach_embeddings(cal + test, encoder)
    return cal, test


def run(mode: str) -> None:
    if mode == "smoke":
        datasets = ["hotpot"]
        n_cal, n_test = 8, 4
        methods = ["norag", "single", "router", "fixed", "debate", "coalition"]
        ablations: list[str] = []
        do_robust = False
        batch_size = 2
        robust_methods = ["fixed", "coalition", "router"]
    elif mode == "cover":
        datasets = ["hotpot", "2wiki", "musique"]
        n_cal, n_test = 200, 300
        methods = ["norag", "single", "router", "fixed", "debate", "residual", "cover"]
        ablations = ["no_chain", "no_term", "no_cite"]
        do_robust = True
        batch_size = 8
        robust_methods = ["fixed", "cover", "router"]
    else:
        datasets = ["hotpot", "2wiki", "musique"]
        n_cal, n_test = 200, 300
        methods = ["norag", "single", "router", "fixed", "debate", "coalition"]
        ablations = ["no_rev", "no_cost", "no_profile", "no_credit"]
        do_robust = True
        batch_size = 8
        robust_methods = ["fixed", "coalition", "router"]

    from sentence_transformers import SentenceTransformer
    import torch

    encoder = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda:0")
    qa = SpanReader()

    bundle = {}
    for dataset in datasets:
        cal, test = prepare_dataset(dataset, n_cal, n_test, seed=7, encoder=encoder, qa=qa)
        print(f"spans {dataset}", flush=True)
        main_names = list(MAIN_AGENTS)
        spans_cal = build_spans(cal, main_names, qa)
        spans_test = build_spans(test, main_names, qa)
        eta = span_f1_matrix(cal, main_names, spans_cal)
        print(f"profiles {dataset}", flush=True)
        prof = estimate_profiles(cal, main_names, eta)
        bundle[dataset] = {
            "cal": cal,
            "test": test,
            "spans": spans_test,
            "profile": prof,
        }
        print(
            "rho",
            {n: round(float(prof.rho[i]), 3) for i, n in enumerate(prof.names)},
            flush=True,
        )

    jobs = []

    def add_job(dataset, split_name, ex, method, prof, spans):
        t0 = time.perf_counter()
        chosen, style, revised, pages = choose_method(method, prof, ex)
        if method == "norag":
            blocks = []
        elif pages is not None:
            span_map = spans.get(ex["id"], {})
            blocks = [
                {
                    "agent": page["agent"],
                    "title": page["title"],
                    "text": page["text"],
                    "span": span_map.get(f"{page['agent']}:{page['idx']}", ""),
                }
                for page in pages
            ]
        else:
            per_agent, cap = method_budget(method, len(chosen))
            blocks = evidence_blocks(ex, prof.names, chosen, per_agent, cap, spans.get(ex["id"], {}))
        prompt = render_prompt(ex["question"], blocks, style)
        select_time = time.perf_counter() - t0
        titles = [b["title"] for b in blocks]
        jobs.append(
            {
                "dataset": dataset,
                "split": split_name,
                "method": method,
                "id": ex["id"],
                "answers": ex["answers"],
                "prompt": prompt,
                "agents": chosen,
                "revised": float(revised),
                "support": support_recall(titles, ex["gold_titles"]),
                "select_time": select_time,
                "size": float(len(chosen)),
            }
        )

    for dataset, pack in bundle.items():
        prof = pack["profile"]
        for method in methods:
            for ex in pack["test"]:
                add_job(dataset, "main", ex, method, prof, pack["spans"])
        if ablations:
            for method in ablations:
                for ex in pack["test"]:
                    add_job(dataset, "ablation", ex, method, prof, pack["spans"])
        if do_robust and dataset == "hotpot":
            robust_test = pack["test"][:150]
            weak_names = [f"weak{k}" for k in range(4)]
            print("robust spans", flush=True)
            cal_weak = build_spans(pack["cal"], weak_names, qa)
            test_weak = build_spans(robust_test, weak_names, qa)
            weak_eta = span_f1_matrix(pack["cal"], weak_names, cal_weak)
            for n_weak in [0, 1, 2, 4]:
                names = list(SPECIALISTS) + weak_names[:n_weak]
                eta = _eta_from_profile(pack["profile"], names, len(pack["cal"]))
                for wname in weak_names[:n_weak]:
                    eta[wname] = weak_eta[wname]
                print(f"robust profiles n_weak={n_weak}", flush=True)
                prof_r = estimate_profiles(pack["cal"], names, eta)
                extra_spans = {ex["id"]: dict(pack["spans"].get(ex["id"], {})) for ex in robust_test}
                for ex in robust_test:
                    for key, span in test_weak.get(ex["id"], {}).items():
                        if any(key.startswith(w + ":") for w in weak_names[:n_weak]):
                            extra_spans[ex["id"]][key] = span
                for method in robust_methods:
                    for ex in robust_test:
                        add_job(dataset, f"robust{n_weak}", ex, method, prof_r, extra_spans)

    del encoder
    del qa
    torch.cuda.empty_cache()
    size_log = defaultdict(list)
    rev_log = defaultdict(list)
    for job in jobs:
        if job["split"] == "main":
            size_log[(job["dataset"], job["method"])].append(job["size"])
            rev_log[(job["dataset"], job["method"])].append(job["revised"])
    for key in sorted(size_log):
        print("size", key, round(float(np.mean(size_log[key])), 3), "revise", round(float(np.mean(rev_log[key])), 3), flush=True)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    jobs_path = RESULT_DIR / f"{mode}_jobs.jsonl"
    with jobs_path.open("w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(json.dumps(job, ensure_ascii=False) + "\n")
    print("wrote jobs", len(jobs), jobs_path, flush=True)
    complete_jobs(jobs, mode, batch_size, n_cal, n_test)


def complete_jobs(jobs: list[dict], mode: str, batch_size: int, n_cal: int, n_test: int) -> None:
    import gc

    gc.collect()
    cache = load_cache()
    generate_missing([job["prompt"] for job in jobs], cache, batch_size=batch_size)

    records = []
    for job in jobs:
        gen = cache[prompt_key(job["prompt"])]
        pred = extract_answer(gen["text"])
        em, f1 = best_em_f1(pred, job["answers"])
        tokens = float(gen["prompt_tokens"] + gen["gen_tokens"])
        records.append(
            {
                "dataset": job["dataset"],
                "split": job["split"],
                "method": job["method"],
                "id": job["id"],
                "pred": pred,
                "gold": job["answers"][0],
                "em": em,
                "f1": f1,
                "support": job["support"],
                "tokens": tokens,
                "latency": job["select_time"] + float(gen["latency"]),
                "size": job["size"],
                "revised": job["revised"],
                "agents": job["agents"],
            }
        )

    grouped = defaultdict(list)
    for row in records:
        grouped[(row["dataset"], row["split"], row["method"])].append(row)

    summary = {}
    for (dataset, split, method), rows in sorted(grouped.items()):
        summary.setdefault(dataset, {}).setdefault(split, {})[method] = summarize(rows)
        if method in {"coalition", "cover", "residual", "no_chain", "no_term", "no_cite"}:
            present = sorted({agent for row in rows for agent in row["agents"]})
            names = list(dict.fromkeys(list(MAIN_AGENTS) + present))
            summary[dataset][split][method]["selection"] = selection_freq(rows, names)

    out = {
        "config": {
            "mode": mode,
            "n_cal": n_cal,
            "n_test": n_test,
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "embed": "BAAI/bge-base-en-v1.5",
            "reader": "deepset/roberta-base-squad2",
            "lam_r": LAM_R,
            "lam_c": LAM_C,
            "lam_comm": LAM_COMM,
            "k_max": K_MAX,
            "revision_unc": REVISION_UNC,
            "utility": "f1 - tokens/4000",
            "seed": 7,
            "cover_beta": COVER_BETA,
            "cover_rule": "anchor-locked complement swap",
        },
        "summary": summary,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULT_DIR / f"{mode}_summary.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    pred_path = RESULT_DIR / f"{mode}_predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2)[:4000])
    print("wrote", out_path)


_QA = None


def qa_reload():
    global _QA
    if _QA is None:
        _QA = SpanReader()
    return _QA


def _eta_from_profile(prof: Profile, names: list[str], n_rows: int) -> dict[str, np.ndarray]:
    out = {}
    for name in names:
        if name in prof.names:
            val = float(prof.eta_raw[prof.index(name)])
        else:
            val = 0.05
        out[name] = np.full(n_rows, val)
    return out


def _cover_stats(
    examples: list[dict],
    names: list[str],
    use_term: bool = True,
    use_chain: bool = True,
    require_cite: bool = True,
) -> dict:
    sup_c = []
    sup_d = []
    sizes = []
    partners = Counter()
    changed = weak_in = 0
    for ex in examples:
        offers = build_offers(ex, names)
        agents, pages = fill_cover(
            ex, offers, 0.0, 0.0, 0.0, use_term, use_chain, True, require_cite=require_cite
        )
        sup_c.append(support_recall([p["title"] for p in pages], ex["gold_titles"]))
        dense_idx = top_indices(cached_agent_scores(ex, "dense"), COVER_SLOTS)
        dense_titles = [ex["titles"][i] for i in dense_idx]
        sup_d.append(support_recall(dense_titles, ex["gold_titles"]))
        sizes.append(len(agents))
        swapped = [p["title"] for p in pages] != dense_titles[: len(pages)] and len(pages) == len(dense_titles)
        # Title identity, not agent identity: a swap replaces one dense page.
        if [p["title"] for p in pages] != dense_titles:
            changed += 1
        for name in agents:
            if name != "dense":
                partners[name] += 1
            if str(name).startswith("weak"):
                weak_in += 1
                break
        del swapped
    n = max(len(examples), 1)
    return {
        "support": float(np.mean(sup_c)),
        "dense_support": float(np.mean(sup_d)),
        "size": float(np.mean(sizes)),
        "changed": changed / n,
        "weak_open": weak_in / n,
        "partners": {k: v / n for k, v in sorted(partners.items())},
    }


def diagnose() -> None:
    """Freeze the anchor-swap closure weight on calibration support. Test is not scored."""
    global COVER_BETA
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda:0")
    prepared = {}
    for dataset in ["hotpot", "2wiki", "musique"]:
        cal, _test = load_examples(dataset, n_cal=200, n_test=300, seed=7)
        attach_embeddings(cal, encoder)
        prepared[dataset] = cal
        print(f"cal {dataset} {len(cal)}", flush=True)
    del encoder

    names = list(MAIN_AGENTS)
    rows = []
    for beta in (0.0, 0.2, 0.5):
        COVER_BETA = beta
        per = {dataset: _cover_stats(cal, names) for dataset, cal in prepared.items()}
        gains = {d: per[d]["support"] - per[d]["dense_support"] for d in per}
        row = {
            "beta": beta,
            "require_cite": True,
            "min_gain": min(gains.values()),
            "mean_gain": float(np.mean(list(gains.values()))),
            "per": per,
        }
        rows.append(row)
        print(
            f"beta={beta:.1f} cite=1 min={row['min_gain']:+.3f} mean={row['mean_gain']:+.3f}",
            {
                d: (
                    round(per[d]["support"], 3),
                    round(per[d]["dense_support"], 3),
                    round(per[d]["changed"], 3),
                    round(per[d]["size"], 2),
                )
                for d in per
            },
            flush=True,
        )

    COVER_BETA = 0.2
    nocite = {dataset: _cover_stats(cal, names, require_cite=False) for dataset, cal in prepared.items()}
    print(
        "beta=0.2 cite=0",
        {
            d: (
                round(nocite[d]["support"], 3),
                round(nocite[d]["dense_support"], 3),
                round(nocite[d]["changed"], 3),
            )
            for d in nocite
        },
        flush=True,
    )

    eligible = [row for row in rows if row["min_gain"] >= 0.0]
    if not eligible:
        print("no beta keeps support on every dataset; hparams not frozen", flush=True)
        return
    eligible.sort(key=lambda row: (row["min_gain"], row["mean_gain"]), reverse=True)
    best = eligible[0]
    COVER_BETA = float(best["beta"])
    weak_names = list(SPECIALISTS) + [f"weak{k}" for k in range(4)]
    weak = _cover_stats(prepared["hotpot"], weak_names)
    print(
        "FROZEN",
        {"beta": COVER_BETA, "min_gain": best["min_gain"], "mean_gain": best["mean_gain"]},
        "weak_open",
        round(weak["weak_open"], 3),
        "weak_support",
        round(weak["support"], 3),
        "partners",
        {d: best["per"][d]["partners"] for d in best["per"]},
        flush=True,
    )
    payload = {
        "rule": "anchor-locked complement swap",
        "beta": COVER_BETA,
        "min_gain": best["min_gain"],
        "mean_gain": best["mean_gain"],
        "weak_open_rate": weak["weak_open"],
        "per": best["per"],
        "nocite_beta_0_2": nocite,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / "cover_hparams.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", path, flush=True)


def load_cover_hparams() -> None:
    global COVER_BETA
    path = RESULT_DIR / "cover_hparams.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    COVER_BETA = float(data["beta"])
    print("frozen cover rule", data.get("rule"), {"beta": COVER_BETA}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full", "diagnose", "cover"], default="smoke")
    parser.add_argument("--resume-jobs", action="store_true")
    args = parser.parse_args()
    if args.mode == "diagnose":
        diagnose()
        return
    if args.mode == "cover":
        load_cover_hparams()
    if args.resume_jobs:
        mode = "full" if args.mode == "smoke" else args.mode
        jobs_path = RESULT_DIR / f"{mode}_jobs.jsonl"
        jobs = [json.loads(line) for line in jobs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        print("resume jobs", len(jobs), jobs_path, flush=True)
        complete_jobs(jobs, mode, batch_size=8, n_cal=200, n_test=300)
        return
    run(args.mode)


if __name__ == "__main__":
    main()
