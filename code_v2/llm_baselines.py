"""LLM-driven retrieval baselines with the shared int4 reader model:
  ircot    : interleaved CoT + dense retrieval (Trivedi et al., ACL'23), bge-large retriever
  setr     : SetR information-requirement set selection prompt (Lee et al., ACL'25), zero-shot
  llmpoint : RankRAG-style pointwise relevance P(True) from the LLM (Yu et al., NeurIPS'24), zero-shot
Writes results/retrieval/{method}[_{pool}]_{ds}.json with a ranked list per question.
"""
import argparse
import math
import re

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from common import DATASETS, MODELS, RES, jdump, jload, load_bench, recall_at
from first_stage import encode

QP = "Represent this sentence for searching relevant passages: "


def fmt(docs, i, max_words=150):
    t, _, x = docs[i].partition("\n")
    return f"{t}: {' '.join(x.split()[:max_words])}"


def pools_for(ds, args):
    if args.pool_file:
        return jload(RES / "pools" / f"{args.pool_file}_{ds}.json")
    return [r[:20] for r in jload(RES / "first_stage" / f"bge_{ds}.json")["ranked"]]


def run_setr(llm, tok, sp_long, docs, qs, pools):
    prompts = []
    for q, pool in zip(qs, pools):
        ctx = "\n".join(f"[{k + 1}] {fmt(docs, i)}" for k, i in enumerate(pool))
        user = (f"I will provide you with {len(pool)} passages, each indicated by a numerical identifier []. "
                f"Select the passages based on their relevance to the search query: {q['question']}.\n\n{ctx}\n\n"
                f"Search Query: {q['question']}\n\nPlease follow the steps below:\n"
                "Step 1. Please list up the information requirements to answer the query.\n"
                "Step 2. for each requirement in Step 1, find the passages that has the information of the requirement.\n"
                "Step 3. Choose the passages that mostly covers clear and diverse information to answer the query. "
                "Number of passages is unlimited. The format of final output should be '### Final Selection: [] []', "
                "e.g., ### Final Selection: [2] [1].")
        prompts.append(tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False))
    outs = llm.generate(prompts, sp_long, use_tqdm=True)
    ranks, sizes = [], []
    for o, pool in zip(outs, pools):
        txt = o.outputs[0].text
        tail = txt.split("Final Selection")[-1] if "Final Selection" in txt else ""
        sel = []
        for m in re.findall(r"\[(\d+)\]", tail):
            k = int(m) - 1
            if 0 <= k < len(pool) and pool[k] not in sel:
                sel.append(pool[k])
        sizes.append(len(sel))
        ranks.append(sel + [i for i in pool if i not in sel])
    print("setr mean selected", np.mean(sizes))
    return ranks


def run_llmpoint(llm, tok, docs, qs, pools):
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    prompts, owners = [], []
    for qi, (q, pool) in enumerate(zip(qs, pools)):
        for i in pool:
            user = (f"Passage: {fmt(docs, i)}\n\nQuestion: {q['question']}\n\n"
                    "Does the passage contain information that helps answer the question (possibly as one step of a "
                    "multi-step reasoning chain)? Answer True or False.")
            prompts.append(tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False))
            owners.append((qi, i))
    outs = llm.generate(prompts, sp, use_tqdm=True)
    scores = {}
    for o, (qi, i) in zip(outs, owners):
        lp = o.outputs[0].logprobs[0]
        t = max([v.logprob for v in lp.values() if v.decoded_token.strip().lower() == "true"], default=-30)
        f = max([v.logprob for v in lp.values() if v.decoded_token.strip().lower() == "false"], default=-30)
        scores.setdefault(qi, []).append((t - f, i))
    return [[i for _, i in sorted(scores[qi], key=lambda x: -x[0])] for qi in range(len(qs))]


def run_ircot(llm, tok, docs, qs, args):
    from vllm import SamplingParams
    etok = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    emod = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5", torch_dtype=torch.float16).cuda().eval()
    D = encode(emod, etok, docs, "cls").cuda()
    Q = encode(emod, etok, [QP + q["question"] for q in qs], "cls").cuda()
    S0 = Q @ D.T
    best = [dict() for _ in qs]                       # doc -> best score over steps
    order = [[] for _ in qs]
    for qi in range(len(qs)):
        sc, ix = S0[qi].topk(args.k_step)
        for s, i in zip(sc.tolist(), ix.tolist()):
            best[qi][i] = max(best[qi].get(i, -1), s); order[qi].append(i)
    cots = ["" for _ in qs]
    done = [False] * len(qs)
    sp = SamplingParams(max_tokens=64, temperature=0.0, stop=["\n"])
    for step in range(args.steps):
        act = [qi for qi in range(len(qs)) if not done[qi]]
        if not act:
            break
        prompts = []
        for qi in act:
            ctx = "\n\n".join(f"Title: {fmt(docs, i)}" for i in order[qi][:15])
            user = (f"{ctx}\n\nQuestion: {qs[qi]['question']}\nAnswer the question step by step, one sentence at a time, "
                    "using the passages. When you know the final answer, write 'So the answer is: <answer>.'\n"
                    f"Reasoning so far: {cots[qi] or '(none)'}\nNext sentence:")
            prompts.append(tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False))
        outs = llm.generate(prompts, sp, use_tqdm=False)
        sents = []
        for qi, o in zip(act, outs):
            s = o.outputs[0].text.strip()
            cots[qi] = (cots[qi] + " " + s).strip()
            if "answer is" in s.lower() or not s:
                done[qi] = True
            sents.append(s)
        E = encode(emod, etok, [QP + s for s in sents], "cls").cuda()
        St = E @ D.T
        for row, qi in enumerate(act):
            if done[qi]:
                continue
            sc, ix = St[row].topk(args.k_step)
            for s, i in zip(sc.tolist(), ix.tolist()):
                best[qi][i] = max(best[qi].get(i, -1), s)
                if i not in order[qi]:
                    order[qi].append(i)
    ranks = [[i for i, _ in sorted(b.items(), key=lambda x: -x[1])] for b in best]
    return ranks, cots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="llmpoint,setr,ircot")
    ap.add_argument("--reader", default="qwen25-7b-awq")
    ap.add_argument("--pool_file", default="")
    ap.add_argument("--n_test", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--k_step", type=int, default=5)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    args = ap.parse_args()
    from vllm import LLM, SamplingParams
    llm = LLM(MODELS[args.reader], quantization="awq_marlin", max_model_len=12288, gpu_memory_utilization=0.80,
              enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    sp_long = SamplingParams(max_tokens=512, temperature=0.0)
    suffix = f"_{args.pool_file}" if args.pool_file else ""
    for ds in args.datasets.split(","):
        docs, qs = load_bench(ds)
        pools = pools_for(ds, args)
        if args.n_test:
            qs, pools = qs[: args.n_test], pools[: args.n_test]
        for m in args.methods.split(","):
            extra = {}
            if m == "setr":
                ranks = run_setr(llm, tok, sp_long, docs, qs, pools)
            elif m == "llmpoint":
                ranks = run_llmpoint(llm, tok, docs, qs, pools)
            else:
                ranks, cots = run_ircot(llm, tok, docs, qs, args); extra = {"cots": cots}
            name = m + (suffix if m != "ircot" else "")
            out = [{"id": q["id"], "rank": r, "top5": r[:5]} for q, r in zip(qs, ranks)]
            jdump(out, RES / "retrieval" / f"{name}_{ds}.json")
            r2 = np.mean([recall_at(r, q["gold"], 2) for r, q in zip(ranks, qs)])
            r5 = np.mean([recall_at(r, q["gold"], 5) for r, q in zip(ranks, qs)])
            print(f"{ds:16s} {name:24s} R@2 {r2:.3f} R@5 {r5:.3f}", flush=True)


if __name__ == "__main__":
    main()
