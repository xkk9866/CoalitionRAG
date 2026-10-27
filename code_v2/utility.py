"""Reader-grounded coalition utilities u(S) = mean log P_reader(answer | q, S) for sampled coalitions S.

One prefill pass per coalition with the frozen int4 reader (vLLM prompt_logprobs). Output rows:
{qid, coalitions: [[pool idx...]], u: [float]}.
"""
import argparse
import random
import zlib

from common import DATA, MODELS, jl_dump, jl_load, reader_messages


def sample_coalitions(n, gold, rng, k_total=20):
    gold = list(gold); neg = [i for i in range(n) if i not in gold]
    cs = [tuple(), tuple(gold)]
    for g in gold:
        cs.append((g,))
        if len(gold) > 1:
            cs.append(tuple(x for x in gold if x != g))
    for _ in range(2):
        cs.append(tuple(gold + rng.sample(neg, min(len(neg), rng.randint(1, max(1, 5 - len(gold)))))))
        if len(gold) > 1:
            drop = rng.choice(gold)
            keep = [x for x in gold if x != drop]
            cs.append(tuple(keep + rng.sample(neg, min(len(neg), rng.randint(1, 3)))))
    for _ in range(2):
        cs.append((rng.choice(neg),))
    while len(set(map(lambda c: tuple(sorted(c)), cs))) < k_total:
        size = rng.randint(2, 5)
        ng = rng.randint(0, min(2, len(gold)))
        c = rng.sample(gold, ng) + rng.sample(neg, min(len(neg), size - ng))
        cs.append(tuple(c))
    seen, out = set(), []
    for c in cs:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key); out.append(sorted(c))
    return out[:k_total]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train_4000.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--reader", default="qwen25-7b-awq")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    rows = jl_load(DATA / "train" / args.train)
    if args.limit:
        rows = rows[:args.limit]
    rows = rows[args.shard::args.nshards]
    llm = LLM(MODELS[args.reader], quantization="awq_marlin", max_model_len=4096, gpu_memory_utilization=0.70, max_num_batched_tokens=2048, enable_chunked_prefill=True,
              enable_prefix_caching=True, max_num_seqs=32)
    tok = llm.get_tokenizer()

    prompts, meta = [], []
    for qi, r in enumerate(rows):
        rng = random.Random(zlib.crc32(r["id"].encode()))
        ans_ids = tok(r["answers"][0].strip(), add_special_tokens=False)["input_ids"]
        for ci, c in enumerate(sample_coalitions(len(r["pool"]), r["gold"], rng, args.k)):
            msgs = reader_messages(r["question"], r["pool"], c)
            p_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
            prompts.append({"prompt_token_ids": p_ids + ans_ids})
            meta.append((qi, c, len(p_ids), len(ans_ids)))

    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
    outs = llm.generate(prompts, sp, use_tqdm=True)
    res = {}
    for o, (qi, c, plen, alen) in zip(outs, meta):
        lps = o.prompt_logprobs
        toks = o.prompt_token_ids
        vals = [lps[plen + j][toks[plen + j]].logprob for j in range(alen)]
        d = res.setdefault(qi, {"qid": rows[qi]["id"], "coalitions": [], "u": []})
        d["coalitions"].append(c); d["u"].append(sum(vals) / max(1, alen))
    out = args.out or f"util_{args.reader}_{args.train.replace('.jsonl', '')}_s{args.shard}.jsonl"
    jl_dump([res[i] for i in sorted(res)], DATA / "train" / out)
    print("done", len(res))


if __name__ == "__main__":
    main()
