"""Reader evaluation: given retrieval outputs, generate answers with a frozen int4 reader and score EM/F1.

--runs tag1,tag2,... reads results/retrieval/{tag}_{ds}.json (list of {id, top5|rank}) ; "none" = no retrieval;
"first:<retriever>" uses first-stage top-5.
"""
import argparse

import numpy as np

from common import DATASETS, MODELS, RES, clean_pred, em_f1, jdump, jload, load_bench, reader_messages


def get_ids(tag, ds, n):
    if tag == "none":
        return [[] for _ in range(n)]
    if tag.startswith("first:"):
        r = jload(RES / "first_stage" / f"{tag[6:]}_{ds}.json")["ranked"]
        return [x[:5] for x in r[:n]]
    rows = jload(RES / "retrieval" / f"{tag}_{ds}.json")
    return [(x.get("top5") or x["rank"][:5]) for x in rows[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--reader", default="qwen25-7b-awq")
    ap.add_argument("--n_test", type=int, default=0)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    args = ap.parse_args()
    from vllm import LLM, SamplingParams
    llm = LLM(MODELS[args.reader], quantization="awq_marlin", max_model_len=6144, gpu_memory_utilization=0.85,
              enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    extra = {"enable_thinking": False} if "qwen3" in args.reader else {}
    sp = SamplingParams(max_tokens=32, temperature=0.0)
    for ds in args.datasets.split(","):
        docs, qs = load_bench(ds)
        n = args.n_test or len(qs)
        qs = qs[:n]
        for tag in args.runs.split(","):
            try:
                idl = get_ids(tag, ds, n)
            except FileNotFoundError:
                print("missing", tag, ds); continue
            prompts = [tok.apply_chat_template(reader_messages(q["question"], docs, ids), add_generation_prompt=True,
                                               tokenize=False, **extra) for q, ids in zip(qs, idl)]
            outs = llm.generate(prompts, sp, use_tqdm=False)
            preds = [clean_pred(o.outputs[0].text) for o in outs]
            sc = [em_f1(p, q["answers"]) for p, q in zip(preds, qs)]
            res = {"EM": float(np.mean([s[0] for s in sc])), "F1": float(np.mean([s[1] for s in sc])), "n": n}
            safe = tag.replace(":", "-")
            jdump({"summary": res, "preds": preds, "em_f1": sc}, RES / "qa" / f"{args.reader}_{safe}_{ds}.json")
            print(f"{ds:16s} {tag:30s} EM {res['EM']:.3f} F1 {res['F1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
