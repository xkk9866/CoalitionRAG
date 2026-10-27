"""Run official HippoRAG 2 (repos/HippoRAG) with the shared LLM (vLLM OpenAI server) and bge-large embeddings,
and save its passage ranking for evaluation with the common reader.

Requires a vLLM OpenAI-compatible server at --base_url serving --llm_name.
"""
import argparse
import os
import sys

from common import DATASETS, RES, ROOT, jdump, load_bench, recall_at

sys.path.insert(0, str(ROOT / "repos" / "HippoRAG"))
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", default="http://localhost:8011/v1")
    ap.add_argument("--llm_name", default="Qwen2.5-7B-Instruct-AWQ")
    ap.add_argument("--emb", default="Transformers/BAAI/bge-large-en-v1.5")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--n_test", type=int, default=0)
    args = ap.parse_args()
    from src.hipporag.HippoRAG import HippoRAG
    from src.hipporag.utils.config_utils import BaseConfig
    import numpy as np

    for ds in args.datasets.split(","):
        docs, qs = load_bench(ds)
        if args.n_test:
            qs = qs[: args.n_test]
        index = {d: i for i, d in enumerate(docs)}
        cfg = BaseConfig(
            save_dir=str(RES / "hipporag2_store" / ds), llm_base_url=args.base_url, llm_name=args.llm_name,
            dataset=ds, embedding_model_name=args.emb, retrieval_top_k=200, linking_top_k=5, qa_top_k=5,
            embedding_batch_size=64, openie_mode="online", max_new_tokens=None, force_index_from_scratch=True,
            rerank_dspy_file_path=str(ROOT / "repos/HippoRAG/src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json"),
        )
        rag = HippoRAG(global_config=cfg)
        rag.index(docs)
        sols = rag.retrieve(queries=[q["question"] for q in qs], num_to_retrieve=10)
        out, r2, r5 = [], [], []
        for q, s in zip(qs, sols):
            ids = [index[d] for d in s.docs if d in index]
            out.append({"id": q["id"], "rank": ids, "top5": ids[:5]})
            r2.append(recall_at(ids, q["gold"], 2)); r5.append(recall_at(ids, q["gold"], 5))
        jdump(out, RES / "retrieval" / f"hipporag2_{ds}.json")
        print(f"{ds} HippoRAG2 R@2 {np.mean(r2):.3f} R@5 {np.mean(r5):.3f}", flush=True)


if __name__ == "__main__":
    main()
