#!/bin/bash
# HippoRAG 2 on MuSiQue (fresh OpenIE; LLM cache kept) on GPU 4.
. "$(dirname "$0")/../env.sh"
export CUDA_VISIBLE_DEVICES=4
L=../logs; R=../results; S=$R/hipporag2_store/musique
mkdir -p $S/../old_openie_musique; mv $S/openie_results_ner_*.json* $S/../old_openie_musique/ 2>/dev/null
[ -d $S/Qwen2.5-7B-Instruct-AWQ_Transformers_BAAI_bge-large-en-v1.5 ] && mv $S/Qwen2.5-7B-Instruct-AWQ_Transformers_BAAI_bge-large-en-v1.5 $S/../old_index_musique2
setsid nohup $PY -m vllm.entrypoints.openai.api_server --model ../models/Qwen2.5-7B-Instruct-AWQ --served-model-name Qwen2.5-7B-Instruct-AWQ \
  --quantization awq_marlin --port 8024 --gpu-memory-utilization 0.6 --max-model-len 8192 > $L/vllm_server_g4.log 2>&1 < /dev/null &
for t in $(seq 1 90); do curl -s localhost:8024/v1/models >/dev/null && break; sleep 10; done
PYTHONPATH=$PYTHONPATH:${CRAG_ROOT}/pylib $PY hippo_run.py --base_url http://localhost:8024/v1 --datasets musique >> $L/hippo_g4.log 2>&1
pkill -f "[p]ort 8024"; echo HM_DONE >> $L/g4.out
