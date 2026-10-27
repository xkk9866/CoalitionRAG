#!/bin/bash
# After the main queue: HippoRAG 2 on MuSiQue with a fresh OpenIE file (the old one fails the provenance check); the LLM cache is kept.
. "$(dirname "$0")/../env.sh"
export CUDA_VISIBLE_DEVICES=1
L=../logs; R=../results; S=$R/hipporag2_store/musique
until [ -f $L/gpu1_queue.out ]; do sleep 120; done
until [ "$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits)" -gt 14000 ]; do sleep 60; done
mkdir -p $S/../old_openie_musique; mv $S/openie_results_ner_*.json* $S/../old_openie_musique/ 2>/dev/null
setsid nohup $PY -m vllm.entrypoints.openai.api_server --model ../models/Qwen2.5-7B-Instruct-AWQ --served-model-name Qwen2.5-7B-Instruct-AWQ \
  --quantization awq_marlin --port 8021 --gpu-memory-utilization 0.5 --max-model-len 8192 > $L/vllm_server_g1m.log 2>&1 < /dev/null &
for t in $(seq 1 90); do curl -s localhost:8021/v1/models >/dev/null && break; sleep 10; done
PYTHONPATH=$PYTHONPATH:${CRAG_ROOT}/pylib $PY hippo_run.py --base_url http://localhost:8021/v1 --datasets musique >> $L/hippo_g1m.log 2>&1
pkill -f "[p]ort 8021"; echo MUSIQUE_DONE >> $L/gpu1_hippo_musique.out
