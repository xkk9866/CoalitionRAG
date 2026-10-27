#!/bin/bash
# Remaining experiments, strictly sequential on GPU 1: seed 1, HippoRAG 2 (official code, Qwen2.5-7B-AWQ), seed 2, reader runs.
. "$(dirname "$0")/../env.sh"
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=../logs; R=../results
waitgpu() { until [ "$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits)" -gt $1 ]; do sleep 60; done; }
seed() { s=$1
  waitgpu 20000
  [ -f $R/ckpt/forest_s$s.pt ] && [ -f $R/retrieval/forest_s${s}_hotpotqa.json ] || $PY forest.py --name forest_s$s --seed $s >> $L/forest_s$s.log 2>&1
  [ -f $R/retrieval/forest_cl_s${s}_hotpotqa.json ] || $PY closure.py --name forest_cl_s$s --eval_only forest_s$s --base 100 --grow 8 >> $L/forest_cl_s$s.log 2>&1
}
seed 1
# HippoRAG 2: fresh index directory; the cached OpenIE extraction of MuSiQue is kept
S=$R/hipporag2_store/musique
[ -d $S/Qwen2.5-7B-Instruct-AWQ_Transformers_BAAI_bge-large-en-v1.5 ] && mv $S/Qwen2.5-7B-Instruct-AWQ_Transformers_BAAI_bge-large-en-v1.5 $S/../old_index_musique
if [ ! -f $R/retrieval/hipporag2_hotpotqa.json ]; then
  waitgpu 14000
  setsid nohup $PY -m vllm.entrypoints.openai.api_server --model ../models/Qwen2.5-7B-Instruct-AWQ --served-model-name Qwen2.5-7B-Instruct-AWQ \
    --quantization awq_marlin --port 8021 --gpu-memory-utilization 0.5 --max-model-len 8192 > $L/vllm_server_g1.log 2>&1 < /dev/null &
  for t in $(seq 1 90); do curl -s localhost:8021/v1/models >/dev/null && break; sleep 10; done
  for ds in musique 2wikimultihopqa hotpotqa; do
    [ -f $R/retrieval/hipporag2_$ds.json ] || PYTHONPATH=$PYTHONPATH:${CRAG_ROOT}/pylib $PY hippo_run.py --base_url http://localhost:8021/v1 --datasets $ds >> $L/hippo_g1.log 2>&1
  done
  pkill -f "[p]ort 8021"; sleep 30
fi
seed 2
waitgpu 21500; $PY qa.py --runs hipporag2,forest_cl_s1,forest_cl_s2 > $L/qa_g1.log 2>&1
waitgpu 21500; $PY qa.py --runs hipporag2,forest_cl_s1,forest_cl_s2 --reader qwen3-8b-awq > $L/qa_g1_q3.log 2>&1
echo QUEUE_DONE >> $L/gpu1_queue.out
