#!/bin/bash
. "$(dirname "$0")/../env.sh"
export CUDA_VISIBLE_DEVICES=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=../logs
while pgrep -f "[l]lm_baselines.py" >/dev/null; do sleep 30; done
$PY llm_baselines.py --methods setr,ircot --datasets 2wikimultihopqa,hotpotqa > $L/llm_base2.log 2>&1
$PY llm_baselines.py --methods llmpoint --datasets hotpotqa > $L/llm_base3.log 2>&1
while [ ! -f ../results/cache/f_cdn_2wikimultihopqa.pt ] || [ ! -f ../results/cache/f_cdn_musique.pt ]; do sleep 60; done
$PY select_games.py f_cdn exp20_s5_k8 > $L/f_select.log 2>&1
RUNS=none,first:bm25,first:contriever,first:bge,zs_rerank_exp,o_point_gold,llmpoint,setr,ircot,f_cdn-add_logit,f_cdn-add_prob,f_cdn-cover
$PY qa.py --runs $RUNS > $L/qa_qwen25.log 2>&1
$PY qa.py --runs none,first:bge,o_point_gold,setr,ircot,f_cdn-cover --reader qwen3-8b-awq > $L/qa_qwen3.log 2>&1
# HippoRAG 2 with the shared LLM behind a local OpenAI-compatible server
setsid nohup $PY -m vllm.entrypoints.openai.api_server --model ../models/Qwen2.5-7B-Instruct-AWQ --served-model-name Qwen2.5-7B-Instruct-AWQ \
  --quantization awq_marlin --port 8011 --gpu-memory-utilization 0.6 --max-model-len 8192 > $L/vllm_server.log 2>&1 < /dev/null &
until curl -s localhost:8011/v1/models >/dev/null; do sleep 10; done
PYTHONPATH=$PYTHONPATH:${CRAG_ROOT}/pylib $PY hippo_run.py > $L/hippo.log 2>&1
pkill -f "[a]pi_server --model ../models/Qwen2.5"
sleep 20
$PY qa.py --runs hipporag2 > $L/qa_hippo.log 2>&1
echo QUEUE5_DONE >> $L/queue5.log
