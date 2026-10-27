#!/bin/bash
. "$(dirname "$0")/../env.sh"
export CUDA_VISIBLE_DEVICES=5
L=../logs
rm -rf ../results/hipporag2_store
setsid nohup $PY -m vllm.entrypoints.openai.api_server --model ../models/Qwen2.5-7B-Instruct-AWQ --served-model-name Qwen2.5-7B-Instruct-AWQ \
  --quantization awq_marlin --port 8011 --gpu-memory-utilization 0.6 --max-model-len 8192 > $L/vllm_server.log 2>&1 < /dev/null &
until curl -s localhost:8011/v1/models >/dev/null; do sleep 10; done
PYTHONPATH=$PYTHONPATH:${CRAG_ROOT}/pylib $PY hippo_run.py > $L/hippo.log 2>&1
pkill -f "[a]pi_server --model ../models/Qwen2.5"; sleep 30
$PY qa.py --runs hipporag2 > $L/qa_hippo.log 2>&1
$PY qa.py --runs hipporag2 --reader qwen3-8b-awq > $L/qa_hippo3.log 2>&1
echo QH_DONE >> $L/queue_h.log
