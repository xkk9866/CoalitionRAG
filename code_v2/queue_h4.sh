#!/bin/bash
# HippoRAG 2 on 2Wiki + HotpotQA with a second LLM server on GPU 4 (parallel to GPU 5's MuSiQue run)
. "$(dirname "$0")/../env.sh"
export CUDA_VISIBLE_DEVICES=4
L=../logs
setsid nohup $PY -m vllm.entrypoints.openai.api_server --model ../models/Qwen2.5-7B-Instruct-AWQ --served-model-name Qwen2.5-7B-Instruct-AWQ \
  --quantization awq_marlin --port 8012 --gpu-memory-utilization 0.6 --max-model-len 8192 > $L/vllm_server4.log 2>&1 < /dev/null &
until curl -s localhost:8012/v1/models >/dev/null; do sleep 10; done
PYTHONPATH=$PYTHONPATH:${CRAG_ROOT}/pylib $PY hippo_run.py --base_url http://localhost:8012/v1 --datasets 2wikimultihopqa,hotpotqa > $L/hippo4.log 2>&1
pkill -f "[p]ort 8012"
echo QH4_DONE >> $L/queue_h4.log
