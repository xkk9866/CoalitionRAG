#!/bin/bash
# When HippoRAG 2 (3 datasets) and seed 2 are done: stop the leftover vLLM server and run the readers on GPU 1.
. "$(dirname "$0")/../env.sh"
export CUDA_VISIBLE_DEVICES=1
L=../logs; R=../results/retrieval
until [ -f $R/hipporag2_hotpotqa.json ] && [ -f $R/hipporag2_2wikimultihopqa.json ]; do sleep 120; done
pkill -f "[p]ort 8021"; sleep 30
until [ "$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits)" -gt 21500 ]; do sleep 60; done
$PY qa.py --runs hipporag2 --datasets 2wikimultihopqa,hotpotqa > $L/qa_g1a.log 2>&1
$PY qa.py --runs hipporag2 --datasets 2wikimultihopqa,hotpotqa --reader qwen3-8b-awq > $L/qa_g1b.log 2>&1
until [ -f $R/hipporag2_musique.json ] && [ -f $R/forest_cl_s2_hotpotqa.json ]; do sleep 120; done
until [ "$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits)" -gt 21500 ]; do sleep 60; done
$PY qa.py --runs hipporag2 --datasets musique > $L/qa_g1c.log 2>&1
$PY qa.py --runs hipporag2 --datasets musique --reader qwen3-8b-awq > $L/qa_g1d.log 2>&1
$PY qa.py --runs forest_cl_s2 > $L/qa_g1e.log 2>&1
$PY qa.py --runs forest_cl_s2 --reader qwen3-8b-awq > $L/qa_g1f.log 2>&1
echo FINAL_DONE >> $L/g1final.out
