#!/bin/bash
. "$(dirname "$0")/../env.sh"
export CUDA_VISIBLE_DEVICES=6
L=../logs
until [ -f ../results/retrieval/forest_cl_s2_hotpotqa.json ]; do sleep 60; done
$PY qa.py --runs forest_cl_s2 > $L/qa_g6c.log 2>&1
$PY qa.py --runs forest_cl_s2 --reader qwen3-8b-awq > $L/qa_g6c_q3.log 2>&1
echo QA3_DONE >> $L/g6.out
