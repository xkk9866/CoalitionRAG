#!/bin/bash
# Reader runs for training seed 1 on GPU 6.
. "$(dirname "$0")/../env.sh"
export CUDA_VISIBLE_DEVICES=6
L=../logs
$PY qa.py --runs forest_cl_s1 > $L/qa_g6.log 2>&1
$PY qa.py --runs forest_cl_s1 --reader qwen3-8b-awq > $L/qa_g6_q3.log 2>&1
echo QA1_DONE >> $L/g6.out
