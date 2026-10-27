#!/bin/bash
. "$(dirname "$0")/../env.sh"
export CUDA_VISIBLE_DEVICES=6
L=../logs
$PY qa.py --runs hipporag2 --datasets musique,2wikimultihopqa > $L/qa_g6b.log 2>&1
$PY qa.py --runs hipporag2 --datasets musique,2wikimultihopqa --reader qwen3-8b-awq > $L/qa_g6b_q3.log 2>&1
echo QA2_DONE >> $L/g6.out
