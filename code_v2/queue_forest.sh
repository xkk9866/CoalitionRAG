#!/bin/bash
# Full evidence-forest experiments, sequential on GPU 4 only. Trainings resume from their last checkpoint if killed.
. "$(dirname "$0")/../env.sh"
export CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=../logs; R=../results
step() { echo "$(date '+%F %T') $*" >> $L/queue_forest.log; }
waitgpu() { until [ "$(nvidia-smi -i 4 --query-gpu=memory.free --format=csv,noheader,nounits)" -gt 20000 ]; do sleep 60; done; }
retry() {  # retry <file that marks completion> <command...>
  local done=$1; shift
  for a in 1 2 3 4 5 6; do
    [ -f "$done" ] && return 0
    waitgpu; step "  attempt $a: $*"; "$@"
  done
}

step start main
retry $R/retrieval/forest_summary.json $PY forest.py --name forest >> $L/forest.log 2>&1
step start bce-control
retry $R/retrieval/point_bce_summary.json $PY forest.py --name point_bce --no_reasons >> $L/point_bce.log 2>&1
step start ablations
$PY forest_ablate.py forest exp20_s5_k8 --save unary,onestep,beam,cover > $L/forest_ablate.log 2>&1
$PY forest_ablate.py f_cdn exp20_s5_k8 --cdn --methods forest,onestep --save forest > $L/cdn_forest_ablate.log 2>&1
step start qa
waitgpu
$PY qa.py --runs forest,point_bce,forest-unary,forest-onestep,forest-beam,forest-cover,f_cdn-forest > $L/qa_forest.log 2>&1
$PY qa.py --runs forest,point_bce --reader qwen3-8b-awq > $L/qa_forest3.log 2>&1
step start hipporag2
waitgpu
bash hippo4.sh >> $L/queue_forest.log 2>&1
$PY qa.py --runs hipporag2 > $L/qa_hippo.log 2>&1
$PY qa.py --runs hipporag2 --reader qwen3-8b-awq > $L/qa_hippo3.log 2>&1
step start pool-sensitivity
retry $R/retrieval/forest_base20_summary.json $PY forest.py --eval_only forest --name forest_base20 --pool_file base20 --n_test 300 >> $L/forest_base20.log 2>&1
step start structure
$PY forest_structure.py forest f_cdn > $L/forest_structure.log 2>&1
step start seed1
retry $R/retrieval/forest_s1_summary.json $PY forest.py --name forest_s1 --seed 1 --n_test 300 >> $L/forest_s1.log 2>&1
step start transfer
retry $R/retrieval/forest_nomus_summary.json $PY forest.py --name forest_nomus --train train_open_nomus.jsonl --datasets musique --n_test 300 >> $L/forest_nomus.log 2>&1
step QUEUE_FOREST_DONE
