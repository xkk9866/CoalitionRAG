#!/bin/bash
# Training-seed replication of the evidence forest, GPU 1 only. Skips finished steps; safe to restart.
. "$(dirname "$0")/../env.sh"
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=../logs; R=../results
for s in 1 2; do
  [ -f $R/ckpt/forest_s$s.pt ] && [ -f $R/retrieval/forest_s${s}_hotpotqa.json ] || $PY forest.py --name forest_s$s --seed $s >> $L/forest_s$s.log 2>&1
  [ -f $R/retrieval/forest_cl_s${s}_hotpotqa.json ] || $PY closure.py --name forest_cl_s$s --eval_only forest_s$s --base 100 --grow 8 >> $L/forest_cl_s$s.log 2>&1
done
echo SEEDS_DONE >> $L/gpu1_seeds.out
