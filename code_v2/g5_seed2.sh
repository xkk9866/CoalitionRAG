#!/bin/bash
# Training seed 2 of the evidence forest on GPU 5. The fixed-pool evaluation that forest.py runs after training is skipped (killed once the checkpoint exists).
. "$(dirname "$0")/../env.sh"
export CUDA_VISIBLE_DEVICES=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=../logs; R=../results
if [ ! -f $R/ckpt/forest_s2.pt ]; then
  $PY forest.py --name forest_s2 --seed 2 >> $L/forest_s2.log 2>&1 &
  P=$!
  until [ -f $R/ckpt/forest_s2.pt ]; do sleep 30; kill -0 $P 2>/dev/null || break; done
  sleep 20; kill $P 2>/dev/null; wait $P 2>/dev/null
fi
$PY closure.py --name forest_cl_s2 --eval_only forest_s2 --base 100 --grow 8 >> $L/forest_cl_s2.log 2>&1
echo S2_DONE >> $L/g5.out
