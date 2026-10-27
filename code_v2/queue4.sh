#!/bin/bash
# sequential GPU-4 queue
. "$(dirname "$0")/../env.sh"
export CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=../logs
while pgrep -f "[n]ame f_cdn --eval_only p_cdn --datasets musique" >/dev/null; do sleep 30; done
$PY -c "
from common import *
for ds in DATASETS:
    r=jload(RES/'first_stage'/f'bge_{ds}.json')['ranked']; jdump([x[:20] for x in r], RES/'pools'/f'base20_{ds}.json')"
$PY cdn.py --name f_cdn --eval_only p_cdn --datasets 2wikimultihopqa > $L/f_cdn_2wiki.log 2>&1
$PY cdn.py --name a_zero --eval_only zero --n_test 300 > $L/a_zero.log 2>&1
$PY cdn.py --name a_base20 --eval_only p_cdn --pool_file base20 --n_test 300 > $L/a_base20.log 2>&1
$PY - <<'P' > $L/mk_transfer.log 2>&1
from common import *
R=jl_load(DATA/'train'/'train_open.jsonl'); jl_dump([r for r in R if r['ds']!='musique'], DATA/'train'/'train_open_nomus.jsonl')
P
$PY cdn.py --name t_nomus --train train_open_nomus.jsonl --datasets musique --n_test 300 > $L/t_nomus.log 2>&1
echo QUEUE4_DONE >> $L/queue4.log
