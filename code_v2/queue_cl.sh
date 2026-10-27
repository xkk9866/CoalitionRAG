#!/bin/bash
# Evidence forests with closure inference: every experiment of the paper, sequential on GPU 4 only.
# Each step is skipped if its output exists; a step whose command is still running (e.g. from an earlier copy of
# this queue) is waited for, not started twice. Safe to restart at any time.
. "$(dirname "$0")/../env.sh"
export CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=../logs; R=../results; RT=$R/retrieval
step() { echo "$(date '+%F %T') $*" >> $L/queue_cl.log; }
waitgpu() { until [ "$(nvidia-smi -i 4 --query-gpu=memory.free --format=csv,noheader,nounits)" -gt 20000 ]; do sleep 60; done; }
once() {  # once <file that marks completion> <command...>
  local done=$1; shift
  for a in 1 2 3; do
    while pgrep -f -- "$*" > /dev/null; do sleep 60; done
    [ -f "$done" ] && return 0
    waitgpu; [ -f "$done" ] && return 0
    step "  attempt $a: $*"; "$@"
  done
}
# cl <tag> <dataset file suffix> <closure args...>: closure run, done when the retrieval file of the last dataset exists
cl() { local tag=$1 last=$2; shift 2; once $RT/${tag}_$last.json $PY closure.py --name $tag "$@" >> $L/$tag.log 2>&1; }

step start
until [ -f $R/ckpt/forest.pt ]; do sleep 60; done
pkill -f "forest.py --name forest$"

step main: closure + recruitment from first-stage top-100
cl forest_cl hotpotqa --eval_only forest --base 100 --grow 8
step closure on the fixed candidate pool of the controls
cl forest_clx hotpotqa --eval_only forest --pool_file exp20_s5_k8
step exact n^2 forest on the fixed pool, 300 questions per dataset
once $RT/forest_full_hotpotqa.json $PY forest.py --eval_only forest --name forest_full --n_test 300 >> $L/forest_full.log 2>&1

step pointwise control: same sub-pools, schedule and code, no reasons
once $RT/point_bce_hotpotqa.json $PY forest.py --name point_bce --no_reasons >> $L/point_bce.log 2>&1
once $RT/point_bce_u100_hotpotqa.json $PY forest.py --eval_only point_bce --name point_bce_u100 --pool_file u100 --no_reasons >> $L/point_bce.log 2>&1

step inference ablations, structure and cost on cached logits
[ -f $RT/forest_full_ablate_summary.json ] || $PY forest_ablate.py forest_full exp20_s5_k8 --save unary,onestep,beam,cover > $L/forest_ablate.log 2>&1
[ -f $R/tables/forest_full_structure.json ] || $PY forest_structure.py forest_full f_cdn > $L/forest_structure.log 2>&1
[ -f $R/tables/forest_full_cost.json ] || $PY forest_cost.py forest_full exp20_s5_k8 > $L/forest_cost.log 2>&1

step qa
once $R/qa/qwen25-7b-awq_point_bce_u100_hotpotqa.json $PY qa.py --runs forest_cl,forest_clx,point_bce,point_bce_u100 > $L/qa_cl.log 2>&1
once $R/qa/qwen3-8b-awq_point_bce_u100_hotpotqa.json $PY qa.py --runs forest_cl,forest_clx,point_bce_u100 --reader qwen3-8b-awq > $L/qa_cl3.log 2>&1

step out-of-domain: MultiHop-RAG, no training on it
(
export CRAG_DATASETS=multihoprag; M=multihoprag
[ -f $R/pools/u100_$M.json ] || $PY union_pool.py exp20_s5_k8 100 u100 >> $L/mh.log 2>&1
once $RT/f_cdn_$M.json $PY cdn.py --name f_cdn --eval_only p_cdn >> $L/mh.log 2>&1
[ -f $RT/f_cdn-cover_$M.json ] || $PY select_games.py f_cdn exp20_s5_k8 >> $L/mh.log 2>&1
once $RT/zs_rerank_exp_$M.json $PY forest.py --eval_only zero --name zs_rerank_exp --no_reasons >> $L/mh.log 2>&1
once $RT/point_bce_u100_$M.json $PY forest.py --eval_only point_bce --name point_bce_u100 --pool_file u100 --no_reasons >> $L/mh.log 2>&1
cl forest_clx $M --eval_only forest --pool_file exp20_s5_k8
cl forest_cl $M --eval_only forest --base 100 --grow 8
cl zero_cl $M --eval_only zero --base 100 --grow 8
once $RT/ircot_$M.json $PY llm_baselines.py --methods setr,ircot >> $L/mh.log 2>&1
once $R/qa/qwen25-7b-awq_forest_cl_$M.json $PY qa.py --runs none,first:bm25,first:bge,zs_rerank_exp,point_bce_u100,f_cdn-cover,setr,ircot,forest_clx,zero_cl,forest_cl > $L/qa_mh.log 2>&1
)

step "zero-shot cross-encoder in the same closure procedure (no multi-hop supervision)"
cl zero_cl hotpotqa --eval_only zero --base 100 --grow 8
step closure on first-stage top-100 without recruitment
cl forest_cl100 hotpotqa --eval_only forest --base 100
once $R/qa/qwen25-7b-awq_forest_cl100_hotpotqa.json $PY qa.py --runs zero_cl,forest_cl100 > $L/qa_zero.log 2>&1

step seeds 1 and 2
for s in 1 2; do
  once $R/ckpt/forest_s$s.pt $PY forest.py --name forest_s$s --seed $s --n_test 1 >> $L/forest_s$s.log 2>&1
  cl forest_s${s}_cl hotpotqa --eval_only forest_s$s --base 100 --grow 8
done
once $R/qa/qwen25-7b-awq_forest_s2_cl_hotpotqa.json $PY qa.py --runs forest_s1_cl,forest_s2_cl > $L/qa_seeds.log 2>&1

step hipporag2 with the same LLM and embedder
waitgpu
[ -f $RT/hipporag2_hotpotqa.json ] || bash hippo4.sh >> $L/queue_cl.log 2>&1
[ -f $RT/hipporag2_multihoprag.json ] || bash hippo4.sh --datasets multihoprag >> $L/queue_cl.log 2>&1
$PY qa.py --runs hipporag2 > $L/qa_hippo.log 2>&1
CRAG_DATASETS=multihoprag $PY qa.py --runs hipporag2 >> $L/qa_hippo.log 2>&1
$PY qa.py --runs hipporag2,zero_cl --reader qwen3-8b-awq > $L/qa_hippo3.log 2>&1

step transfer: train without MuSiQue, test on MuSiQue
once $R/ckpt/forest_nomus.pt $PY forest.py --name forest_nomus --train train_open_nomus.jsonl --datasets musique --n_test 1 >> $L/forest_nomus.log 2>&1
cl forest_nomus_cl musique --eval_only forest_nomus --base 100 --grow 8 --datasets musique
step QUEUE_CL_DONE
