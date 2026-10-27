#!/bin/bash
# usage: run.sh <gpu> <logname> <script.py> [args...]  -- detached launch
. "$(dirname "$0")/../env.sh"
G=$1; L=$2; shift 2
CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup $PY "$@" > ../logs/$L.log 2>&1 < /dev/null &
echo launched $L on gpu $G
