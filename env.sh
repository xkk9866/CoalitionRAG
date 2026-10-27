#!/usr/bin/env bash
# Source this file from a Bash launcher, or from an interactive shell.
export CRAG_ROOT="${CRAG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export PY="${PYTHON:-python}"
mkdir -p "$CRAG_ROOT/results" "$CRAG_ROOT/logs" "$CRAG_ROOT/models"
cd "$CRAG_ROOT/code_v2" || return 1
# Keep machine-specific settings in this ignored file.
if [ -f "$CRAG_ROOT/env.local.sh" ]; then
    . "$CRAG_ROOT/env.local.sh"
fi
