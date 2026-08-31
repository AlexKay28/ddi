#!/usr/bin/env bash
# Sprint D (VIBEQUEUE-28): rope_theta override on restart layout.
#
# Tests whether ABF-style rope_theta adjustment reduces the restart-layout
# quality gap. The 16L finetuned checkpoint is evaluated under both sequential
# and restart layouts with different rope_theta values.
#
# Qwen2.5-7B default rope_theta = 1,000,000.
#
# Usage: bash scripts/sprint_d_restart.sh
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL="Qwen/Qwen2.5-7B-Instruct"
CKPT="results/checkpoints/Qwen__Qwen2.5-7B-Instruct__ddi_qf__16L"
CONFIG="experiments/cache_restart.yaml"
NREC=100

run() {
    local theta="$1"
    local store="$2"
    local extra=""
    if [ -n "$theta" ]; then
        extra="--set rope_theta=${theta}"
    fi
    echo "=== rope_theta=${theta:-default} ==="
    python -m ddi.run "$CONFIG" \
        --model "$MODEL" \
        --set "checkpoint=${CKPT}" \
        --set "max_records=${NREC}" \
        --set "output_store=${store}" \
        $extra
}

# Default (no override) — baseline
run ""        "results/sprint_d_restart_default.jsonl"

# ABF variants
run 100000    "results/sprint_d_restart_theta100k.jsonl"
run 500000    "results/sprint_d_restart_theta500k.jsonl"
run 10000000  "results/sprint_d_restart_theta10M.jsonl"

echo "=== All Sprint D runs complete ==="
