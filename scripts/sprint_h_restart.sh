#!/usr/bin/env bash
# Sprint H (VIBEQUEUE-35): corrected restart-layout control.
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m ddi.run experiments/cache_restart.yaml \
    --model Qwen/Qwen2.5-7B-Instruct \
    --set checkpoint=results/checkpoints/Qwen__Qwen2.5-7B-Instruct__ddi_qf__16L \
    --set 'datasets=[hotpot_qa]' \
    --set max_records=100 \
    --set output_store=results/sprint_h_decode_positions_hotpot.jsonl
