#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="results/checkpoints/Qwen__Qwen2.5-7B-Instruct__ddi_qf__8L"
LOG="logs/phase3_doc_sweep2.log"

echo "=== Phase 3 document-count sweep v2 (separate stores) ===" | tee "$LOG"

for N in 2 5 10; do
    echo "--- N=$N ---" | tee -a "$LOG"
    STORE="results/phase3_n${N}.jsonl"
    rm -f "$STORE"
    DDI_HOTPOT_N_PARAGRAPHS=$N \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python -m ddi.run experiments/quality.yaml \
        --model Qwen/Qwen2.5-7B-Instruct \
        --set "datasets=[hotpot_qa]" \
        --set "masks=[full,ddi_qf]" \
        --set "checkpoint=$CKPT" \
        --set "output_store=$STORE" \
        2>&1 | tee -a "$LOG"
    echo "" | tee -a "$LOG"
done

echo "=== Aggregating ===" | tee -a "$LOG"
for N in 2 5 10; do
    STORE="results/phase3_n${N}.jsonl"
    echo "--- N=$N ---" | tee -a "$LOG"
    python -c "
import json
from collections import defaultdict
vals = defaultdict(list)
with open('$STORE') as f:
    for line in f:
        r = json.loads(line)
        rid = r['run_id']
        parts = rid.split(':')
        mask = parts[3]
        vals[(mask, r['metric'])].append(r['value'])
for (mask, metric), vlist in sorted(vals.items()):
    avg = sum(vlist) / len(vlist)
    print(f'  mask={mask:8s}  {metric:10s}  mean={avg:.4f}  n={len(vlist)}')
" 2>&1 | tee -a "$LOG"
done

echo "=== Done ===" | tee -a "$LOG"
