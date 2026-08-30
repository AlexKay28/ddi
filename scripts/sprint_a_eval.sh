#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL="Qwen/Qwen2.5-7B-Instruct"
CKPT_16L="results/checkpoints/Qwen__Qwen2.5-7B-Instruct__ddi_qf__16L"
YAML="experiments/quality.yaml"

echo "============================================"
echo "SPRINT A — $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "============================================"

# ============================================================
# A-1: 3-seed baseline eval (full mask, seeds 0/1/2)
#       → gives us Δ = k·σ from seed variance
# ============================================================
echo ""
echo "=== A-1: 3-seed baseline (full, 100 records) ==="
for S in 0 1 2; do
    echo "--- seed $S ---"
    python -m ddi.run "$YAML" \
        --model "$MODEL" \
        --seed $S \
        --set "datasets=[hotpot_qa, musique]" \
        --set "masks=[full]" \
        --set "max_records=100" \
        --set "output_store=results/sprint_a_baseline.jsonl"
done

# ============================================================
# A-2: 500-record eval on 16L checkpoint (full + ddi_qf)
#       → tighter gap estimate vs Δ
# ============================================================
echo ""
echo "=== A-2: 500-record 16L eval (full + ddi_qf) ==="
python -m ddi.run "$YAML" \
    --model "$MODEL" \
    --set "checkpoint=$CKPT_16L" \
    --set "datasets=[hotpot_qa, musique]" \
    --set "masks=[full, ddi_qf]" \
    --set "max_records=500" \
    --set "output_store=results/sprint_a_16L_500.jsonl"

# ============================================================
# A-3: ddi_strict ablation
#       base model (train-free) + 16L checkpoint
# ============================================================
echo ""
echo "=== A-3a: ddi_strict on base model (100 records) ==="
python -m ddi.run "$YAML" \
    --model "$MODEL" \
    --set "datasets=[hotpot_qa, musique]" \
    --set "masks=[ddi_strict]" \
    --set "max_records=100" \
    --set "output_store=results/sprint_a_strict_base.jsonl"

echo ""
echo "=== A-3b: ddi_strict on 16L checkpoint (100 records) ==="
python -m ddi.run "$YAML" \
    --model "$MODEL" \
    --set "checkpoint=$CKPT_16L" \
    --set "datasets=[hotpot_qa, musique]" \
    --set "masks=[ddi_strict]" \
    --set "max_records=100" \
    --set "output_store=results/sprint_a_strict_16L.jsonl"

echo ""
echo "============================================"
echo "SPRINT A complete — $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "============================================"
echo "Stores:"
echo "  results/sprint_a_baseline.jsonl    (3-seed full baseline)"
echo "  results/sprint_a_16L_500.jsonl     (16L, 500 records)"
echo "  results/sprint_a_strict_base.jsonl (ddi_strict, base)"
echo "  results/sprint_a_strict_16L.jsonl  (ddi_strict, 16L)"
