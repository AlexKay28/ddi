#!/usr/bin/env bash
# exp_01_1_baseline.bash — Fragment 1/6 of #33. Baseline + Δ.
set -euo pipefail

SEEDS=(0 1 2)
MODEL="Qwen/Qwen2.5-7B-Instruct"
DATASET="nq"
MAX_RECORDS=100

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/logs/exp_01_1_baseline"
CONFIG_DIR="$LOG_DIR/configs"
mkdir -p "$LOG_DIR" "$CONFIG_DIR"
STORE="$ROOT/results/runs.jsonl"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG_DIR/master.log" >&2; }

run_cell() {
  local cfg="$1" seed="$2" tag="$3"
  local cell_log="$LOG_DIR/cells/${tag}.log"
  mkdir -p "$(dirname "$cell_log")"
  log "  [cell] $tag"
  if make run CONFIG="$cfg" SEED="$seed" >"$cell_log" 2>&1; then
    echo "OK  $tag" >>"$LOG_DIR/cells.tsv"
  else
    echo "FAIL $tag" >>"$LOG_DIR/cells.tsv"; log "    FAILED — see $cell_log"
  fi
}

log "== preflight =="
make gates >>"$LOG_DIR/gates.log" 2>&1
make gpu-check >>"$LOG_DIR/gpu.log" 2>&1

log "== P1. baseline =="
# Manifests are keyed per (dataset, seed) — results/splits/${DATASET}__seed<N>.json — since
# nq/composition/synthetic_qa all partition train/val/test as a function of `seed` (issue #40).
# No manifest reset needed between seeds.
for seed in "${SEEDS[@]}"; do
  cfg="$CONFIG_DIR/baseline_s${seed}.yaml"
  cat > "$cfg" <<YAML
task: phase1_full_baseline
model: $MODEL
dataset: $DATASET
split: val
max_new_tokens: 64
max_records: $MAX_RECORDS
YAML
  run_cell "$cfg" "$seed" "baseline_s${seed}"
done

log "== P1.5. Δ =="
for metric in em f1 presence loss; do
  log "  Δ $metric"
  make delta MODEL="$MODEL" DATASET="$DATASET" METRIC="$metric" \
    >>"$LOG_DIR/delta.log" 2>&1 || log "    (n<3 seeds; skipped)"
done

{
  echo "## exp_01_1_baseline summary — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "- git SHA: $(git rev-parse HEAD)"
  echo "- host: $(hostname)"
  echo "- store rows: $(wc -l < "$STORE")"
  echo
  echo "### Δ rows"
  echo '```'
  grep -a '"metric": "delta_' "$STORE" | tail -10 || echo "n/a"
  echo '```'
  echo
  echo "### Cells"
  echo '```'
  cat "$LOG_DIR/cells.tsv"
  echo '```'
} > "$LOG_DIR/summary.md"
log "Done. Paste $LOG_DIR/summary.md into #33."
