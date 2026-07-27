#!/usr/bin/env bash
# exp_01_2_quality.bash — Fragment 2/6 of #33. Phase 2.1 train-free sweep.
set -euo pipefail

SEEDS=(0)
MODEL="Qwen/Qwen2.5-7B-Instruct"
DATASET="synthetic_qa"
MASKS=(full ddi_qf ddi_strict blind)
MAX_RECORDS=100

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/logs/exp_01_2_quality"
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

log "== P2. Phase 2.1 train-free =="
for mask in "${MASKS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    cfg="$CONFIG_DIR/q_${mask}_s${seed}.yaml"
    cat > "$cfg" <<YAML
task: phase2_train_free
model: $MODEL
dataset: $DATASET
split: val
max_new_tokens: 64
max_records: $MAX_RECORDS
mask:
  variant: $mask
  bridge_k: null
YAML
    run_cell "$cfg" "$seed" "q_${mask}_s${seed}"
  done
done

log "== render T2 =="
mkdir -p "$LOG_DIR/reports"
make report NAME=T2 > "$LOG_DIR/reports/T2.md" 2>&1 || log "  T2 empty"

{
  echo "## exp_01_2_quality summary — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "- git SHA: $(git rev-parse HEAD)"
  echo "- host: $(hostname)"
  echo "- store rows: $(wc -l < "$STORE")"
  echo "- OK: $(grep -c '^OK' "$LOG_DIR/cells.tsv"), FAIL: $(grep -c '^FAIL' "$LOG_DIR/cells.tsv")"
  echo
  echo "### T2 quality table"
  echo
  cat "$LOG_DIR/reports/T2.md" 2>/dev/null || echo "n/a"
  echo
  echo "### Cells"
  echo '```'
  cat "$LOG_DIR/cells.tsv"
  echo '```'
} > "$LOG_DIR/summary.md"
log "Done. Paste $LOG_DIR/summary.md into #33."
