#!/usr/bin/env bash
set -euo pipefail
SEEDS=(0)
BASE_MODEL="Qwen/Qwen2.5-7B"
DATASETS=(hotpot_qa musique)
MASKS=(full ddi_qf ddi_strict blind)
MAX_RECORDS=100

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/logs/exp_02_2_base_model"
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

log "== preflight ==" && make gates >>"$LOG_DIR/gates.log" 2>&1 && make gpu-check >>"$LOG_DIR/gpu.log" 2>&1

log "== step 1: base-model baseline =="
for dataset in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    cfg="$CONFIG_DIR/base_${dataset}_s${seed}.yaml"
    cat > "$cfg" <<YAML
task: phase1_full_baseline
model: $BASE_MODEL
dataset: $dataset
split: val
max_new_tokens: 64
max_records: $MAX_RECORDS
YAML
    run_cell "$cfg" "$seed" "base_${dataset}_s${seed}"
  done
done

log "== step 2: base-model Δ =="
for dataset in "${DATASETS[@]}"; do
  for metric in em f1 presence loss; do
    make delta MODEL="$BASE_MODEL" DATASET="$dataset" METRIC="$metric" \
      >>"$LOG_DIR/delta.log" 2>&1 || log "    (skipped: $dataset $metric)"
  done
done

log "== step 3: base-model quality sweep =="
for dataset in "${DATASETS[@]}"; do
  for mask in "${MASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      cfg="$CONFIG_DIR/q_${dataset}_${mask}_s${seed}.yaml"
      cat > "$cfg" <<YAML
task: phase2_train_free
model: $BASE_MODEL
dataset: $dataset
split: val
max_new_tokens: 64
max_records: $MAX_RECORDS
mask:
  variant: $mask
  bridge_k: null
YAML
      run_cell "$cfg" "$seed" "q_${dataset}_${mask}_s${seed}"
    done
  done
done

mkdir -p "$LOG_DIR/reports"
make report NAME=T2 > "$LOG_DIR/reports/T2.md" 2>&1 || log "  T2 empty"

{
  echo "## exp_02_2_base_model summary — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "- git SHA: $(git rev-parse HEAD)"
  echo "- store rows: $(wc -l < "$STORE")"
  echo "- OK: $(grep -c '^OK' "$LOG_DIR/cells.tsv"), FAIL: $(grep -c '^FAIL' "$LOG_DIR/cells.tsv")"
  echo
  echo "### base-model Δ"
  echo '```'
  grep -a '"metric": "delta_' "$STORE" | grep -a "$BASE_MODEL" | tail -20 || echo "n/a"
  echo '```'
  echo
  echo "### T2 (base-model rows)"
  cat "$LOG_DIR/reports/T2.md" 2>/dev/null || echo "n/a"
} > "$LOG_DIR/summary.md"
log "Done."
