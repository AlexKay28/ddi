#!/usr/bin/env bash
# exp_02_1_multihop_n3.bash — Fragment 1/6 of #58.
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SEEDS=(0)
MODEL="Qwen/Qwen2.5-7B-Instruct"
DATASETS=(hotpot_qa musique ddi_stress kilt_nq)
MASKS=(full ddi_qf ddi_strict blind)
MAX_RECORDS=100

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/logs/exp_02_1_multihop_n3"
CONFIG_DIR="$LOG_DIR/configs"
mkdir -p "$LOG_DIR" "$CONFIG_DIR"
STORE="$ROOT/results/runs.jsonl"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG_DIR/master.log" >&2; }
slug() { echo "$1" | tr '/:' '__' | tr -c 'a-zA-Z0-9_.-' '_' | tr -s '_'; }
run_cell() {
  local cfg="$1" seed="$2" tag="$3"
  local cell_log="$LOG_DIR/cells/${tag}.log"
  if grep -qx "OK  $tag" "$LOG_DIR/cells.tsv" 2>/dev/null; then
    log "  [cell] $tag (skip, already OK)"
    return
  fi
  mkdir -p "$(dirname "$cell_log")"
  log "  [cell] $tag"
  if make run CONFIG="$cfg" SEED="$seed" >"$cell_log" 2>&1; then
    echo "OK  $tag" >>"$LOG_DIR/cells.tsv"
  else
    echo "FAIL $tag" >>"$LOG_DIR/cells.tsv"; log "    FAILED — see $cell_log"
  fi
}

log "== preflight ==" && make gates >>"$LOG_DIR/gates.log" 2>&1 && make gpu-check >>"$LOG_DIR/gpu.log" 2>&1

log "== multi-hop quality n=3 =="
for dataset in "${DATASETS[@]}"; do
  for mask in "${MASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      cfg="$CONFIG_DIR/q_${dataset}_${mask}_s${seed}.yaml"
      cat > "$cfg" <<YAML
task: phase2_train_free
model: $MODEL
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
  echo "## exp_02_1_multihop_n3 summary — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "- git SHA: $(git rev-parse HEAD)"
  echo "- store rows: $(wc -l < "$STORE")"
  echo "- OK: $(grep -c '^OK' "$LOG_DIR/cells.tsv"), FAIL: $(grep -c '^FAIL' "$LOG_DIR/cells.tsv")"
  echo
  echo "### T2 (multi-hop rows)"
  cat "$LOG_DIR/reports/T2.md" 2>/dev/null || echo "n/a"
  echo
  echo "### Cells"
  echo '```'
  cat "$LOG_DIR/cells.tsv"
  echo '```'
} > "$LOG_DIR/summary.md"
log "Done. Paste $LOG_DIR/summary.md into #58."
