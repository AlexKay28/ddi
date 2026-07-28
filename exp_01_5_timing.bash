#!/usr/bin/env bash
# exp_01_5_timing.bash — Fragment 5/6 of #33 (issue #38). Timing sanity.
set -euo pipefail

SEEDS=(0 1 2)
MODEL="Qwen/Qwen2.5-7B-Instruct"
MASKS=(full ddi_qf)
NS=(1 5 13)
L=512
Q=64
BACKEND="sdpa"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/logs/exp_01_5_timing"
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

log "== P6. timing =="
for mask in "${MASKS[@]}"; do
  for N in "${NS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      cfg="$CONFIG_DIR/t_${mask}_N${N}_s${seed}.yaml"
      cat > "$cfg" <<YAML
task: phase3_timing
model: $MODEL
N: $N
L: $L
Q: $Q
answer_len: 16
backend: $BACKEND
n_repeats: 5
warmup: 3
n_new_tokens: 16
mask:
  variant: $mask
  bridge_k: null
YAML
      run_cell "$cfg" "$seed" "t_${mask}_N${N}_s${seed}"
    done
  done
done

mkdir -p "$LOG_DIR/reports"
make report NAME=T3 > "$LOG_DIR/reports/T3.md" 2>&1 || log "  T3 empty"

{
  echo "## exp_01_5_timing summary — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "- git SHA: $(git rev-parse HEAD)"
  echo "- store rows: $(wc -l < "$STORE")"
  echo "- OK: $(grep -c '^OK' "$LOG_DIR/cells.tsv"), FAIL: $(grep -c '^FAIL' "$LOG_DIR/cells.tsv")"
  echo
  echo "### T3 (partial — 3 Ns only)"
  cat "$LOG_DIR/reports/T3.md" 2>/dev/null || echo "n/a"
  echo
  echo "### Cells"
  echo '```'
  cat "$LOG_DIR/cells.tsv"
  echo '```'
} > "$LOG_DIR/summary.md"
log "Done."
