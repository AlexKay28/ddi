#!/usr/bin/env bash
# exp_01_4_composition.bash — Fragment 4/6 of #33 (issue #37). Composition probe.
#
# No deviations from the issue's script. composition's val split is 20 records/seed (same as
# #35/#36), so MAX_RECORDS=100 below is a no-op cap, consistent with prior fragments.
set -euo pipefail

SEEDS=(0 1 2)
MODEL="Qwen/Qwen2.5-7B-Instruct"
VARIANTS=(full ddi_qf)
LAYOUTS=(sequential restart)
MAX_RECORDS=100

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/logs/exp_01_4_composition"
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

log "== P5.3 composition =="
for v in "${VARIANTS[@]}"; do
  for l in "${LAYOUTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      cfg="$CONFIG_DIR/comp_${v}_${l}_s${seed}.yaml"
      cat > "$cfg" <<YAML
task: phase5_composition
model: $MODEL
dataset: composition
split: val
max_new_tokens: 32
max_records: $MAX_RECORDS
mask:
  variant: $v
  bridge_k: null
layout: $l
YAML
      run_cell "$cfg" "$seed" "comp_${v}_${l}_s${seed}"
    done
  done
done

mkdir -p "$LOG_DIR/reports"
make report NAME=T6 > "$LOG_DIR/reports/T6.md" 2>&1 || log "  T6 empty"
make report NAME=F7 OUT="$LOG_DIR/reports/F7.pdf" >>"$LOG_DIR/reports.log" 2>&1 || log "  F7 render failed"

{
  echo "## exp_01_4_composition summary — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "- git SHA: $(git rev-parse HEAD)"
  echo "- store rows: $(wc -l < "$STORE")"
  echo "- OK: $(grep -c '^OK' "$LOG_DIR/cells.tsv"), FAIL: $(grep -c '^FAIL' "$LOG_DIR/cells.tsv")"
  echo
  echo "### T6"
  cat "$LOG_DIR/reports/T6.md" 2>/dev/null || echo "n/a"
  echo
  echo "### F7 render"
  ls -la "$LOG_DIR/reports/F7.pdf" 2>/dev/null || echo "n/a"
  echo
  echo "### Cells"
  echo '```'
  cat "$LOG_DIR/cells.tsv"
  echo '```'
} > "$LOG_DIR/summary.md"
log "Done."
