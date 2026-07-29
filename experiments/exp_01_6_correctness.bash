#!/usr/bin/env bash
# exp_01_6_correctness.bash — Fragment 6/6 of #33 (issue #39). Correctness spot-checks.
#
# No deviations from the issue's script.
set -euo pipefail

SEEDS=(0 1 2)
MODEL="Qwen/Qwen2.5-7B-Instruct"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/logs/exp_01_6_correctness"
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

log "== Phase 4.1 restart quality =="
for seed in "${SEEDS[@]}"; do
  cfg="$CONFIG_DIR/restart_s${seed}.yaml"
  cat > "$cfg" <<YAML
task: phase4_restart_quality
model: $MODEL
dataset: composition
split: val
max_new_tokens: 32
max_records: 20
mask:
  variant: ddi_qf
  bridge_k: null
layouts: [sequential, restart]
YAML
  run_cell "$cfg" "$seed" "restart_s${seed}"
done

log "== Phase 4.2 cache correctness =="
cfg="$CONFIG_DIR/cache.yaml"
cat > "$cfg" <<YAML
task: phase4_cache_correctness
model: $MODEL
dataset: composition
mask:
  variant: ddi_qf
  bridge_k: null
layout: restart
max_records: 10
cache_dir: cache/doc_kv
YAML
run_cell "$cfg" 0 "cache"

log "== Phase 4.3 order invariance =="
for layout in sequential restart; do
  for seed in "${SEEDS[@]}"; do
    cfg="$CONFIG_DIR/order_${layout}_s${seed}.yaml"
    cat > "$cfg" <<YAML
task: phase4_order_invariance
model: $MODEL
dataset: composition
mask:
  variant: ddi_qf
  bridge_k: null
layouts: [$layout]
n_permutations: 4
tolerance: 1.0e-3
max_records: 10
YAML
    run_cell "$cfg" "$seed" "order_${layout}_s${seed}"
  done
done

{
  echo "## exp_01_6_correctness summary — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "- git SHA: $(git rev-parse HEAD)"
  echo "- store rows: $(wc -l < "$STORE")"
  echo "- OK: $(grep -c '^OK' "$LOG_DIR/cells.tsv"), FAIL: $(grep -c '^FAIL' "$LOG_DIR/cells.tsv")"
  echo
  echo "### Correctness flags in store"
  echo '```'
  grep -aE '"metric": "(kv_deterministic|kv_cache_roundtrip|invariance|max_delta)"' "$STORE" | tail -30 || echo "n/a"
  echo '```'
  echo
  echo "### Cells"
  echo '```'
  cat "$LOG_DIR/cells.tsv"
  echo '```'
} > "$LOG_DIR/summary.md"
log "Done."
