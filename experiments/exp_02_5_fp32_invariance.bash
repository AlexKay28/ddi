#!/usr/bin/env bash
# exp_02_5_fp32_invariance.bash — Fragment 5/6 of #58 (issue #63). Closes #57.
#
# EXP-01.6 (#39) found ddi_qf+restart has ~11x smaller max_delta than sequential under doc-order
# permutation, but the binary invariance@atol=1e-3 check failed on every record. Suspected cause:
# bf16 precision floor at ~1e-2 across 28 decoder layers, not a real position/mask leak. This
# fragment reruns a handful of records in fp32 to disambiguate: if restart's max_delta collapses
# toward zero while sequential stays large, bf16 was the confound and ddi_qf+restart is truly
# order-invariant. If restart's max_delta stays finite under fp32, it's a real leak.
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL="Qwen/Qwen2.5-7B-Instruct"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/logs/exp_02_5_fp32_invariance"
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

log "== fp32 order-invariance =="
for layout in sequential restart; do
  cfg="$CONFIG_DIR/order_fp32_${layout}.yaml"
  cat > "$cfg" <<YAML
task: phase4_order_invariance
model: $MODEL
dtype: fp32
dataset: composition
mask:
  variant: ddi_qf
  bridge_k: null
layouts: [$layout]
n_permutations: 4
tolerance: 1.0e-5
max_records: 10
YAML
  run_cell "$cfg" 0 "order_fp32_${layout}"
done

{
  SHA8="$(git rev-parse --short=8 HEAD)"
  echo "## exp_02_5_fp32_invariance summary — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "- git SHA: $(git rev-parse HEAD)"
  # run_id doesn't encode dtype directly — this run's 8-char git SHA is what distinguishes
  # these fp32 rows from EXP-01.6's bf16 rows for the same model/dataset/variant/layout/seed.
  echo "- rows added: $(grep -ac "\"run_id\": \"phase4_order:.*:$SHA8:" "$STORE" 2>/dev/null || echo 0)"
  echo
  echo "### fp32 invariance rows (this run, SHA=$SHA8)"
  echo '```'
  grep -aE "\"run_id\": \"phase4_order:.*:$SHA8:.*\"metric\": \"(invariance|max_delta)\"" "$STORE" | tail -40 || echo "n/a"
  echo '```'
  echo
  echo "### Comparison with bf16 rows from EXP-01.6"
  echo "Grep the store for phase4_order rows and compare max_delta distributions."
  echo
  echo "### Cells"
  echo '```'
  cat "$LOG_DIR/cells.tsv"
  echo '```'
} > "$LOG_DIR/summary.md"
log "Done."
