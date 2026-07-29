#!/usr/bin/env bash
# exp_01_3_bridge.bash — Fragment 3/6 of #33 (issue #36). Bridge endpoints.
#
# Deviation from the issue's script: the issue specifies BRIDGE_KS=(1 64), with k=64 meant to
# approximate ddi_qf ("k=64 ~ ddi_qf at Q=64"). Measured against composition's actual val split
# under this model's chat-template tokenization, query_len is only 26-28 tokens across all 3
# seeds (val split is 20 records/seed, so MAX_RECORDS=100 below is a no-op cap, same as prior
# fragments). build_mask requires 1 <= bridge_k <= query_len, so k=64 raises ValueError on every
# record. Using k=26 instead: it's the min query_len across all 3 seeds, so it's valid everywhere;
# records with query_len=26 bridge exactly (== ddi_qf), records with query_len 27-28 leave 1-2
# leading query rows strict — a near-exact approximation of the ddi_qf endpoint.
set -euo pipefail

SEEDS=(0 1 2)
MODEL="Qwen/Qwen2.5-7B"   # RESEARCH_PLAN §2.3 pins the bridge sweep to base Qwen
DATASET="composition"
BRIDGE_KS=(1 26)
MAX_RECORDS=100

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/logs/exp_01_3_bridge"
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

log "== P4. Bridge endpoints =="
for k in "${BRIDGE_KS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    cfg="$CONFIG_DIR/bridge_k${k}_s${seed}.yaml"
    cat > "$cfg" <<YAML
task: phase2_train_free
model: $MODEL
dataset: $DATASET
split: val
max_new_tokens: 64
max_records: $MAX_RECORDS
mask:
  variant: ddi_bridge_k
  bridge_k: $k
YAML
    run_cell "$cfg" "$seed" "bridge_k${k}_s${seed}"
  done
done

mkdir -p "$LOG_DIR/reports"
make report NAME=F2 OUT="$LOG_DIR/reports/F2.pdf" >>"$LOG_DIR/reports.log" 2>&1 || log "  F2 render failed"

{
  echo "## exp_01_3_bridge summary — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "- git SHA: $(git rev-parse HEAD)"
  echo "- store rows: $(wc -l < "$STORE")"
  echo "- OK: $(grep -c '^OK' "$LOG_DIR/cells.tsv"), FAIL: $(grep -c '^FAIL' "$LOG_DIR/cells.tsv")"
  echo
  echo "### Cells"
  echo '```'
  cat "$LOG_DIR/cells.tsv"
  echo '```'
  echo
  echo "### F2 render"
  ls -la "$LOG_DIR/reports/F2.pdf" 2>/dev/null || echo "n/a"
} > "$LOG_DIR/summary.md"
log "Done."
