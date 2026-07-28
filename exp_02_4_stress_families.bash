#!/usr/bin/env bash
# exp_02_4_stress_families.bash — Fragment 4/6 of #58 (issue #62).
#
# Runs ddi_stress's other four task families (fact_aggregation, contradiction, coreference,
# needle) across all 4 masks x 1 seed. EXP-01 only covered fact_chain. Depends on the
# run_id family-tagging fix (commit a3025e0, already on main): phase2_train_free appends
# DDI_STRESS_TASK to the dataset segment as `ddi_stress__<family>` so T2/F4 slice by family
# automatically.
#
# 1 seed, not 3 (explicit call, not the default for quality-metric fragments): normally
# quality/accuracy sweeps keep 3 seeds since seed variance is the only variance control for
# sampling noise in which synthetic records get generated. Cut to 1 here to get a first look
# across all 4 families at 1/3 the cost; rerun with seeds 1-2 later if any family's result is
# surprising or borderline and needs the full seed spread to trust.
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SEEDS=(0)
MODEL="Qwen/Qwen2.5-7B-Instruct"
FAMILIES=(fact_aggregation contradiction coreference needle)
MASKS=(full ddi_qf ddi_strict blind)
MAX_RECORDS=100

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/logs/exp_02_4_stress_families"
CONFIG_DIR="$LOG_DIR/configs"
mkdir -p "$LOG_DIR" "$CONFIG_DIR"
STORE="$ROOT/results/runs.jsonl"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG_DIR/master.log" >&2; }
run_cell() {
  local cfg="$1" seed="$2" tag="$3" family="$4"
  local cell_log="$LOG_DIR/cells/${tag}.log"
  mkdir -p "$(dirname "$cell_log")"
  log "  [cell] $tag (DDI_STRESS_TASK=$family)"
  if DDI_STRESS_TASK="$family" make run CONFIG="$cfg" SEED="$seed" >"$cell_log" 2>&1; then
    echo "OK  $tag" >>"$LOG_DIR/cells.tsv"
  else
    echo "FAIL $tag" >>"$LOG_DIR/cells.tsv"; log "    FAILED — see $cell_log"
  fi
}

log "== preflight ==" && make gates >>"$LOG_DIR/gates.log" 2>&1 && make gpu-check >>"$LOG_DIR/gpu.log" 2>&1

log "== ddi_stress families (16 cells: 4 families x 4 masks, seed 0) =="
for family in "${FAMILIES[@]}"; do
  for mask in "${MASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      cfg="$CONFIG_DIR/s_${family}_${mask}_s${seed}.yaml"
      cat > "$cfg" <<YAML
task: phase2_train_free
model: $MODEL
dataset: ddi_stress
split: val
max_new_tokens: 64
max_records: $MAX_RECORDS
mask:
  variant: $mask
  bridge_k: null
YAML
      run_cell "$cfg" "$seed" "s_${family}_${mask}_s${seed}" "$family"
    done
  done
done

mkdir -p "$LOG_DIR/reports"
make report NAME=T2 > "$LOG_DIR/reports/T2.md" 2>&1 || log "  T2 empty"

{
  echo "## exp_02_4_stress_families summary — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "- git SHA: $(git rev-parse HEAD)"
  echo "- OK: $(grep -c '^OK' "$LOG_DIR/cells.tsv"), FAIL: $(grep -c '^FAIL' "$LOG_DIR/cells.tsv")"
  echo
  echo "### T2 (ddi_stress rows across families)"
  cat "$LOG_DIR/reports/T2.md" 2>/dev/null || echo "n/a"
} > "$LOG_DIR/summary.md"
log "Done."
