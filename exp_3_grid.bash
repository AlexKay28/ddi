#!/usr/bin/env bash
# exp_3_grid.bash — Issue #73 driver for the EXP-3 grid runs (#72).
#
# Generates 138 configs per model (30 quality + 108 performance), runs each cell in
# isolation (per-cell failure is logged as FAIL and the sweep continues), and emits
# a summary.md with the 10 template-shaped renderers concatenated, ready to paste
# into #74 or #75.
#
# Usage: ./exp_3_grid.bash <MODEL_ID>
# Example: ./exp_3_grid.bash Qwen/Qwen2.5-7B-Instruct
#
# fa2 cells may fail on machines without flash-attn — that's expected; the summary
# annotates them as "cell blocked by #70". The (N=25, L=2048) cells push T=51264
# tokens and may OOM on <40GB VRAM; the runner fails gracefully in that case.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <MODEL_ID>" >&2
  echo "example: $0 Qwen/Qwen2.5-7B-Instruct" >&2
  exit 2
fi

MODEL="$1"
SEED=0

# Slugify model id for the log dir (turn slashes and other unsafe chars into underscores).
MODEL_SLUG="$(printf '%s' "$MODEL" | tr '/: ' '___')"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/logs/exp_3_grid_${MODEL_SLUG}"
CONFIG_DIR="$LOG_DIR/configs"
CELL_LOG_DIR="$LOG_DIR/cells"
REPORT_DIR="$LOG_DIR/reports"
mkdir -p "$LOG_DIR" "$CONFIG_DIR" "$CELL_LOG_DIR" "$REPORT_DIR"
STORE="$ROOT/results/runs.jsonl"
CELLS_TSV="$LOG_DIR/cells.tsv"
: > "$CELLS_TSV"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG_DIR/master.log" >&2; }

run_cell() {
  local cfg="$1" tag="$2"
  shift 2
  local cell_log="$CELL_LOG_DIR/${tag}.log"
  log "  [cell] $tag"
  # Any additional env=value pairs are prefixed to `make run` so ddi_stress cells can
  # inject DDI_STRESS_TASK without polluting the outer shell environment.
  if env "$@" make run CONFIG="$cfg" SEED="$SEED" >"$cell_log" 2>&1; then
    echo -e "OK\t${tag}" >>"$CELLS_TSV"
  else
    echo -e "FAIL\t${tag}" >>"$CELLS_TSV"
    log "    FAILED — see $cell_log"
  fi
}

log "== preflight (${MODEL}) =="
make gates >>"$LOG_DIR/gates.log" 2>&1 || log "  gates non-zero — continuing"
make gpu-check >>"$LOG_DIR/gpu.log" 2>&1 || log "  gpu-check non-zero — continuing"

# ---- Quality sweep (30 cells) ---------------------------------------------------------------
# 3 masks x (5 direct datasets + 5 ddi_stress families) = 30.
QUALITY_MASKS=(full ddi_qf blind)
DIRECT_DATASETS=(composition synthetic_qa hotpot_qa musique kilt_nq)
STRESS_FAMILIES=(fact_chain fact_aggregation contradiction coreference needle)
QUALITY_MAX_RECORDS=100

log "== quality (30 cells) =="
for mask in "${QUALITY_MASKS[@]}"; do
  for dataset in "${DIRECT_DATASETS[@]}"; do
    tag="quality_${dataset}_${mask}"
    cfg="$CONFIG_DIR/${tag}.yaml"
    cat > "$cfg" <<YAML
task: phase2_train_free
model: $MODEL
dataset: $dataset
split: val
max_new_tokens: 64
max_records: $QUALITY_MAX_RECORDS
mask:
  variant: $mask
  bridge_k: null
YAML
    run_cell "$cfg" "$tag"
  done

  for family in "${STRESS_FAMILIES[@]}"; do
    tag="quality_ddi_stress_${family}_${mask}"
    cfg="$CONFIG_DIR/${tag}.yaml"
    cat > "$cfg" <<YAML
task: phase2_train_free
model: $MODEL
dataset: ddi_stress
split: val
max_new_tokens: 64
max_records: $QUALITY_MAX_RECORDS
mask:
  variant: $mask
  bridge_k: null
YAML
    run_cell "$cfg" "$tag" "DDI_STRESS_TASK=$family"
  done
done

# ---- Performance sweep (108 cells) ----------------------------------------------------------
# 3 masks x 6 N x 3 L x 2 backends = 108.
PERF_MASKS=(full ddi_qf blind)
PERF_NS=(1 3 5 9 13 25)
PERF_LS=(128 512 2048)
PERF_BACKENDS=(sdpa fa2)
PERF_Q=64

log "== performance (108 cells) =="
for mask in "${PERF_MASKS[@]}"; do
  for N in "${PERF_NS[@]}"; do
    for L in "${PERF_LS[@]}"; do
      for backend in "${PERF_BACKENDS[@]}"; do
        tag="perf_${mask}_N${N}_L${L}_${backend}"
        cfg="$CONFIG_DIR/${tag}.yaml"
        cat > "$cfg" <<YAML
task: phase3_timing
model: $MODEL
mask:
  variant: $mask
  bridge_k: null
N: $N
L: $L
Q: $PERF_Q
answer_len: 16
backend: $backend
n_repeats: 5
warmup: 3
n_new_tokens: 16
YAML
        run_cell "$cfg" "$tag"
      done
    done
  done
done

# ---- Render summary -------------------------------------------------------------------------
log "== render summary =="
# Names must match the @register decorators in ddi/reports/{phase2_quality,phase3_realization}.py.
REPORTS=(
  T2_em T2_f1 T2_presence T2_loss
  T_perf_ttft T_perf_tpot T_perf_sparsity
  T_perf_analytical T_perf_realized T_perf_realization
)
for name in "${REPORTS[@]}"; do
  out="$REPORT_DIR/${name}.md"
  if make report NAME="$name" >"$out" 2>>"$LOG_DIR/reports.log"; then
    log "  [render] $name"
  else
    log "  [render] $name — empty or failed (see reports.log)"
    echo "_(no data for $name in store)_" >"$out"
  fi
done

OK_COUNT=$(grep -c '^OK' "$CELLS_TSV" 2>/dev/null || echo 0)
FAIL_COUNT=$(grep -c '^FAIL' "$CELLS_TSV" 2>/dev/null || echo 0)
FA2_FAILS=$(grep -c '^FAIL.*_fa2$' "$CELLS_TSV" 2>/dev/null || echo 0)

{
  echo "## exp_3_grid summary — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "- model: \`$MODEL\`"
  echo "- git SHA: $(git rev-parse HEAD 2>/dev/null || echo n/a)"
  echo "- host: $(hostname)"
  echo "- store rows: $(wc -l < "$STORE" 2>/dev/null || echo n/a)"
  echo "- OK: ${OK_COUNT}, FAIL: ${FAIL_COUNT} (of which fa2: ${FA2_FAILS} — likely blocked by #70)"
  echo
  echo "### Quality tables (T2 per metric — rows=mask, cols=10 datasets)"
  echo
  for name in T2_em T2_f1 T2_presence T2_loss; do
    echo "#### $name"
    echo
    cat "$REPORT_DIR/${name}.md"
    echo
  done
  echo "### Performance tables (T_perf_* — rows=mask, cols=N, one block per L x backend)"
  echo
  for name in T_perf_ttft T_perf_tpot T_perf_sparsity T_perf_analytical T_perf_realized T_perf_realization; do
    echo "#### $name"
    echo
    cat "$REPORT_DIR/${name}.md"
    echo
  done
  echo "### Cells"
  echo '```'
  cat "$CELLS_TSV"
  echo '```'
} > "$LOG_DIR/summary.md"

log "Done. Paste $LOG_DIR/summary.md into #74 or #75."
