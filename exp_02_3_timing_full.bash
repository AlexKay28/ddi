#!/usr/bin/env bash
# exp_02_3_timing_full.bash — Fragment 3/6 of #58 (issue #61). Full realization-ratio sweep, settles C3.
#
# Review-pass fixes applied from #61:
# 1. Loop nesting reordered so `backend` is outermost — the runner reloads the
#    model on every `attn_implementation` change, so this cuts reloads from 216
#    (one per innermost iteration) down to 2 (one per backend).
# 2. (N=25, L=2048) is skipped outright: T = 25*2048+64 = 51264 tokens, which
#    OOMs on <40GB VRAM. All other cells (up to N=13/L=2048 = 26688 tokens) fit
#    comfortably on 40GB and are safe on 24GB.
set -euo pipefail

SEEDS=(0 1 2)
MODEL="Qwen/Qwen2.5-7B-Instruct"
BACKENDS=(sdpa fa2)
MASKS=(full ddi_qf)
NS=(1 3 5 9 13 25)
LS=(128 512 2048)
Q=64

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/logs/exp_02_3_timing_full"
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

log "== timing sweep (204 cells; 12 skipped: N=25,L=2048 OOM) =="
for backend in "${BACKENDS[@]}"; do
  for mask in "${MASKS[@]}"; do
    for N in "${NS[@]}"; do
      for L in "${LS[@]}"; do
        if [[ "$N" == "25" && "$L" == "2048" ]]; then
          for seed in "${SEEDS[@]}"; do
            echo "SKIP t_${mask}_N${N}_L${L}_${backend}_s${seed} (expected OOM: T=$((N*L+Q)) tokens)" >>"$LOG_DIR/cells.tsv"
          done
          log "  [skip] N=$N L=$L ${backend} ${mask} — expected OOM, T=$((N*L+Q)) tokens"
          continue
        fi
        for seed in "${SEEDS[@]}"; do
          cfg="$CONFIG_DIR/t_${mask}_N${N}_L${L}_${backend}_s${seed}.yaml"
          cat > "$cfg" <<YAML
task: phase3_timing
model: $MODEL
N: $N
L: $L
Q: $Q
answer_len: 16
backend: $backend
n_repeats: 5
warmup: 3
n_new_tokens: 16
mask:
  variant: $mask
  bridge_k: null
YAML
          run_cell "$cfg" "$seed" "t_${mask}_N${N}_L${L}_${backend}_s${seed}"
        done
      done
    done
  done
done

mkdir -p "$LOG_DIR/reports"
make report NAME=T3 > "$LOG_DIR/reports/T3.md" 2>&1 || log "  T3 empty"
make report NAME=F3 OUT="$LOG_DIR/reports/F3.pdf" 2>&1 || log "  F3 render failed"

{
  echo "## exp_02_3_timing_full summary — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "- git SHA: $(git rev-parse HEAD)"
  echo "- OK: $(grep -c '^OK' "$LOG_DIR/cells.tsv"), FAIL: $(grep -c '^FAIL' "$LOG_DIR/cells.tsv"), SKIP: $(grep -c '^SKIP' "$LOG_DIR/cells.tsv")"
  echo
  echo "### T3"
  cat "$LOG_DIR/reports/T3.md" 2>/dev/null || echo "n/a"
} > "$LOG_DIR/summary.md"
log "Done."
