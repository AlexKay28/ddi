#!/usr/bin/env bash
# exp_02_6_throughput.bash — Fragment 6/6 of #58 (issue #64). QPS vs hit-rate, F5.
#
# Review-pass fixes applied from #64:
# 1. `dataset: nq` was misleading (the runner's synthetic doc universe never consumes
#    it) — set to `synthetic`, matching the runner's own default (config.get("dataset",
#    "synthetic")).
# 2. `cache_dir` is parameterised per seed (cache/doc_kv_s${seed}) and wiped before each
#    ddi_cache run. compute_key() hashes doc content, and doc content is seed-derived, so
#    cross-seed key collisions were already vanishingly unlikely — but starting each seed
#    from an empty cache removes any doubt about the hit_rate distribution being confused
#    by leftover state from a prior invocation of this script.
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SEEDS=(0 1 2)
MODEL="Qwen/Qwen2.5-7B-Instruct"
MODES=(ddi_cache ddi_recompute full_causal)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/logs/exp_02_6_throughput"
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

log "== throughput 3 modes x 3 seeds (9 cells; ~200 req each) =="
for mode in "${MODES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    cache_dir="cache/doc_kv_s${seed}"
    if [[ "$mode" == "ddi_cache" ]]; then
      rm -rf "$ROOT/$cache_dir"
    fi
    cfg="$CONFIG_DIR/tp_${mode}_s${seed}.yaml"
    cat > "$cfg" <<YAML
task: phase4_throughput
model: $MODEL
mask:
  variant: ddi_qf
  bridge_k: null
layout: restart
dataset: synthetic
n_requests: 200
reuse_distribution: zipf
zipf_a: 1.5
top_docs: 32
docs_per_request: 5
warmup_requests: 20
mode: $mode
cache_dir: $cache_dir
YAML
    run_cell "$cfg" "$seed" "tp_${mode}_s${seed}"
  done
done

mkdir -p "$LOG_DIR/reports"
make report NAME=F5 OUT="$LOG_DIR/reports/F5.pdf" 2>&1 || log "  F5 render failed"

{
  echo "## exp_02_6_throughput summary — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "- git SHA: $(git rev-parse HEAD)"
  echo "- OK: $(grep -c '^OK' "$LOG_DIR/cells.tsv"), FAIL: $(grep -c '^FAIL' "$LOG_DIR/cells.tsv")"
  echo
  echo "### qps + hit_rate rows"
  echo '```'
  grep -aE '"metric": "(qps|hit_rate)"' "$STORE" | tail -30 || echo "n/a"
  echo '```'
  echo
  echo "### F5 render"
  ls -la "$LOG_DIR/reports/F5.pdf" 2>/dev/null || echo "n/a"
} > "$LOG_DIR/summary.md"
log "Done."
