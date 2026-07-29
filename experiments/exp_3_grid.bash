#!/usr/bin/env bash
# exp_3_grid.bash — thin wrapper that drives the C1 + C3 grid for both models.
#
# Replaces the 138-cell inline heredoc driver from #73; the actual sweep shape now
# lives in the four YAMLs below (schema per issue #81). This script just orchestrates
# `python -m ddi.run` per YAML and then `python -m ddi.aggregate` per YAML into a
# per-experiment summary in logs/.
#
# Usage: ./experiments/exp_3_grid.bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

YAMLS=(
  qwen_instruct_c1
  qwen_instruct_c3
  llama_instruct_c1
  llama_instruct_c3
)

for yaml in "${YAMLS[@]}"; do
  python -m ddi.run       "experiments/${yaml}.yaml"
  python -m ddi.aggregate "experiments/${yaml}.yaml" --out "logs/${yaml}_summary.md"
done
