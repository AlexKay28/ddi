#!/bin/bash
# =====================================================================
# paper/build.sh — compile paper/preprint.pdf.
#
# Tables in paper/generated/*.tex are currently hand-edited from local result
# stores and the tracked aggregate files cited in each table's header comment.
# The automated renderer was removed; the former paper/render.py was retired
# because it consumed missing summaries and emitted incompatible table shapes.
# Do NOT treat generated/*.tex as auto-generated.
#
# VIBEQUEUE-50 tracks preservation of raw sources and reproducible rendering.
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"

echo "[paper] Tables are hand-audited (see VIBEQUEUE-50). Skipping regeneration."
echo "[paper]   T2         — results/runs.jsonl (phase2_train_free rows)"
echo "[paper]   T3         — results/runs.jsonl (phase3_timing rows)"
echo "[paper]   T3_flex    — logs/timing_flex_grid3.log"
echo "[paper]   T4         — results/runs.jsonl (phase2_train_free rows)"
echo "[paper]   T4_16L     — results/eval_16L.jsonl"
echo "[paper]   T4_restart — results/sprint_h_decode_positions_hotpot.jsonl"
echo "[paper]   T5         — results/phase3_n{2,5,10}.jsonl"
echo "[paper]   T6         — results/sprint_a_baseline.jsonl"
echo "[paper]   T7         — experiments/cross_architecture_transfer_r5.result.json"
echo "[paper]   F5         — results/cache_v51_throughput_summary.json"

if command -v pdflatex >/dev/null 2>&1; then
    pdflatex -interaction=nonstopmode preprint.tex
    (bibtex preprint || true)
    pdflatex -interaction=nonstopmode preprint.tex
    pdflatex -interaction=nonstopmode preprint.tex
    echo "[paper] preprint.pdf built."
else
    echo "[paper] pdflatex not found; skipping compile. Generated artifacts are in paper/generated/."
fi
