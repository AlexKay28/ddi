#!/bin/bash
# =====================================================================
# paper/build.sh — compile paper/preprint.pdf.
#
# Tables in paper/generated/*.tex are currently hand-edited from
# results/runs.jsonl and the logs/results files cited in each table's
# header comment. The automated renderer (ddi.report_cli) was removed
# in #89 and never replaced. Do NOT treat generated/*.tex as
# auto-generated — verify numbers against the cited source files.
#
# Sprint B (VIBEQUEUE-26) will restore the renderer or document the
# hand-editing workflow permanently.
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"

echo "[paper] Tables are hand-edited (see VIBEQUEUE-26). Skipping regeneration."
echo "[paper]   T2         — results/runs.jsonl (phase2_train_free rows)"
echo "[paper]   T3         — results/runs.jsonl (phase3_timing rows)"
echo "[paper]   T3_flex    — logs/timing_flex_grid3.log"
echo "[paper]   T4         — results/runs.jsonl (phase2_train_free rows)"
echo "[paper]   T4_16L     — results/eval_16L.jsonl"
echo "[paper]   T4_restart — results/runs.jsonl (phase4_restart rows)"
echo "[paper]   T5         — results/phase3_n{2,5,10}.jsonl"

if command -v pdflatex >/dev/null 2>&1; then
    pdflatex -interaction=nonstopmode preprint.tex
    (bibtex preprint || true)
    pdflatex -interaction=nonstopmode preprint.tex
    pdflatex -interaction=nonstopmode preprint.tex
    echo "[paper] preprint.pdf built."
else
    echo "[paper] pdflatex not found; skipping compile. Generated artifacts are in paper/generated/."
fi
