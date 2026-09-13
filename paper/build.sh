#!/bin/bash
# =====================================================================
# paper/build.sh — validate/render paper artifacts, then compile when TeX exists.
#
# paper/evidence.json is the tracked evidence boundary. paper/render.py validates
# its schema and source hashes, then deterministically regenerates every input.
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"

python3 render.py

if command -v pdflatex >/dev/null 2>&1; then
    if ! command -v bibtex >/dev/null 2>&1; then
        echo "[paper] pdflatex is present but bibtex is missing." >&2
        exit 1
    fi
    # Keep TeX's clock and PDF metadata stable. Override only when intentionally
    # dating a new manuscript snapshot.
    export TZ=UTC
    export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1788725843}"
    export FORCE_SOURCE_DATE=1
    pdflatex -halt-on-error -interaction=nonstopmode preprint.tex
    bibtex preprint
    pdflatex -halt-on-error -interaction=nonstopmode preprint.tex
    pdflatex -halt-on-error -interaction=nonstopmode preprint.tex
    echo "[paper] preprint.pdf built."
else
    echo "[paper] pdflatex not found; artifacts rendered and validated, PDF compile skipped."
fi
