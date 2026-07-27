#!/bin/bash
# =====================================================================
# paper/build.sh — regenerate every table and figure from the results
# store, then compile paper/preprint.pdf.
#
# Ground rule 3 (paper/RESEARCH_PLAN.md, issue #1): every empirical number
# in the preprint comes from ddi/reports/ via the results store. This
# script is the only path from the store to the compiled PDF; nothing in
# paper/sections/ types a number by hand.
#
# When a renderer has no rows to work with (empty store, or a renderer
# that has not been implemented yet), we emit a \todoreport{...} stub in
# place of the generated .tex / .pdf so the paper still compiles and the
# resulting PDF advertises exactly which sweep unblocks each cell.
#
# Invoked as `make paper` from the repo root; see the top-level Makefile.
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"

REPORT_CMD=(uv run python -m ddi.report_cli)

# pandoc is required to convert the renderer's markdown output into .tex.
# When it is missing we still emit \todoreport stubs so the paper compiles
# (build.sh is useful on containers without pandoc; the operator is told
# what is missing rather than the script hard-failing).
have_pandoc=1
if ! command -v pandoc >/dev/null 2>&1; then
    have_pandoc=0
    echo "[paper] pandoc not found — every table will be stubbed."
fi

emit_table_stub() {
    local name="$1"
    local tex_out="$2"
    local reason="$3"
    cat > "${tex_out}" <<EOF
\\todoreport{${name} unavailable — ${reason}}{make paper (after the relevant sweep lands)}
EOF
}

# ---------------------------------------------------------------------
# Tables — markdown from ddi.report, converted to .tex via pandoc.
# Every table that appears in sections/*.tex must be listed here.
# ---------------------------------------------------------------------
tables=(T2 T3 T4_probe_instruct T5 T6)
for name in "${tables[@]}"; do
    md_out="generated/${name}.md"
    tex_out="generated/${name}.tex"
    if "${REPORT_CMD[@]}" --name "${name}" > "${md_out}" 2>/dev/null; then
        if [ "${have_pandoc}" -eq 1 ] && pandoc "${md_out}" -o "${tex_out}"; then
            echo "[paper] rendered ${name}"
        else
            emit_table_stub "${name}" "${tex_out}" "pandoc conversion unavailable"
            echo "[paper] ${name} rendered as markdown but pandoc unavailable — stubbed"
        fi
    else
        # Empty-store fallback: overwrite any stale markdown so a later
        # successful render is unambiguous, and emit a \todoreport stub.
        rm -f "${md_out}"
        emit_table_stub "${name}" "${tex_out}" "no rows in the results store yet"
        echo "[paper] ${name} unavailable — stubbed"
    fi
done

# T1 is a static markdown source (paper/generated/T1_mask_taxonomy.md is
# checked in). Convert on every build so the .tex is always fresh.
if [ -f generated/T1_mask_taxonomy.md ]; then
    if [ "${have_pandoc}" -eq 1 ] && pandoc generated/T1_mask_taxonomy.md -o generated/T1_mask_taxonomy.tex; then
        echo "[paper] rendered T1 (static)"
    else
        emit_table_stub "T1" "generated/T1_mask_taxonomy.tex" \
            "pandoc unavailable to convert the static markdown source"
        echo "[paper] T1 unavailable — stubbed"
    fi
fi

# ---------------------------------------------------------------------
# Figures — pdf from ddi.report. On failure we delete any stale file so
# the \IfFileExists in the .tex falls through to a \todoreport stub.
# ---------------------------------------------------------------------
figures=(F2 F3 F4 F5 F6 F7)
for name in "${figures[@]}"; do
    pdf_out="generated/${name}.pdf"
    if "${REPORT_CMD[@]}" --name "${name}" --output "${pdf_out}" 2>/dev/null; then
        echo "[paper] rendered ${name}"
    else
        rm -f "${pdf_out}"
        echo "[paper] ${name} unavailable — figure hidden (TODO stub in PDF)"
    fi
done

# ---------------------------------------------------------------------
# Compile the paper. In containers without a TeX toolchain we still want
# the regeneration step to succeed, so we exit 0 with a note instead.
# ---------------------------------------------------------------------
if command -v pdflatex >/dev/null 2>&1; then
    pdflatex -interaction=nonstopmode preprint.tex
    (bibtex preprint || true)
    pdflatex -interaction=nonstopmode preprint.tex
    pdflatex -interaction=nonstopmode preprint.tex
    echo "[paper] preprint.pdf built."
else
    echo "[paper] pdflatex not found; skipping compile. Generated artifacts are in paper/generated/."
fi
