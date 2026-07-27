# paper/ — DDI preprint scaffold

This directory holds the arXiv-submittable LaTeX skeleton for the DDI preprint.
The document compiles today with `\todoreport{...}` placeholders wherever the
results store is empty, and it auto-fills as experiments land: every table
and figure in `sections/*.tex` is regenerated from `results/runs.jsonl` via
`ddi.report` (per ground rule 3 in [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) and
[issue #1](https://github.com/AlexKay28/ddi/issues/1)). No section file types
a number by hand; an unrendered cell surfaces as an explicit TODO in the
compiled PDF that names the sweep to run.

To rebuild, run `make paper` from the repository root. That target invokes
[`build.sh`](build.sh), which (i) regenerates every markdown table via
`python -m ddi.report_cli --name <T*>` and converts it to `.tex` with
`pandoc`, (ii) regenerates every figure PDF via
`python -m ddi.report_cli --name <F*> --output generated/<name>.pdf`, and
(iii) runs `pdflatex`/`bibtex`/`pdflatex`/`pdflatex` on `preprint.tex` if a
TeX toolchain is available. `pandoc` is required for the table conversions;
`pdflatex` is only needed for the final PDF (its absence is a warning, not an
error, so containers without a TeX distribution can still refresh the
generated artifacts). The `paper/generated/` directory is populated
exclusively by this script — do not edit its contents by hand.
