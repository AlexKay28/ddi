# paper/ — DDI preprint scaffold

This directory holds the LaTeX scaffold for the DDI preprint. Missing generated
inputs compile as `\todoreport{...}` placeholders. The current generated tables
are hand-audited artifacts with source files named in their header comments;
they are not produced from `results/runs.jsonl` by an active renderer.
Most historical raw stores cited by T2--T6 are ignored local files and are not
available in a clean checkout. T7 and the F5 aggregate have tracked evidence;
VIBEQUEUE-50 tracks preservation and reproducible rendering for the rest.

To compile the checked-in artifacts, run `./paper/build.sh` from the repository
root. It does not regenerate tables or figures; it runs the LaTeX toolchain
when available. `paper/generated/` is versioned and hand-audited, so update its
source comment and claim boundary whenever changing an empirical table.
The removed `paper/render.py` expected absent summary files and generated
table/figure shapes that differed from the checked-in artifacts.
