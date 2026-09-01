# paper/ — DDI preprint scaffold

This directory holds the LaTeX scaffold for the DDI preprint. Missing generated
inputs compile as `\todoreport{...}` placeholders. The current generated tables
are hand-audited artifacts with source files named in their header comments;
they are not produced from `results/runs.jsonl` by an active renderer.

To compile the checked-in artifacts, run `./paper/build.sh` from the repository
root. It does not regenerate tables or figures; it runs the LaTeX toolchain
when available. `paper/generated/` is versioned and hand-audited, so update its
source comment and claim boundary whenever changing an empirical table.
