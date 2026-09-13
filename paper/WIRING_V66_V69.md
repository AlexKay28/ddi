# Wiring V66 + V69 into the paper evidence/render pipeline

Date: 2026-09-11. CPU-only session; nothing committed; no V70 files touched.

## What changed

- `paper/evidence.json`
  - New sources (hash-pinned, `required: true`): `v66_boundary_summary`,
    `v66_boundary_t9_followup`, `v69_attribution`.
  - New artifacts: `T13_v66_boundary` (15-cell L x decode-budget crossover
    grid) and `T14_v69_attribution` (three-optimization decomposition of the
    all-hit reuse reduction), both with `data_role: development` and a
    `data_role_note` scoping them to development data / derivation-only.
- `paper/render.py`
  - `_render_t13` (T9 follow-up crossover table; adapted from
    `results/v66_boundary_20260910T132922Z/t9_followup_fragment.tex`, numbers
    not re-invented) and `_render_t14` (V69 attribution table, adapted from
    `attribution_table.md`); row-width and value validation per existing
    conventions. Regenerated outputs: `paper/generated/T13_v66_boundary.tex`,
    `paper/generated/T14_v69_attribution.tex` (plus `analysis.json`, which now
    carries the verified `T13_v66_boundary` / `T14_v69_attribution` blocks from
    `paper/expanded.py`).
- `paper/expanded.py`
  - `v66_boundary_analysis` / `v69_attribution_analysis` verify the pinned
    rows against the fragment/summary/attribution JSONs before anything
    renders; wired into `render_visuals` and `analysis.json`. One output-
    preserving reflow of the `V10_coverage.tex` literal (long line split into
    two concatenated chunks; rendered bytes unchanged, confirmed by
    `paper.render --check`).
- `paper/sections/main_story.tex`
  - The "high-hit-rate" scoping sentence (intro paragraph) replaced with the
    measured-boundary sentence from
    `results/v66_boundary_20260910T132922Z/scoping_sentence.md` (LaTeX-
    adapted): measured crossovers, `2775/3600` byte-identical repeats with the
    bf16 reduction-order caveat, quality non-inferiority NOT established.
  - T9 caption: "scoped to high-hit-rate warm traffic" replaced by the
    Table~\ref{tab:v66boundary} boundary reference.
  - New T13 table environment + lead-in sentence in Section `sec:cache`
    (direct follow-up to Table `tab:v65serving`).
  - New short "Where the audited win came from (V69)" paragraph citing the
    attribution table: 53.9% / 37.2% / 8.9% of the 221.24 -> 62.30 ms
    reduction, envelope inferred not measured, residual 1.6%.
  - Conclusion paragraph: "scoped to high-hit-rate warm traffic" replaced by
    the conclusion-paragraph variant from `scoping_sentence.md`.
  - Reproducibility paragraph now lists V66/V69 among the pinned artifacts.
- `paper/sections/appendix.tex`
  - The `T14_v69_attribution` table lives in Appendix `app:cache` (Cache
    Anatomy and Provenance) with a derivation-scope paragraph, referenced from
    the main story as Table `tab:v69attribution`; kept out of the main story
    per minimal-clutter convention.

## Artifact hashes consumed

- `results/v66_boundary_20260910T132922Z/summary.json`
  sha256 `3a4752465fe6cf0ec7b326e1a6f7108b5401dad588792188bf331f70b5a4a215`
  (8231 bytes) — crossover verdicts, identity counts (2775/3600), grid.
- `results/v66_boundary_20260910T132922Z/t9_followup_evidence_fragment.json`
  sha256 `8e25ccea000ceddeebed639c826ac22f349666dddcd9b4ba8a1559eca8d74316`
  (2106 bytes) — the 15 pinned T13 rows.
- `results/v69_attribution_20260910T162613Z/attribution.json`
  sha256 `90f34eec2d9b24426a0af79ba757bedad178b8cb1abecf503cb0a87aea2188d1`
  (9971 bytes) — pre/post totals, phase removals, envelope budget, residual.
- Not pinned (human-readable companions only): `crossover_table.md`,
  `scoping_sentence.md`, `t9_followup_fragment.tex`, `attribution_table.md`.

## Verification

- `python -m paper.render --check`: validated (all generated outputs match).
- `pytest tests/test_paper_render.py -q`: 102 passed.
- `ruff check paper/render.py paper/expanded.py`: clean.
- `pdflatex preprint.tex` (single syntax pass, temp output dir): compiles.

## Deliberately NOT claimed

- V66 is development-data only: frozen V62 dev split, seed 47, single seed,
  15 timed repeat pairs per cell. No confirmation-quality claim; quality
  non-inferiority is NOT established by it.
- Byte-identity is reported as 2775/3600 timed repeats (825 divergent repeat
  records; bf16 reduction-order flips deeper in the trajectory; first token
  always identical) — never as unconditional byte-identity for V66.
- Crossover h is the smallest nominal grid h with a CI-supported win under the
  frozen materiality rule; "none in grid" means no supporting cell in the
  frozen grid, not impossibility.
- V69 is a derivation (prereg verdict `derivation_only`): pre/post totals
  trace to the confirmation-frozen V60/V65 runs (V64-adjudicated), but phase
  leaves are uniformly rescaled from the development-role V65 overhead profile
  (factor 0.945248). OPT-3 (14.20 ms, 8.9%) is a bounded envelope within the
  23.04 ms pre-optimization `cache_assemble` budget — inferred, not measured;
  the implied 8.84 ms post-optimization assemble is not a measurement. The
  3.48 ms unattributed remainder (1.6% of the pre-total) is within the
  declared 5% tolerance, not zero.
