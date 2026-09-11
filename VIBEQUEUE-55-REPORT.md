# VIBEQUEUE-55 Report

## Gate Record: Reproduction, RCA, and Plan

This report records the required pre-edit gate for VIBEQUEUE-55. It integrates
the completed V53 withdrawal and V54 matched-transfer analysis without reopening
either issue. No GPU, finetune, Flex, Tracker mutation, commit, or push is part
of this work.

### Primary evidence inspected

- `VIBEQUEUE-53-REPORT.md`: the V51 cache-throughput result is withdrawn. The
  local exact-input diagnostic found 25/192 greedy-ID failures and 192/192
  numerical-gate failures; it is explicitly not clean-revision paper evidence.
- `VIBEQUEUE-54-REPORT.md` and
  `experiments/transfer_v54_posthoc_sensitivity.json`: the retained matched
  three-seed study uses `C = 1 - (T_F - T_D) / (B_F - B_B)` as a normalized
  remaining tuned gap, not training recovery. The only valid natural-QA
  architecture contrast is musique F1, Llama minus Qwen `+0.028`, equal-seed
  interval `[-0.085, +0.179]`; it establishes no practical difference.
- `EVIDENCE_TO_CLAIM.md`: legacy quality tables are descriptive; matched cells
  are limited to named models, checkpoints, frozen manifests, and retained
  outputs. Flex is deferred after its full-model parity gate failed.
- `paper/evidence.json`: retained T3 values at `N=25, l=512, Q=64` give
  `0.9503976425683476` sparsity. F5 is marked withdrawn.
- `ddi/masks.py`: strict has one query-to-document bridge row, but all answer
  rows directly attend document columns; `N=1` returns the full causal mask for
  the DDI isolation family.
- `ddi/cache.py`: the implementation assembles restart-layout document KV
  blocks, so any reuse beyond unchanged full-causal prefixes is conditional on
  this layout and implementation.

### Confirmed conflicts

1. The manuscript calls the current work single-hop/train-free while its
   central matched study uses HotpotQA, MuSiQue, and self-distillation.
2. It says no full-causal document KV is reusable and any changed document
   invalidates the entire prefix. Exact unchanged prefixes remain reusable;
   only suffixes after a changed prefix position are invalidated.
3. It implies strict prevents answer-document attention and treats `N=1` as
   evidence of general strict isolation. The implementation contradicts both:
   answer rows attend documents directly, and `N=1` is an explicit full-causal
   correctness convention because there is no cross-document isolation to test.
4. The cost prose reports approximately 94% sparsity and 16.6x at the retained
   `N=25` point. Exact lower-triangular pair counts are full `82,747,680`,
   DDI-QF `4,104,480`, ratio `20.160332125x`, and sparsity `95.039764257%`.
5. The SDPA text claims backend dispatch, matrix materialization, bottlenecks,
   and a ratio grid not retained by evidence. The retained measurement only
   establishes no observed TTFT gain across the listed `N` values at
   `l=512, Q=64`.
6. Related-work citations do not directly establish the manuscript's precise
   decoder-only sparse-attention self-distillation claim. The wording must be
   narrowed to the implementation in this study.

### Root cause and hypotheses

The paper evolved around legacy descriptive artifacts while V53 and V54 changed
the admissible evidence boundary. As a result, broad historical framing and
mechanism stories survived beside the newer matched/withdrawn results.

- Hypothesis A: replacing broad prose with explicit evidence labels resolves
  the contradictions without changing the evidence schema or renderer.
- Hypothesis B: exact triangular pair counting reconciles T3 arithmetic; no
  new timing or profiling is needed because causal attribution will be removed.
- Hypothesis C: a source-only validation is all that is available because no
  TeX engine or PDF inspection tool is installed; installing dependencies is
  outside scope.

### Minimal plan and verification

1. Reframe abstract, introduction, related work, method, design, results, and
   limitations around matched, legacy-descriptive, and deferred evidence.
   Verify with manuscript searches for superseded claims and a source review.
   Rollback: reverse the focused prose patch.
2. Correct strict, `N=1`, caching, and pair-count wording; remove unmeasured
   SDPA causal attribution. Verify equations with the deterministic arithmetic
   reproduction and relevant mask tests. Rollback: reverse the focused patch.
3. Keep evidence schema and renderer unchanged unless a source artifact must
   change. Run the renderer in require-all-sources mode, CUDA-disabled full
   pytest, Ruff, and `git diff --check`.
4. Inspect available TeX tools and run the existing build if one exists. None
   was found before edits, so record the precise PDF blocker and validate source
   artifacts, references, and TODO macros instead.

## Amendments and Validation

### Exact amended claims

- The abstract and introduction now lead with the matched three-seed result,
  preserve legacy rows as descriptive, and state that the cache throughput
  result is withdrawn rather than presenting it as a result.
- Strict semantics now state that only the last query row bridges while answer
  rows directly attend documents. The `N=1` full-causal return is labeled an
  implementation convention and correctness condition, not evidence for
  multi-document behavior.
- Full-causal caching now permits unchanged exact-prefix reuse; DDI's broader
  per-document reuse is conditional on restart layout and the implementation.
- The prefill equations use exact lower-triangular pair counts. The retained
  largest T3 cell is consistently reported as 20.1603x and 95.0398%, not
  approximately 16.6x and 94%.
- SDPA prose retains the observed no-TTFT-gain result at fixed `l=512, Q=64`
  and the listed `N` values, while removing dispatch, math-backend,
  materialization, and bottleneck attribution.
- Related work now treats distillation generically and identifies the
  block-diagonal decoder-only procedure as this study's implementation. The
  uncited Linformer entry was removed instead of used as indirect support.

### Validation results

- `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q`: 548 passed, 28
  skipped, 4 existing multiprocessing-fork deprecation warnings.
- `.venv/bin/ruff check paper/render.py tests/test_paper_render.py ddi/masks.py
  ddi/cache.py`: passed. Repository-wide `.venv/bin/ruff check .` is not a
  usable baseline: it reports 125 pre-existing violations in unrelated
  Nirvana, test, and utility files.
- `.venv/bin/python paper/render.py --check --require-all-sources`: passed.
- `./paper/build.sh`: rendered deterministic artifacts successfully; it did
  not build a PDF because `pdflatex` is absent.
- Citation audit: no missing or uncited bibliography keys. There are no active
  inline `\todo` claims. Ten `\todoreport` calls are guarded missing-artifact
  fallbacks; the generated artifacts they guard are present.
- `git diff --check`: passed before the final report update and is rerun below.

### PDF and metadata completion

Approved author metadata is now present in `paper/preprint.tex`:
Aleksandr Kaigorodov, Yandex, `alexkay28@yandex-team.ru`.

The local TeX environment was completed with `texlive-latex-base`,
`texlive-latex-recommended`, and `texlive-latex-extra`, plus `poppler-utils` for
inspection. The first PDF build exposed an unescaped `hotpot_qa` in the abstract;
that source error was corrected. `./paper/build.sh` subsequently completed its
pdflatex/bibtex/pdflatex/pdflatex sequence and produced `paper/preprint.pdf`
(15 pages). Parsed-PDF and rendered-page inspection covered the title/abstract,
method equations, transfer tables, efficiency/cache tables, withdrawal figure,
and bibliography. No unresolved citations or references remain. One 14.16pt
overfull abstract-box warning remains in the log, but visual inspection confirms
that it does not clip content or cross the page margin; the small dataset-table
underfull boxes are ordinary narrow-column spacing warnings.
