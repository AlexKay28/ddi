# Paper audit notes

Running log of scope- and factual-currency edits to `paper/sections/*.tex`
and `paper/preprint.tex`. New entries append at the bottom.

## Scope cut (post #81)

Issue #82 cuts the preprint scope to C1 (single-hop retrieval quality),
C3 (TTFT / realized speedup), and C4 (KV-cache reuse under restart
layout). C5 (failure envelope), C6 (distractor robustness), fine-tuning,
PCW comparison, and the composition dataset are removed from the
preprint. This audit trail records every file touched and what was
dropped, so a future maintainer can reconstruct the "why".

### Files deleted

- `paper/sections/07_failure_envelope.tex` --- removed entirely. It was
  the full C5/C6/PCW section (single-hop vs multi-hop split, PCW
  baseline, cross-document composition probe; tables T5, T6 and figures
  F4, F7). The `\input{sections/07_failure_envelope.tex}` line in
  `paper/preprint.tex` was removed with it.

### Files edited

- `paper/preprint.tex` --- abstract rewritten to drop the bridge-width /
  frontier framing and to state the scope cut. `\input` for section 07
  removed.
- `paper/sections/00_intro.tex` --- contribution list rewritten. Dropped
  the bridge-width bullet (F2/F6-adjacent) and the failure-envelope
  bullet (broken `\ref{sec:failure-envelope}`). Added an explicit
  out-of-scope sentence pointing to `sec:limitations`.
- `paper/sections/01_related_work.tex` --- PCW paragraph now says direct
  PCW comparison is out of scope; the "serves as a baseline
  (Section~\ref{sec:failure-envelope})" clause is removed to unbreak
  the ref. Attention-sinks and differentiators paragraphs reworded to
  drop "with and without fine-tuning" and the "failure mode" framing.
- `paper/sections/02_method.tex` --- mask taxonomy prose dropped
  `ddi_bridge_k`, since the bridge-width sweep is out of scope; the T1
  table itself still enumerates whatever the renderer emits. Removed
  the "pivot of the study" sentence. In the $\Delta$ subsection,
  replaced the broken failure-envelope ref with "excluded from the
  parity table".
- `paper/sections/03_experimental_design.tex` --- dataset table dropped
  the cross-document stress row (MoNaCo / 2WikiMultiHop / MuSiQue) and
  the "Comparability with prior work" row (PCW / FiD overlap). The
  parity-detection row is trimmed to NQ and TriviaQA (HotpotQA is
  multi-hop). Axis T (train-free vs finetuned) was removed. Axis M
  dropped `ddi_bridge_k` and the "centerpiece" note. Models-table
  caption dropped the mechanism-probe reference. Ground-rule (2) ref
  updated from `sec:failure-envelope` to `sec:results-caching`.
- `paper/sections/04_results_quality.tex` --- reduced to the train-free
  mask sweep (T2). Deleted the fine-tuned columns from T2's caption,
  the bridge-width curve subsection (F2), the quality/speed Pareto
  frontier subsection (F6), and the base-versus-instruct mechanism
  probe subsection (T4).
- `paper/sections/06_results_caching.tex` --- deleted the paragraph
  that referenced `sec:results-quality-bridge` (the bridge-width
  sweep). The rest of the section --- prerequisites, throughput under
  document reuse (F5) --- is C4 and stays.
- `paper/sections/08_limitations_conclusion.tex` --- removed the Model
  scale paragraph (audit target per issue #82). Added the required
  verbatim "Preprint scope" paragraph. Conclusion rewritten to drop
  the F6 frontier reference, the failure-envelope reference, and the
  bridge-$k$ sentence.

### Grep-and-scrub dispositions

`grep -rnE 'composition|ddi_stress|fact_chain|fact_agg|contra|coref|needle|pcw|finetune|probe_instruct|T4|T5|T6|F2|F6|F7' paper/sections/`

All hits under `07_failure_envelope.tex` cleared by file deletion. All
hits under `04_results_quality.tex` (F2, F6, T4, fine-tuned) cleared
by subsection deletion. The remaining `pcw` hits in `00_intro.tex` and
`01_related_work.tex` are cited-only references to Ratner et al. 2023
as prior art in the related-work discussion --- these are legitimate
literature citations, not experiment references, so they stay.

### Code dependencies flagged for #89 (code-deletion pass)

None of the removed sections referenced code by module name in a way
that forces retention beyond what #81 already lists as in-scope
(`ddi_stress`, `phase5_*`, `probe_instruct`, PCW baseline runner,
fine-tuning entry points are all cleared to delete). No sections
survived that require any of the dropped code to keep compiling; #89
can proceed on the plan in #81 without paper-side blockers.

## VIBEQUEUE-46: matched transfer result

VIBEQUEUE-46 supersedes the single-seed evidence boundary below. Qwen2.5-7B
and Llama-3.1-8B now each have three independently trained top-8-layer
checkpoints, fixed seed-specific manifests, paired per-example outputs, and
matched base/tuned controls. Neither reaches natural-QA parity. Only musique F1
admits a cross-architecture contrast: Llama minus Qwen normalized closure is
+0.028 with a 95% equal-seed bootstrap interval of [-0.085, +0.179]. This does
not establish a practical architecture difference or architecture equivalence.

## VIBEQUEUE-43: bounded Llama result (superseded)

The available Llama-3.1-8B checkpoints support only a single-seed diagnostic:
the 16-layer, 20k-record checkpoint scored full/DDI-QF EM of 0.45/0.40 on
100 hotpot examples and 0.36/0.18 on 100 musique examples. The earlier 8-layer
hotpot diagnostic had a 0.16 gap. The runner uses `seed` to select the
validation split and no fixed-ID manifest or independently trained Llama seeds
is stored, so a seed sweep from the current artifacts would not be a controlled
fixed-set cross-architecture evaluation. `04_results_quality.tex`,
`08_limitations_conclusion.tex`, `00_intro.tex`, and `preprint.tex` now state
this as an architecture- and task-dependent negative result rather than a
universal transfer claim.

## VIBEQUEUE-42: evidence-to-claim boundary

The paper and repository-facing documentation now distinguish four retained
findings from model-level conclusions: canonical mask semantic tests; raw direct
sparse construction at N=25,L=2048; archived batch-1 flex timing observations
(1.29x within flex and 2.13x versus dense SDPA at N=9,L=2048); and the corrected
single-run restart control (0.45 sequential, 0.47 restart). The flex observations
are not full-model parity, end-to-end, or batched-throughput evidence. The restart
control invalidates the earlier restart-wall and RoPE interpretations but does not
establish parity. Only the four Qwen full-baseline descriptive t-intervals are CIs;
all other quality values remain single-run estimates.

## 2026-09-06: evidence and paper reproducibility audit

The paper workflow had split into three contradictory states: prose promised
automatic generation, `paper/build.sh` documented manual artifacts, and a stale
`paper/render.py` expected absent summary files while producing incompatible
tables and figures. The stale renderer and duplicate generated-directory build
script were retired. VIBEQUEUE-50 remains open because the raw stores behind
T2--T6 are local ignored files; only T7 and the F5 aggregate have tracked source
evidence sufficient for a clean-checkout audit.

The audit corrected the T4 restart-tuned hotpot delta from +0.03 to -0.03,
identified the actual T4-16L 100-record source, recomputed T6 F1 intervals from
unrounded seed means, aligned T3 prose with its TTFT/TPOT/sparsity columns, and
removed F3 from the manuscript because its archived PDF has no matching tracked
source and disagrees with the retired renderer. Flex proxy values now report all
three failed mask arms from VIBEQUEUE-47 rather than an older aggregate value.

VIBEQUEUE-52 is retained only as campaign bookkeeping. Its preregistered seed-0
pilot stopped after DDI F1 regressed by 0.121 on hotpot_qa and 0.041 on musique,
but it is not a deciding paper cell: the run used a dirty tree, its checkpoint
was removed after hashing, and three-seed expansion and paired bootstrap did not
run. VIBEQUEUE-48, VIBEQUEUE-49, and VIBEQUEUE-50 remain open, so the research
program is not complete.

## 2026-09-06: VIBEQUEUE-50 deterministic rendering

This entry supersedes the open VIBEQUEUE-50 status in the preceding audit entry.
`paper/evidence.json` now preserves the audited aggregates behind every retained
paper artifact and records each raw input's full SHA-256 and byte size. The
standard-library `paper/render.py` validates that manifest, verifies all tracked
sources and every historical source available locally, and deterministically
renders T1, T2, T3, T4, T4-16L, T4-restart, T5, T6-delta, T7-transfer, and F5.
`paper/build.sh` runs that renderer before its optional TeX pass.

T3-flex and its numerical claims were removed because the cited historical log
is absent and therefore cannot receive hash provenance. F5 is now deterministic
TeX rather than an opaque PDF. T4-restart retains only the corrected matched
control; its old defect row combined cells from different lineages and was
removed rather than preserved as a comparison. PDF compilation was not run on the audit host,
which has neither `pdflatex` nor `bibtex`; cross-installation PDF byte identity
is not claimed without a validated pinned TeX distribution.

## 2026-09-06: VIBEQUEUE-48 scope disposition

This entry supersedes VIBEQUEUE-48's open-prerequisite interpretation in the
earlier audit entry. The bounded characterization paper does not require the
powered quality expansion or larger-model cells. Their protocols and experiment
plans remain available as deferred future work so VIBEQUEUE-48 can be reopened,
but no unrun cell is counted as completed or successful.

The admitted quality evidence is limited to the measured
Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct checkpoints, named datasets,
frozen manifests, and retained outputs from VIBEQUEUE-46. VIBEQUEUE-52 remains
a stopped one-seed pilot and adds no deciding cell. The sole valid natural-QA
architecture contrast is musique F1; it establishes no parity, equivalence,
architecture invariance, or scale trend.
