# V59–V63 consolidated evidence review

Reviewed 2026-09-10. This is a review of retained results, not a new GPU campaign
or a replacement of original reports. Ticket comments are attributed below;
V59's report and V60's compact result were additionally read locally.

**Defect note (required by V64 sidecar):** this document cites
`results/v60_confirmation24_20260910T093500Z/`. That artifact's self-metadata
(`data_role: development` and its note) is a runner-hardcode provenance defect;
the authoritative label is `confirmation_frozen` (freeze manifest sha
`2dffebf5…` predates the run). All V60 numbers below are the pre-optimization
V60-run values; V64's adjudicated values and V65's optimized results are in the
Addendum at the end of this file.

## Sources and status at review

| Ticket | Status | Source inspected |
|---|---|---|
| VIBEQUEUE-59 | Closed | Latest comments; `VIBEQUEUE-59-REPORT.md`, including addendum |
| VIBEQUEUE-60 | Closed | Latest comments/checklist; `results/v60_confirmation24_20260910T093500Z/result.json` |
| VIBEQUEUE-61 | Open | Latest parity, FP64 calibration and timing comments; checklist |
| VIBEQUEUE-62 | Closed, stopped pilot | Latest training and development-evaluation comments |
| VIBEQUEUE-63 | Open, conditional fallback | Latest activation comment |

Closed does not mean a scientific gate passed. The queue's closure resolution
is not itself the result; V59 explicitly notes that the workflow offered only
the `Won't fix` resolution. V63's latest activation note predates V62's final
negative evaluation and therefore needs reconsideration.

## 1. V59: a measured geometry-dependent TTFT benefit

The later measured campaigns report:

| Geometry | Reported outcome |
|---|---|
| L=32, N=2/5/10 | All-hit reuse loses: approximately 0.60–0.73x |
| L=128 | Approximately 0.94x at N2 and 1.13x/1.36x at N5/N10; high hits needed |
| L=512 | Approximately 1.81–2.47x all-hit TTFT in later campaigns |
| L=512, fixed 16-token request | Approximately 1.10x at N2 and 1.27x at N5 |

The original grid in the report has different numbers (up to 2.635x); do not mix
its ratios with later partial-hit/decode campaigns. Repeat and seed aggregation
must be identified for every plotted number. Early crossover values were linear
mixture estimates; later artifacts contain actual partial-hit schedules. Retain
that distinction, including how a crossover between observed bins is estimated.

The measured cost includes substantial persisted loading, transfer/assembly and
query-mask preparation. TPOT is approximately 24–26 ms in the cited fixed-work
campaigns, explaining dilution of TTFT benefit. First-token agreement is not a
gold-answer QA or full-trajectory preservation result. These remain dirty-source
performance diagnostics; V58's negative quality result is not superseded.

## 2. V60: canonical consistency, not competitive request performance yet

The reported 24-record confirmation used canonical SDPA execution:

| Arm | EM | F1 | Median TTFT ms |
|---|---:|---:|---:|
| A original monolithic DDI | .5000 | .59317 | 124.9597 |
| B canonical fresh | .5417 | .61515 | 417.6208 |
| C canonical all-hit | .5417 | .61515 | 221.2411 |
| D full causal | .4583 | .63167 | 125.6557 |

The compact result records `b_vs_c_equal_all=true`; ticket comments describe
bitwise cold/hit agreement on all 24 records. This supports consistency within
the tested canonical contract, not arbitrary future batching/dtypes/requests.

Arithmetic correction: the displayed median ratio B/C is about **1.888x**, not
the comment's 1.77x. C/A is about **1.771x**: cached canonical execution remains
slower than original monolithic DDI. A paired per-request ratio may differ from
a ratio of medians; recompute and name the chosen estimand before publication.

### Unresolved evidence questions

- Positive mean B−A changes (+.0417 EM, +.0220 F1) do not alone establish the
  declared two-point non-inferiority criterion. The inspected compact result
  has no paired QA interval. Compute the frozen interval/test from raw paired
  outcomes; report inconclusive if its bound does not pass. Do not choose a new
  test or increase sample size opportunistically after observing outcomes.
- `data_role` says `development`, while a separate manifest is described as
  untouched confirmation. Audit actual ID intersections and selection history,
  not only the declaration. Preserve original metadata and issue an addendum.
- Backend labels/caveats conflict with `canonical_on_sdpa` claims. Verify actual
  dispatch and source/config identity before admitting cross-arm timings.
- `C_cold` and `C_all_hit` medians are nearly identical. This is an audit
  question, not proof of a bug: inspect hit traces, warmups, cache reset boundaries
  and timing scopes to establish what the cold label actually measures.

## 3. V61: segmented FA2 gains with bounded numerical calibration

Ticket-reported same-mask segmented-FA2 versus SDPA prefill:

| N | L | Speed ratio | Reported interval |
|---:|---:|---:|---|
| 5 | 512 | 1.118x | [1.115,1.121] |
| 10 | 512 | 1.287x | [1.285,1.289] |

Only 2/8 timing cells meet the 1.05 lower-bound gate; short cases regress.
These are conditional geometry-specific gains, not broad serving evidence.

All six segmented parity cells failed the original all-prefill-position argmax
gate; decode argmax agreed in 5/6. A layer-0 N2/L128/Q64 reference experiment
found bf16 FA2 and bf16 SDPA at similar RMS relative error (~.001391) against
FP64. The ticket then adopted a .02 threshold after seeing those calibration
results, describing owner authorization. This review does not independently
verify that authorization. The calibration explains a plausible numerical
effect at that cell, not every shape/layer or task outcome.

Required: preserve old failures, freeze the revised criterion and validate on
independent inputs/shapes/layers; separately evaluate generated behavior and
paired natural-QA non-inferiority. Complete requested kernel/module/KV-producing
prefill/request scopes and compare strong baselines. V61 remains open.

## 4. V62: one negative k0 training recipe; link tokens untested

The 100-step pilot completed, with no skipped/NaN steps reported. Development
evaluation on 100 records reports:

| Execution | EM base → pilot | F1 base → pilot |
|---|---|---|
| DDI-QF | .600 → .400 | .689 → .524 |
| Full | .490 → .590 | .633 → .708 |

This rejects the tested pilot's quality-recovery hypothesis on its development
set. It does not prove model-wide capability preservation just because this
full-path QA arm improves. k1/k5 link-token arms were stopped without evaluation;
their efficacy remains unknown. Any new pilot requires a new frozen hypothesis,
train/eval mask/position/gradient audit, disjoint confirmation and approved budget.

## 5. V63: stale activation decision

No selective-recompute result was reported in the inspected comment. Its latest
note still awaits the k0 outcome, which now exists. Reassess with both facts:
canonical cold/hit consistency improved, but full-reference quality remains a
separate question and the k0 pilot regressed. Neither automatically requires
nor disproves selective repair. Owner should explicitly activate a bounded
feasibility pilot or defer it. It changes the method to partial reuse + repair.

## Remaining work, in order

Filed follow-ups, assigned to alexkay28:
- [VIBEQUEUE-64](https://st.yandex-team.ru/VIBEQUEUE-64): V60 evidence audit.
- [VIBEQUEUE-65](https://st.yandex-team.ru/VIBEQUEUE-65): combined competitive
  implementation and matched validation after audit/kernel prerequisites.
Consolidated findings posted to VIBEQUEUE-56; targeted comments to V60/V61/V63.

1. Audit V60's QA intervals, metadata, timing estimand and cache-state labels.
2. Make canonical cached execution competitive with original optimized fresh
   execution, guided by V59's measured overhead. Show A/B/C/D raw timings.
3. Complete V61 independent numerical and QA validation and timing scopes.
4. Run one locked combined implementation on matched unseen QA requests,
   with strong full/prefix-cache references where claimed. Do not multiply V59,
   V60 and V61 speedups from different experiments.
5. Decide whether a new quality-recovery pilot or V63 fallback is justified;
   the unrun link-token arms remain hypotheses, not failed results.

The combined paper claim remains unproved. The current supported pieces are
geometry-dependent reuse timing, bounded canonical consistency, and bounded
segmented-prefill timing. No original report, paper evidence tier or historical
gate is changed by this review.

## Addendum 2026-09-10 (later): V64 audit and V65 combined validation executed

The owner subsequently executed V64 (preregistered, additive-only, seed-47
frozen method) and V65 (four steps). Authoritative artifacts, not comments:
`results/v64_audit_20260910T091231Z_QA_RECOMPUTATION/`,
`results/v64_audit_20260910T091417Z_METADATA_SIDECAR/`,
`results/v64_audit_20260910T091755Z_TIMING_CLAIMS/` (claims table C1–C7),
`results/v65_profile_20260910T094553Z/`, `results/v65_equiv_20260910T100522Z/`,
`results/v65_benchmark_20260910T101710Z/`,
`results/v65_combined_20260910T102918Z/`.

### V64 adjudications (supersede this review's open questions)

- QA non-inferiority (24 records, frozen paired bootstrap seed 47): **INCONCLUSIVE**
  — means favorable (EM +0.0417, F1 +0.0220) but CI lower bounds −0.0833 / −0.0708
  fall below the −0.02 floor; underpowered at n=24. A larger preregistered
  confirmation is required for a passed claim. (Matches this review's
  illustrative seed-60024 computation.)
- `data_role`: **confirmation_frozen authoritative** (freeze manifest predates
  the run); the artifacts' `development` label is a runner-hardcode defect
  (scripts/run_canonical_qa_dev.py L235/L443), preserved verbatim in the
  inventory; every citation must carry the defect note (added at the top).
- Backend: **sdpa for all five arms** per per-request records; every `eager`
  narrative is a metadata defect; A-vs-B/C is same-backend.
- Timing (ratio of medians, same backend, n=24): B/C_all_hit **1.8876×**
  [1.8419,1.9780] faster than fresh canonical; C_all_hit/A **1.7705×**
  [1.5783,1.9317] slower than monolithic. The V60 comment's "B→C 1.77×" is
  **falsified as stated** (contrast/reciprocal confusion). This review's
  1.888×/1.771× arithmetic is confirmed by the frozen method.
- Untouched-ID intersection (V64 checklist item 2) remains the owner's separate
  call; an independent review sweep found **0 overlap** between the 24
  confirmation IDs and V58's 100 observed IDs or any other retained manifest.

### V65 results (the first audited win versus the strong baseline)

- Optimized all-hit reuse, 24 frozen records, byte-identical A/C outputs 24/24:
  **C/A = 0.4885× [0.4327,0.5504]** — approximately **2.05× faster than the
  original monolithic baseline**; declared gate (≤1.05 with byte-identical
  outputs) **PASS**. Canonical fresh B/A = 3.27× slower (cold path still slow).
  Fresh-process disk load costs ~62.6 ms/pass until the LRU tier warms; cold
  per-document encode median 690.97 ms, excluded by design (prebuilt store).
- Combined serving (24 natural requests): all-hit TTFT 58.78 ms vs A 127.62 ms
  = **2.17× [1.94,2.44] faster**; end-to-end wall 5.216 s vs 7.479 s. But
  **mixed-hit (50/50) is slower than A** (386 ms, 0.33×) → the combined
  any-workload win gate **FAIL**; the win is scoped to high-hit-rate traffic.
- **Untouched n=100 QA: NOT NON-INFERIOR** under the frozen −0.02-margin rule:
  EM −0.0100 [−0.0700,+0.0500], F1 −0.0095 [−0.0527,+0.0312] (mean < 0 → fail
  classification; near-neutral point estimates, prediction-equal 78/100,
  first-token-equal 91/100).

### Updated bottom line

The warm path now has an audited, byte-identical **~2× TTFT win over the
original strong baseline** on frozen high-hit requests. Remaining gaps: (1)
quality non-inferiority unproven (near-neutral, underpowered/narrow misses);
(2) mixed-hit traffic still loses — reuse overhead dominates below high hit
rates; (3) canonical cold path 3.27× slower than monolithic — V61 sparse
integration or V63-style selective repair remain the levers. Paper promotion
must scope the speed claim to high-hit-rate warm traffic and keep quality
unresolved; no historical gate or tier was changed.
