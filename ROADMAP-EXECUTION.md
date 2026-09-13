# DDI Research Roadmap Execution

Status is a factual execution ledger, not a claim register.  Existing dirty-source
diagnostics remain local development evidence only; no production default, training,
manuscript, Tracker, commit, or push has been changed by this ledger.

| # | Stage | Actual status | Evidence / boundary |
|---:|---|---|---|
| 1 | Plan and budget | Partially executed | V60 localization and V61 canonical smoke were constrained to one GPU, 600 seconds, 45 GiB GPU memory, under 2 GiB artifacts, and at least 20 GiB free disk. V61 used 29.78 GiB peak reserved, 35.86 seconds wall time, and 8,007,963 bytes. |
| 2 | V58 localization | Completed as development observation | V58’s observed seed-58 input IDs (`N=2,L=32,Q=8`) were reused without retokenization in V60 attempt3. It is not untouched confirmation, QA, timing, or a quality gate. |
| 3 | Canonical contract | Implemented and bounded GPU-gated | `ddi/canonical_cache.py` supplies an opt-in eager Qwen2 per-document contract. V61 ran its real checkpoint B/C path over seven observed-development variants. This does not validate a full production contract or untouched data. |
| 4 | A/B/C/D validation | Partial A/B/C diagnostic; D not run | V61 verified exact B/C repeat encoding, disk roundtrip, assembly, query logits, teacher forcing and natural decode. A is SDPA while B/C are eager, so A/B is backend-and-execution diagnostic only. No D arm, QA matrix, or speed comparison was run. |
| 5 | Precision | Bf16 reduction ablation completed; no remedy observed | On the frozen V58 development witness, toggling only `allow_bf16_reduced_precision_reduction` left the first layer-0 `q_proj` A/B divergence at `0.015625` in both arms. FP32 is expressly omitted, so no cross-precision or QA-recovery conclusion exists. |
| 6 | V59 full cost attribution | Completed and closed | Audited measured campaign: partial-hit TTFT curves for `L={32,128,512}` x `N={2,5,10}` with actual hit states 0..N, fixed-16-token TPOT/request totals, peak memory, and the bounded verdict published to the ticket and VIBEQUEUE-56. Natural-QA and production claims remain out of scope. |
| 7 | Incremental overhead | Not started | No measured incremental/partial-hit curve was run in V60; the V59 linear mixture is not such a measurement. |
| 8 | V61 sparse | CPU reference implemented; kernel integration not started | `ddi/segmented_attention.py` reproduces `build_mask` semantics for explicit Q/K/V under full, ddi_qf and blind, including native GQA and packed slots. It is dense reference math, not a sparse kernel, model backend or speed result. |
| 9 | Prospective numerics | Not started | Current result is observed V58 development input, not a frozen untouched confirmation set. |
| 10 | Matched matrix | Not started | No full matched A/B/C/D quality matrix, seeds, or frozen held-out manifests were run. |
| 11 | Assess QA recovery | Not started | No recovery claim may be made: V58 exact and paired QA gates remain failed. |
| 12 | Approved-training-only V62 | Awaiting explicit approval | No training was launched or modified. |
| 13 | Owner-discussed-only V63 | Awaiting owner discussion | No V63 work was started. |
| 14 | Untouched confirmation | Not started | The V60 input is explicitly labelled `development_observed_V58`; it cannot serve as confirmation. |
| 15 | Independent audit / publication | Not started | Parent owns audit/publication. V60 retains source snapshots, patch, manifest digest, input, events, result, and failed attempts for independent review. |

## V60 localization result (attempt3)

- Artifact: `results/v60_localization_20260908T141940Z/attempt3/`.
- Freeze: observed V58 seed 58 IDs, two 32-token documents and an 8-token query;
  Qwen2.5-7B restart checkpoint, bf16 SDPA, source snapshot and dirty patch retained.
- First A-versus-B divergence is layer 0, document 0, `q_proj`, shape
  `[1,32,3584]`, max absolute difference `0.015625`; it precedes RoPE/cache assembly.
- All 56 serialized B KV tensors per document round-trip bitwise into C; each of
  the A-slice, B, and C assemblies equals its own concatenated entries exactly.
- B versus C query logits are bitwise equal (`max_abs=0`) at all 8 query tokens,
  and B versus C prefill/final KV comparisons are 56/56 bitwise equal.
- A versus B query logits are not equal (`max_abs=0.234375`; last-token
  `0.151123046875`). A versus A-replay is also not equal (`max_abs=0.15625`),
  which keeps monolithic-query-versus-replayed-query shape effects distinct from
  the independently computed document effect.
- Teacher forcing always consumed A’s selected token. Across the retained 16
  shared-prefix steps, B versus C remained bitwise equal; no A/B argmax
  disagreement occurred in that bounded trace. Therefore there are no
  post-argmax-divergence steps to interpret.

## Stop record

Attempt1 failed before GPU during freeze because `accelerate` metadata was absent.
Attempt2 failed before GPU because provenance verification attempted to hash a vanished
temporary module. Both attempts and their partial snapshots are retained. Attempt3
completed. A later sandbox-only CUDA error 304 / NVML failure did not diagnose the host
driver and did not block the later parent-executed V61 smoke. Broader TODO3--4 validation
remains unrun.

## TODO5 bf16 reduced-precision-reduction ablation

- Completed artifact: `results/todo5_bf16_reduction_20260908T181000Z/`. The
  freeze retained observed V58 IDs, checkpoint/tokenizer identity, source
  snapshots, dirty patch, manifest digest, and a fresh subprocess for each arm.
- The default arm recorded `allow_bf16_reduced_precision_reduction=true`; the
  intervention arm recorded it as `false`. Both first diverged at layer 0,
  document 0, `q_proj`, with max absolute difference `0.015625` and failed the
  `1e-4` allclose check. The switch therefore did not remedy this frozen
  development witness.
- In both arms, B/C disk round-trip, assembly, query logits, and all 16
  teacher-forced shared-prefix comparisons were bitwise exact. A/B query logits
  remained unequal: max absolute `0.234375` by default and `0.25` with the
  switch disabled. No bounded-trace A/B argmax disagreement occurred.
- The parent run took 67.73 seconds and 7,735,760 bytes. Each arm peaked at
  15,485,370,368 reserved bytes, within the 600-second, 45-GiB, and 2-GiB
  limits. This is a no-remedy result for one bf16 flag and one observed input,
  not FP32 evidence, QA evidence, timing evidence, or untouched confirmation.
- Retained failed artifacts
  `results/todo5_bf16_reduction_20260908T173427Z/` and
  `results/todo5_bf16_reduction_20260908T180000Z/` were instrumentation
  failures. A projection forward hook returned its CPU trace tensor, which
  replaced the live CUDA output under PyTorch hook semantics. The completed
  artifact freezes the corrected observational hook and regression test.

## V61 canonical smoke result

- Artifact: `results/v61_canonical_smoke_20260908T171231Z/`.
- Freeze: the observed V58 seed-58 IDs and checkpoint were retained with a project-source
  snapshot, dirty patch, checkpoint/tokenizer tree identity, fixed request variants and runtime
  command. This is development diagnostic provenance, not an untouched or complete external
  runtime-source snapshot for confirmation evidence.
- B and C used the actual eager `CanonicalCache` API. The exact B/C gate passed for base,
  warm, reordered, duplicate, one-token document change, one-token query change and a
  shortened neighbor. Every repeat encoding, disk payload, assembly, query logit,
  three-step teacher-forced comparison and up-to-four-token natural decode was bitwise equal.
- A remained the unmodified monolithic SDPA DDI-QF baseline. A/B logit max error ranged from
  2.53125 to 3.125 across the seven cases and all but the changed-document case had differing
  all-position argmax IDs. Because B/C deliberately use eager attention, this result does not
  isolate independent document encoding from backend numerical behavior.
- V61 used 29,783,752,704 peak reserved bytes, 35.86 seconds wall time and 8,007,963 artifact
  bytes, within the frozen limits. It contains no timing measurement, QA result or acceleration
  claim.

## Session results 2026-09-09 (Tracker-documented)

- **V58 closed negative**: bf16 SDPA persisted document-KV reuse is not exact or
  quality-preserving on the frozen V58 workload (`won't fix` resolution, 8/8 checklist).
- **V59 closed with a measured verdict** (`results/v59_partial_*` artifacts, all audited):
  L=32 reuse never wins (0.60–0.73x, no feasible crossover); L=128 marginal (0.94x at N=2,
  1.13–1.36x at N=5/10 needing 92–96% hits); L=512 material (1.81–2.47x all-hit TTFT,
  measured crossovers 0.61–0.74 hit rate, ~2.4–3.6 uses). Fixed-16-token request totals
  dilute L=512 gains to ~1.10x (N=2) / ~1.27x (N=5) because TPOT (~24–26 ms) is
  arm-invariant. Peak memory ~15.4–15.7 GB allocated, 16.0 GB reserved; ~57.4 KiB per
  document token.
- **V60 development matrix + diagnosis**: first real-checkpoint A/B/C/D row
  (`results/v60_matrix_dev_20260909T091000Z/`) and 20-record development QA matrix
  (`results/v60_qa_dev20_20260909T100000Z/`, `results/v60_qa_dev20_sdpa_20260909T103000Z/`).
  Canonical B/C reuse is bitwise exact in every scenario and backend. The eager contract's
  QA deficit (EM 0.40 vs A 0.50) is dominated by eager-vs-SDPA numerics: with the SDPA
  backend the canonical contract matches A's EM exactly (0.50, 18/20 first tokens; residual
  F1 −0.057 remains undiagnosed). A same-backend (sdpa) canonical policy is the only
  candidate for policy lock; untouched confirmation requires owner-frozen data.
- **V61 prospective Flex parity fails** (`results/v61_flex_parity_prospective_20260909T110000Z.json`):
  all six cells (full/ddi_qf/blind at N=1,2) fail the 100% argmax gate with margins recorded
  (agreement 97.8–99.4%, cosine passes, max_abs up to 0.9375). Flex remains a promotion
  blocker; the segmented varlen candidate's GPU integration is the next step.
- `ddi/canonical_cache.py` now admits `backend="sdpa"` in `ExecutionContract` (identity keys
  on backend, so eager and sdpa caches cannot collide); 243 focused tests pass across the
  session's touched files.

## Final closure 2026-09-10

- V61: FP64 calibration adopted (bf16 FA2 == bf16 SDPA at RMS 1.39e-3 vs fp64); segmented-FA2
  timing campaign `results/v61_timing_20260910T071010Z/` — speed gate passed only at
  L=512/N>=5 (1.118x, 1.287x); no broad sparse claim.
- V62: k0 pilot `results/v62_pilot_20260910T071244Z/` trained finite; dev eval
  `results/v62_dev_eval_20260910T073635Z/` — ddi_qf EM 0.600->0.400 while full path improved;
  preregistered stop rule applied, k=1/5 arms stopped; closed negative.
- V63: inactive by rule (V60 confirmation passed; V62 recovery failed at dev stage).
- Program summary published to VIBEQUEUE-56; all seven tickets closed with audited artifacts.

## V64–V65 cycle closure 2026-09-10 (afternoon)

- V64 audit: preregistered recomputation downgraded V60 QA confirmation to INCONCLUSIVE
  (EM CI LB −0.083 < −0.02 floor, underpowered n=24); sidecar adjudicated metadata defects
  (sdpa everywhere; data_role note); "B→C 1.77x" falsified (true 1.89x; C/A 1.77x slower).
- V61 closed: calibration reproduced on new seeds (gate metric in-band, 10.1x margin);
  matched QA non-inferior (99/100 first tokens identical); timing scopes show CPU-launch-bound
  wrapper (13x kernel win → 1.28x prefill; request 1.17x at N=10/L=512).
- V63 closed deferred-by-rule with preregistered re-activation triggers.
- V65 closed split verdict: canonical-path CPU overhead removed (store-time validation, LRU,
  staged H2D; byte-identical outputs, suite 1026 passed) → all-hit reuse 2.05–2.17x FASTER
  than monolithic (was 1.77x slower); mixed-hit (~3x slower) and untouched-record QA
  non-inferiority (n=100 CI LBs below floor) gates FAILED and documented.
- All five tickets (V61/V63/V64/V65 + V60 correction) closed in Tracker with artifact shas.

## Ablation cycle V66–V69 + V67 closure 2026-09-10 (evening)

- V68 (closed): anatomy of 22/100 divergent confirmation records — 8 near-tie / 11 semantic / 3 truncation; two-sided, consistent with bf16 independent-encoding sensitivity, not cache-specific corruption. V60 "no semantic signature" was an n=3 artifact. Reshaped V67 hypothesis.
- V66 (closed): serving boundary measured (240 preregistered cells, 69 min). TTFT crossover h≥0.9 (L=128) / h≥0.7 (L=512); wall dilution 2.90x→1.22x (d=1→64); L=32 never wins; all-hit TTFT 4.28x [4.16,4.41]. 2775/3600 byte-identical; divergences are first-token-identical bf16 flips. Paper scoping sentence updated.
- V69 (closed): 3.55x attribution by derivation only — validation markers 53.9%, LRU tier 37.2%, staged H2D 8.9%, residual 1.6%; reconstruction closes at 62.30 ms. No GPU toggles needed.
- V67 (closed, partial success): owner-approved 3-arm experiment (k0/k1/k5, revised perturbation-consistent recipe, 1.4 h A100, 6 h cap). H2 drove dev recovery (EM +0.17, F1 +0.195 vs V62); H1 marginal; latency win preserved (2.13x). Confirmation gate FAILED (EM CI LB −0.050, F1 −0.085, both below −0.02) — dev recovery did not generalize to untouched confirmation. Documented as negative confirmation.
- All four tickets closed in Tracker with checklists ticked and artifact shas; verdicts published to VIBEQUEUE-56.

## V70 cycle: invalid V67 evidence replaced by valid PASS 2026-09-11

- Validity preflight invalidated V67 (labels-not-treatments, text-only "perturbation", no cache eval, no bootstrap):
  V67 claims retracted in Tracker; audit pinned (experiments/v70_blocker_audit.*).
- New harness: real fixed-marker k0/k5 in actual tensors, differentiable independent-bf16-KV training path,
  genuine A/B/C/D with fail-closed reopened hit-only C, paired bootstrap gates; 44 CPU tests + 15/15 real-Qwen
  connectivity smoke; prereg frozen before measurement (amendments 1-3).
- Trained k0/k5 1000 optimizer steps each (1.71 GPU-h, finite, converged); dev selection froze k5
  (C-F1 0.7054 vs 0.7011).
- CONFIRMATION n=300: C-B delta 0.000000 CI [0,0] (lossless persistence, byte-identical generations);
  C-D EM +0.240 [0.177,0.300], F1 +0.275 [0.212,0.335]; BOTH preregistered gates PASS.
- Divergent-22 post hoc 9/22; A-vs-C −0.013 EM diagnostic only; k1 untested in valid harness;
  serving latency with markers not re-measured (~5·N extra prefill tokens).
- Ops: ~63GB dead checkpoints + ~36GB host caches freed; Arcadia mount recovered twice; root disk 100%→93%.
- Paper wired: V66 measured-boundary scoping sentence + V69 attribution table (T13/T14), 102 render tests pass.
