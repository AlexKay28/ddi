# Changelog

## 2026-09-06 — real end-to-end document KV reuse

The cache audit found that the historical `ddi_cache` runners never consumed cached K/V:
correctness populated `DocCache` and then recomputed the full DDI request, while throughput
stored empty sentinels and timed an unrelated query-only generation. The prior exact-logit and
1.72x QPS claims are withdrawn.

The replacement concatenates validated restart-layout K/V blocks in request order into an HF
`DynamicCache`, prefills the query with explicit restart/cache positions and the sliced DDI-QF
mask, and preserves that mask and position layout during greedy decode. Qwen2.5-7B-Instruct
bf16 on an A100 produced exact 16-token greedy text for original, reordered, and 2-hit/1-miss
partial-reuse requests. Numerical logit identity is not established: a corrected synthetic
`N=2,L=32,Q=8` cell reports cache-vs-recompute max absolute difference 0.3125. A three-request
tiny-shape throughput sanity check measured 7.19 cache QPS versus 7.74 recompute and 8.67 full;
it validates plumbing only and is not a speedup claim.

The pinned production-shape follow-up uses three seeds, 50 measured requests per arm, Zipf
$a=1.5$, `top=32,N=5,L=500,Q=32,A=16`, and records every row at clean git `4ae5750`.
Cache/recompute paired speedups are 1.245, 1.220, and 1.192 (mean 1.219, 95% t-CI
[1.152, 1.286]); cache/full-causal mean speedup is 1.094 (95% t-CI [1.022, 1.166]).
This is evidence for that request shape, not a general serving claim.

Chronological log of experiments and project decisions. **Newest first.**

## 2026-09-04 — VIBEQUEUE-47 Flex parity root cause

Added `python -m ddi.flex_parity`, a reproducible full-model Flex-versus-SDPA
proxy diagnostic. On Qwen2.5-7B-Instruct at `N=2,L=128,Q=64`, bf16, torch
2.13.0+cu126, and an A100 80GB, all required masks fail the preregistered
model gate: token-argmax agreement is 87.19% (`full`), 95.00% (`ddi_qf`), and
94.38% (`blind`). Divergence begins in the first attention layer on identical
projected Q/K/V. Explicit score scaling, Flex prescaling, and FP32 do not
repair it, identifying a compiled-kernel numerical mismatch rather than a DDI
mask or wrapper-semantic defect. Flex model timing remains blocked.

## 2026-09-04 — VIBEQUEUE-46 matched cross-architecture transfer

Completed the preregistered fixed-manifest transfer protocol for Qwen2.5-7B
and Llama-3.1-8B: three independently trained top-8-layer DDI-QF checkpoints
per architecture, with matched base/tuned `full`, `ddi_qf`, and `blind` arms
on 100-record natural-QA manifests per seed. Neither architecture reaches the
natural-QA parity gate. Qwen normalized closure is 0.781 on hotpot F1 and 0.452
on musique F1; Llama musique F1 closure is 0.480, while Llama hotpot is
invalidated by blind-control movement. EM architecture contrasts are
unavailable because at least one side is inadmissible.

Only musique F1 permits an architecture comparison. Llama minus Qwen closure
is +0.028 with a three-seed bootstrap interval of [-0.085, +0.179], below the
preregistered 0.10 practical threshold and crossing zero. Therefore no
practical architecture difference is established; this is not evidence of
equivalence. Synthetic QA is saturated and remains a control.

## 2026-09-01 — VIBEQUEUE-42 evidence-to-claim audit

Current framing retains four bounded findings: canonical mask semantic tests;
raw direct-sparse construction at N=25,L=2048; archived batch-1 flex timing
observations (1.29x within flex and 2.13x versus dense SDPA at N=9,L=2048); and
the corrected single-run restart control (0.45 sequential, 0.47 restart).

The flex observations are not full-model parity, end-to-end, or batched
throughput evidence. The corrected control invalidates the apparent restart
wall and its RoPE interpretation, but is not a parity conclusion. Only the
four Qwen full-baseline descriptive t-intervals are CIs; all other quality
values are single-run estimates. Older entries remain chronology and must be
read through this boundary.

## 2026-08-31 — VIBEQUEUE-39 RoPE-theta sweep retired

The proposed 100M/1B RoPE-base extension is not run. Sprint D's apparent
restart gap, including its smaller gap at 10M, was measured by the evaluator
that omitted explicit cached-decode positions. The corrected control on the
same 100 hotpot_qa records is 0.47 restart EM versus 0.45 sequential EM, so
there is no gap for a larger `rope_theta` to close. This control reaches a
maximum prompt position of 4,720, below Qwen's 32,768 configured context.

The Sprint D rows remain provenance for the defect but are not a
RoPE-sensitivity curve. A new sweep would require a new question and a
pre-registered corrected-code protocol; it is not a continuation of VIBEQUEUE-39.

## 2026-08-31 — VIBEQUEUE-35 restart decode-position control

The restart-layout evaluator passed explicit position IDs for prefill but not
for cached decode. HF then inferred answer positions from the cache length,
which discontinuously changed the position scheme after a restart-layout
prefill. Propagating the explicit tail position on every decode step fixes the
evaluation contract.

| evaluator | hotpot sequential EM | hotpot restart EM | restart - sequential |
|---|---:|---:|---:|
| legacy implicit decode positions | 0.45 | 0.06 | -0.39 |
| corrected explicit decode positions | 0.45 | 0.47 | +0.02 |

The corrected 100-record control uses the local 16-layer Qwen2.5-7B checkpoint
on an A100 80 GB with torch 2.5.1+cu121 and transformers 4.52.4. It does not
show a restart deficit, but is not a parity comparison; the prior RoPE-theta
sweep used the invalid evaluator and is not interpreted. Dynamic NTK scaling is
an exact no-op here because prompt
positions peak at 4,720, below the configured 32,768 context.

## 2026-08-31 — VIBEQUEUE-33 bounded Llama 16L diagnostic

The 16-layer Llama-3.1-8B-Instruct self-distillation diagnostic completed on
one A100: 19,000 training records, 4,750 optimizer steps, 3.49B trainable
parameters, and 3h 29m 58.5s elapsed.  Holdout student DDI perplexity fell
from 3.355 to 2.118; CUDA allocator retry warnings were recoverable and the
finite-weight gate passed before the checkpoint was saved.

Matched 100-record EM evaluation:

| checkpoint | hotpot full | hotpot ddi_qf | gap | musique full | musique ddi_qf | gap |
|---|---:|---:|---:|---:|---:|---:|
| Llama 8L | 0.47 | 0.31 | 0.16 | 0.33 | 0.15 | 0.18 |
| Llama 16L, 20k records | 0.45 | 0.40 | 0.05 | 0.36 | 0.18 | 0.18 |

The bounded 16L run raises hotpot `ddi_qf` EM by 0.09 and narrows its gap by
0.11, but leaves the musique gap unchanged.  This is a single-seed,
20k-record diagnostic with no valid three-seed Llama `full` baseline for the
repository's Delta criterion.  It is therefore evidence of capacity-sensitive
improvement, not a Llama parity claim or a reason to revise the paper result.

## 2026-08-28 — flex_attention production integration: 54/54 timing grid passed

**Superseded as a model-level claim by VIBEQUEUE-40 and VIBEQUEUE-44.** The
timing rows below are archived batch-1 observations only: current full-model
flex parity fails, so they do not establish model throughput or batched serving.

### flex_attention integrated into production timing harness

Completed the integration of `flex_attention` as a first-class backend in the DDI timing harness:

- **`ddi/flex_attention.py`**: Rewrote `patch_model_for_flex()` to store `block_mask` as `model._flex_block_mask`. Patched layers use compiled `flex_attention` when set (prefill), fall back to `F.scaled_dot_product_attention` when None (decode, Q_LEN=1, memory-bound).
- **`ddi/run.py`**: Added `_measure_prefill_flex()` helper with explicit torch.compile warm-up outside timed region. Added `gc.collect()` + `torch.cuda.empty_cache()` between cells to prevent OOM cascade.
- **`experiments/performance.yaml`**: `backends: [sdpa, fa2, flex]`.
- **All 489 unit tests pass, 54/54 timing cells pass.**

### Production timing grid results (sdpa vs flex, Qwen2.5-7B-Instruct, A100 80GB)

| N | L | sdpa full (ms) | flex ddi_qf (ms) | DDI ratio sdpa | DDI ratio flex | System speedup |
|---|---|---|---|---|---|---|
| 3 | 512 | 151 | 147 | 0.98x | 0.98x | 1.03x |
| 9 | 512 | 485 | 397 | 1.00x | 1.05x | 1.22x |
| 25 | 512 | 1901 | 1066 | 1.01x | 1.21x | 1.78x |
| 3 | 2048 | 685 | 553 | 0.98x | 1.01x | 1.24x |
| 9 | 2048 | 3247 | 1601 | 1.00x | 1.25x | 2.03x |
| 25 | 2048 | 17762 | 4458 | 1.01x | **1.81x** | **3.98x** |

Key findings:
1. sdpa confirms the regression: 4D additive mask forces sdpa to slow math backend, DDI speedup flat at ~1.0x.
2. flex realizes the speedup: block-sparse kernel skips masked QK^T blocks, DDI speedup grows to 1.81x at N=25 L=2048.
3. Total system speedup (flex+ddi_qf vs sdpa+full) reaches 3.98x at N=25 L=2048.

## 2026-08-28 — flex_attention: C3 speedup breakthrough

### Historical C3 prefill-speedup interpretation — superseded

Three approaches tested to realize the analytical speedup from block-diagonal mask sparsity:

**Step 1: torch.compile — FAILED.** All three modes (default, reduce-overhead) give ~1.0x realized speedup. torch.compile fuses Python overhead and MLP but cannot restructure sdpa's dense QK^T into a sparse computation.

**Step 2: flash-attention-2 — BUILD FAILED.** System nvcc 12.2 cannot compile torch 2.13 headers (needs CUDA 12.6). No pre-built wheel for Python 3.13 + torch 2.13. Abandoned.

**Step 2b: flex_attention — BREAKTHROUGH.** `torch.nn.attention.flex_attention` (built into torch 2.13, no install needed) compiles the block-diagonal mask pattern into a block-sparse kernel that skips masked QK^T blocks entirely.

### Raw attention layer results (single layer, 28 heads, head_dim=128)

| N | L | sparsity | sdpa ratio | flex ratio | analytical |
|---|---|---|---|---|---|
| 3 | 512 | 80.7% | 0.33x | 0.96x | 5.2x |
| 9 | 512 | 93.2% | 0.25x | 1.44x | 14.8x |
| 25 | 512 | 97.5% | 0.06x | 3.24x | 40.3x |
| 25 | 2048 | 97.9% | 0.06x | 8.24x | 47.2x |

### Full model results (28 layers, Qwen2.5-7B-Instruct, A100 80GB)

| N | L | sparsity | sdpa ratio | flex ratio | analytical |
|---|---|---|---|---|---|
| 3 | 512 | 80.7% | 0.96x | 1.00x | 5.2x |
| 9 | 512 | 93.2% | 0.83x | 1.05x | 14.8x |
| 25 | 512 | 97.5% | 0.63x | 1.21x | 40.3x |
| 25 | 2048 | 97.9% | 0.40x | 1.77x | 47.2x |

### Three critical findings

1. **sdpa + 4D additive mask is a REGRESSION** (0.40x at N=25 L=2048 — ddi_qf is 2.5x SLOWER than full). The 4D additive mask forces sdpa from the optimized flash/efficient backend to the slow math backend. All previous timing numbers were measuring this regression, not "no speedup."

2. **flex_attention realizes 1.21x-1.77x model-level speedup.** At N=25 L=2048: flex ddi_qf = 4541ms vs flex full = 8060ms. The speedup is diluted by MLP/other ops but still significant at long sequences where attention dominates.

3. **flex ddi_qf beats sdpa full** — at N=25 L=2048: flex ddi_qf (4541ms) vs sdpa full (7080ms) = 1.56x faster. The masked model with flex_attention is faster than the unmasked baseline with sdpa.

### Paper impact

C3 transforms from "no speedup" (realized ~1.0x with sdpa, actually 0.40x regression) to "1.77x realized speedup at 97.9% sparsity" (with flex_attention). The previous realization gap was not structural — it was a backend limitation. flex_attention closes it.

Scripts: `scripts/timing_compile_test.py`, `scripts/timing_flex_test.py`, `scripts/timing_flex_model.py`
Tracker: VIBEQUEUE-21 (steps 1, 2, 2b documented)

## 2026-08-28 — Restart-only finetune: HARD WALL DEFINITIVE

**Superseded by VIBEQUEUE-35.** Every restart EM result in this entry used the
cached-decode evaluator that omitted explicit `position_ids`. The values and
the derived hard-wall/exposure-bias conclusion are retained only as chronology,
not as model evidence.

### Restart-only finetune results (Step 6)

Restart-only finetune: 13.8hr, 26,231 steps, trainable_layers=8, layout=restart
(exclusively restart positions; teacher always sequential). Base model =
sequential-tuned checkpoint.

Post-epoch perplexities: student_ddi_qf (sequential) = 1.638,
student_ddi_qf_restart = 1.629 — **perplexity gap INVERTED** (restart BETTER
than sequential by 0.009).

Quality eval (100 records/cell, EM):
- full hotpot=0.39, ddi_qf hotpot=0.42 (parity ✓)
- full musique=0.21, ddi_qf musique=0.12 (gap not closed, same as other checkpoints)

Restart quality (100 records/cell, EM):
- sequential hotpot=0.42, restart hotpot=0.07 — **generation gap persists**
- sequential musique=0.12, restart musique=0.00 — **generation gap persists**

### Definitive conclusion

Three finetuning strategies attempted, all fail to fix restart generation:
1. Sequential training: restart=0.06 vs sequential=0.46 (perplexity gap 1.648 vs 1.443)
2. Mixed-layout training: restart=0.07 vs sequential=0.39 (perplexity gap closed: 1.628 vs 1.635)
3. Restart-only training: restart=0.07 vs sequential=0.42 (perplexity gap inverted: 1.629 < 1.638)

Root cause: exposure bias amplified by position perturbation. Teacher-forced
perplexity is stable under restart positions, but greedy autoregressive decoding
compounds small errors that cascade under shifted positional encodings. This is
NOT a training deficiency — it's a fundamental train/test mismatch.

Paper stands with acknowledged limitation. Cache throughput claim (1.72x QPS)
scoped to sequential layout only.

## 2026-08-27 — 7B resolution chain COMPLETE

**Superseded by VIBEQUEUE-35 for restart quality.** The restart EM values and
the derived generation-gap and RoPE interpretations below used the invalid
cached-decode evaluator; they are historical chronology only.

### Paper framing decision

The paper will acknowledge the restart limitation honestly as a negative
finding. Claims that hold: quality parity (ddi_qf ≈ full after finetune),
cache correctness, cache throughput (1.72x). Claim that does NOT hold:
restart layout quality (generation degrades even after mixed finetune,
despite matching perplexities).

### Mixed-layout finetune results (Step 5)

Mixed-layout finetune: 13.8hr, 26,231 steps, trainable_layers=8, layout=mixed
(50/50 coin-flip per record between sequential and restart positions; teacher
always sequential). Base model = sequential-tuned checkpoint (HF cache for
Qwen/Qwen2.5-7B-Instruct was lost in an arc FUSE mount drop).

Post-epoch perplexities: student_ddi_qf (sequential) = 1.628,
student_ddi_qf_restart = 1.635 — **perplexity gap closed** (Δ=0.007).

Quality eval (100 records/cell, EM):
- full hotpot=0.40, ddi_qf hotpot=0.39 (parity ✓ at lower level)
- full musique=0.22, ddi_qf musique=0.12 (gap not closed)

Restart quality (100 records/cell, EM):
- sequential hotpot=0.39, restart hotpot=0.07 — **generation gap persists**
- sequential musique=0.12, restart musique=0.00

**Key insight**: perplexity (teacher-forced) closed but generation (greedy
decode) did not. The model can predict tokens given correct context under
both layouts, but greedy decoding under restart positions compounds small
errors and diverges. Root cause: 50/50 coin flip per record means each record
sees only ONE layout per epoch; the model never learns consistent behavior
across layouts for the same record.

### Sequential finetune results (Step 2-4)

Sequential finetune: 13.5hr, 26,231 steps, trainable_layers=8, layout=sequential.
Student pplx 1.648 < teacher 2.028.

Quality eval: hotpot ddi_qf=full=0.46 EM (parity ✓, but full regressed from
base 0.57→0.46). Musique ddi_qf=0.13, full=0.23 (gap not closed).

Restart eval: hotpot sequential=0.46, restart=0.06 (87% drop — CRITICAL
FAILURE). Mixed finetune was the contingency plan.

### Gradient bug fix (Step 1)

`ddi/masks.py::to_additive` was using `float("-inf")` for blocked entries.
In a fully-masked bf16 tile (doc_i × doc_j under ddi_qf), this caused NaN in
fused attention backward: `exp(-inf)/0 → NaN`. The 7B finetune was blocked
by non-finite gradients. Fix: `torch.finfo(dtype).min` (commit `7c9dc87`).

### Implementation: mixed-layout finetune

Added `layout` parameter to `finetune()`: sequential (default), restart, or
mixed (50/50 coin-flip per record). Teacher always uses sequential positions;
student gets the chosen layout. Added `layout` to `_SET_KEYS` in `run.py`.
Periodic checkpointing every 5000 steps, keep only last (~15GB). bitsandbytes
fallback to `torch.optim.AdamW` via try/except.

### Base 7B experiments (Step 0)

Quality (9 cells): full hotpot EM=0.57, ddi_qf=0.31, blind=0.13; full musique
EM=0.29, ddi_qf=0.05, blind=0.03. Timing (54 sdpa cells): realization ≈1.0x
at 95% sparsity. Cache correctness: ddi_cache ≡ ddi_recompute. Cache
throughput: ddi_cache 2.53 QPS = 1.72x vs ddi_recompute 1.47 QPS. Restart
quality: FAILS on untrained 7B (EM 0.31→0.07 hotpot).

Results posted to VIBEQUEUE-2, VIBEQUEUE-14 (timing), VIBEQUEUE-15 (cache).
Arcadia PR: https://a.yandex-team.ru/review/15334619

Numbers here are pointers, not sources of truth. Every experiment traces to a row in
`results/runs.jsonl` (ground rule 2 in [issue #1](https://github.com/AlexKay28/ddi/issues/1));
tables and figures regenerate from that store (ground rule 3). This file is the narrative layer
on top — what was run, why, and what it told us.

> **Ground rule 3 is not currently enforced, and two entries below overstate it.** The
> 2026-07-27 entry says "Ground rule 3 enforced by construction: no hand-typed numbers in the
> .tex sources" and that the root Makefile "gains `make paper`". Neither holds:
> `paper/build.sh` invokes `ddi.report_cli`, which does not exist (`ddi/reports/` holds only
> `_provenance.py`), and there is no `make paper` target in any Makefile. Worse, `build.sh`
> redirected the renderer's stderr and fell through to its `\todoreport` stub path, so it
> exited 0 and produced a paper whose every number was a stub. It now fails loudly instead
> (VIBEQUEUE-16, 2026-08-08). Nothing in `paper/sections/` types a number by hand today — but
> that is because nothing generates one either, which is a different claim.
>
> Also: `results/runs.jsonl` is gitignored, so the store this file calls the source of truth
> does not travel with a checkout. The revision behind a published number lives in the
> Nirvana instance comment; `make repro-pin INSTANCE=<id>` reads it.

---

## Entry format

**Experiment entries** — one per run or sweep:

- **Date + title** — `## YYYY-MM-DD — [phase/sub-phase] short title`
- **Sub-issue** — link (e.g., `#13` for Phase 2.1)
- **SHA** — `abcdef1` (mark dirty if applicable)
- **Config** — `configs/<file>.yaml`
- **Runs** — `run_id` list, or count + query template
- **Finding** — one or two sentences; link the generated figure / table
- **`Δ` check** — if a quality claim: gap to `full` vs. `Δ` (green / red / n/a)
- **Follow-up** — issues opened, next sweep, or `closes #N`

**Non-experiment entries** — infrastructure, decisions, rebuilds — use the same date header
without the sub-issue / SHA / runs fields.

---

## Unreleased

Phase 0 in progress. No experiment entries until the three correctness gates
([issue #4](https://github.com/AlexKay28/ddi/issues/4)) are green — per ground rule 6, no DDI
number enters the store before that.

---

## Pin history — `nirvana/common/constant.py::ARC_REFERENCE`

Every graph exports Arcadia at one revision, `constant.ARC_REFERENCE`. Almost everything
under `ddi/` and `experiments/` is job-side, so it reaches a run only through that value —
a graph published against an older revision ships the old behaviour whatever the graph
files say. What follows is what each bump fixed and what it cost to find out. **Newest
first.** Read it before lowering the pin; the constant itself carries only the trap and
the summary.

- **`be94ea06ef`** — adds `ddi/diagnose_grads.py` and the graph that runs it. NOT tidiness:
  the diagnostic block invokes `python3 -m ddi.diagnose_grads`, and that module does not
  exist at any earlier revision, so an instance published against the previous pin dies on
  `No module named ddi.diagnose_grads` after queueing for a GPU. Nothing job-side on the
  training or eval path changed, so in-flight instances were unaffected.

- **`acda80b6f5`** — adds the 500-record parity eval and the three-seed rows, and fixes
  `ddi.parity._verdict` to read the SIGN of the preservation gap: it tested `abs(r)`, so a
  finetune that IMPROVED `full` was reported as "PARITY BY REGRESSION: full degraded". That
  module is offline analysis rather than job-side, so the bump was for tidiness — but a
  published graph should never ship a parity checker that can report the opposite of what
  happened.

- **`d73e39c73d`** — clips gradients (`max_grad_norm`, default 1.0), skips a step whose
  gradients are not finite rather than stepping on them, aborts after `max_skipped_steps`
  (default 20) such steps, and refuses to `save_pretrained` a model holding non-finite
  weights. Pin at or above this revision for any training row.

  The cost of not having it, measured: on 2026-08-07 the Qwen2.5-7B row-2 finetune diverged
  between step 1 and step 50, trained on NaN for the remaining ~3,230 steps, saved the
  checkpoint, uploaded 15 GB of NaN weights to YT, evaluated them, and posted a table of
  zeros to VIBEQUEUE-2 — reporting success at every layer. Nirvana was green, the failure
  guard stayed silent, and only a human reading `loss = nan` in the table caught it. Eight
  GPU-hours for a result that had to be deleted. Qwen3.5-4B on the identical revision was
  healthy throughout, so this is model-specific divergence, not a regression — which is
  exactly why it needs a guard rather than a fix.

- **`0ca69996b1`** — caps the holdout perplexity at `holdout_eval_records` (500 in
  `finetune.yaml`). Job-side: the cap is read from the config by `ddi.run`, and both the
  config and `ddi/` only reach a job through the pin. It matters more than a tuning knob
  because of what it interacts with — the module pin (`constant.LARGE_GPU_MODULE`). Those
  perplexities run on rank 0 alone while seven ranks wait, and every YT restart repays them
  in full, so uncapped they turn each restart into a 15-40 minute tax on an 8-GPU
  allocation.

- **`72cbcf7f69`** — adds `trainable_layers` to the finetune config: freeze every decoder
  layer but the top N. It is the only reason a 27B row can exist at all — a full 27B
  finetune needs ~222 GB per rank against a 141 GB card, and no number of GPUs divides it
  because DDP replicates every term on every rank. The arithmetic lives with the code, in
  `ddi/finetune.py::_freeze_trunk`.

  This is job-side code, and at that revision `--set` merged unknown keys into the config
  without validating them (`cfg[key] = value`), so a 27B graph published below it did NOT
  fail on the unknown override — it silently ignored it, attempted the full finetune, and
  OOMed after taking the GPU slot. (`--set` has been validated per kind since VIBEQUEUE-16;
  see `ddi/run.py::_validate_set_keys`. The pin still matters for the feature itself.) The
  same commit makes gradient checkpointing non-reentrant, without which a frozen trunk
  rebuilds no autograd graph and trains nothing while still reporting a falling loss.

- **`4867c1f572`** — makes `ddi.finetune` record the FIRST and LAST optimizer step, not only
  every `log_every`-th. `log_every` defaults to 50, so row 8 — 57 records, ~7 steps per rank
  — wrote no `kl_loss` rows at all and finished an 8-GPU job with an empty loss curve and no
  error saying why. Also single-sources the eval dataset list through
  `constant.EVAL_DATASETS`, after four graph notes spent a day advertising a `kilt_nq`
  column that no run produced.

- **`671bb3499d`** — passes `enable_thinking=False` to chat templates that understand it
  (review R13). Every Qwen3.x checkpoint mirrored under `YT_MODELS_ROOT` opens a `<think>`
  block in its generation prompt, and `max_new_tokens` is 64 — so the model reasoned through
  its whole budget and never emitted a bare answer span. That is the entire explanation for
  `em == 0.0000` on the 4B and the 27B while Qwen2.5-7B (no thinking template) scored
  em 0.5700 on identical records. Anything published below this revision measures a model
  that cannot answer.

- **`91f63399f1`** — first revision that does NOT ship `results/runs.jsonl`. That file was
  tracked, so a 55,186-row metrics store travelled to every job in the checkout, and
  `ddi.aggregate.load_rows` filters only by MODEL — not by dataset, mask, config hash or
  recency. Every historical row for a model was therefore averaged into its reported table.
  Qwen2.5-7B carried 41,304 such rows across 12 datasets, so rows 22/23 reported months-old
  broken-harness numbers complete with a `ddi_strict` mask and `kilt_nq` columns that the
  config cannot produce. Qwen3.5-4B and Qwen3.6-27B carried zero, which is the only reason
  rows 19/20/24/25 are readable. Pin at or above this revision so a job starts with an empty
  store and reports what it actually measured.

- **`13d5630a4f`** — fixes the decode attention mask. The R2 rewrite sliced one big
  allow-matrix per step, handing cuBLAS a bf16 tensor whose base pointer was not 16-byte
  aligned; every run died at the first decode step with `CUBLAS_STATUS_EXECUTION_FAILED` and
  then a sticky `misaligned address`, on `full` as well as the DDI variants. Nine runs were
  lost to it on 2026-08-06.

- **`d1e2f6662c`** — drops `kilt_nq` from `quality.yaml` and `finetune.yaml`. That is
  job-side config, and without it EVERY row that uses the default dataset list aborts at the
  third dataset, because the `kilt_nq` loader raises rather than fabricating title-only
  documents (review R1). Six runs were stopped and relaunched on this revision for exactly
  that reason.

- **`e65d32d274`** — branch tip at the time; differs from `ee0dcdc06d` only in the pin value
  and a comment in `ddi_rows.py` — no job-side code at all. Set so a fresh publish exports
  exactly the tip rather than a revision two commits behind it. The rows 19/20 instances
  started on 2026-08-06 stayed pinned at `ee0dcdc06d` and were deliberately *not* restarted
  for this bump: `make exp-quality` runs out of `ddi/`, which is byte-identical across the
  two, so restarting would have cost an hour of GPU to change nothing. Re-pin freely; only
  restart running work when the diff touches `ddi/`.

- **`ee0dcdc06d`** — carries the VIBEQUEUE-2 harness review fixes, and is the reason every
  graph running on 2026-08-06 was stopped rather than left to finish: the decode path no
  longer lets `blind` read the documents (R2), the prompt names the passages and asks for a
  bare span instead of posing a bare question (R3), `kilt_nq` raises instead of fabricating
  title-only documents (R1), the KL is a mean rather than a length-weighted sum (R5),
  single-document records are dropped before they contribute exactly zero gradient (R4), and
  `ddi.delta` reads the run_ids the harness actually writes instead of a prefix nothing has
  ever produced. Anything published below this value measures what the ticket has already
  declared superseded; do not lower it to reproduce an old number — use the documented env
  knobs (`DDI_PROMPT_INSTRUCTION`, `DDI_KILT_TITLE_PLACEHOLDER`), which exist for that.

- **`819bd015f9`** — first revision where `ddi.run` leaves the process with `os._exit`
  instead of running interpreter shutdown, so the NCCL/UCX teardown segfault can no longer
  poison torchrun's exit code — and where a failed cell actually exits non-zero instead of
  reporting success. That is how a complete 4h18m 8-GPU run was discarded on 2026-08-06.

- **`ae72d6b3d6`** — first revision whose Makefile single-quotes each `SETS=` entry. Without
  it `--set datasets=[kilt_nq]` reaches the shell as an unquoted glob, so VIBEQUEUE-2 row 5
  depends on this pin rather than merely preferring it. **`28ce939d16`** is its descendant
  that adds rows 8-11; those rows put `datasets=` on the *training* config, which reaches
  the same Makefile expansion, so they inherit that requirement.

- Before that, `ddi_qwen` and `ddi_grid` each carried their own hash (`83b2840a` and
  `50a6c3b5`) — both predating the DDP barrier timeout fix in `2b2a8a891a`, so publishing
  either would have deadlocked every GPU cell on the epoch-0 holdout perplexities. That is
  why there is one pin and not one per graph.

---

## 2026-08-08 — **Starlark aggregator and cached-baseline path deleted** (VIBEQUEUE-16 P1)

The graph builder had two report backends and two eval-source modes. After the six legacy
entry points were deleted earlier the same day, exactly one combination was still reachable:
`aggregator="python"` with `with_baseline=False`. Nothing in the tree passed either argument
— all 38 publishable entry points took the defaults — so `_starlark`, `compare_starlark` /
`COMPARE_STARLARK`, `grid_starlark` / `GRID_STARLARK`, `baseline_eval_script`,
`YT_BASELINES_ROOT` and the baseline block were unreachable code kept alive only by their own
mutual references. 376 lines deleted across `scripts.py`, `ddi_pipeline.py` and
`constant.py`, 40 lines of comment and docstring added in their place — net −336.

`json_process` and every operation in `operations/external.py` survived the deletion: each
was checked individually rather than assumed dead by association. `constant.SEEDS` likewise —
it still feeds `_PROTECTED_SUFFIXES` in the checkpoint guard.

The **validation guards** that went with them encoded real operational knowledge, so the
reasoning is preserved here before the code goes.

### The baseline cache key does not cover `--set` or the environment

The cached-baseline block existed because the base model's quality is a constant:
re-deriving it on every graph spends a GPU job to learn something already known. It keyed
its YT entry on `<safe-model>__quality__<sha of experiments/quality.yaml>` — model identity
plus a hash of the eval config as committed.

That key is complete only for a row that runs the eval config **as committed**. It knows
nothing about `eval_sets` (the `--set` overrides, e.g. `max_records=500`) or `eval_env`
(e.g. `DDI_HOTPOT_N_PARAGRAPHS`), both of which change what the eval measures without
touching `quality.yaml`. So an eval-only row that set either would:

1. **read** a cache entry describing a *different* eval and report it as its own baseline
   column — a 100-record number silently standing in for a 500-record one; and worse,
2. on a miss, **write** its own numbers back under a key that does not describe them,
   poisoning the baseline for every later graph that shares the model and config hash.

`build_ddi_pipeline` therefore refused `with_baseline=True` in eval-only mode outright,
rather than letting a stale hit be compared against a fresh eval. The prescribed
alternative was to list the base model in `eval_models` and pay for the GPU job — which is
what every surviving row does, and why they are readable: `ddi_parity_eval_500` and the
`ddi_tuned_gap_*` sweep score base and tuned **in one graph, on one split**, so the
subtraction between the columns is paired rather than two runs that happen to share a label.

**The rule this leaves behind, now that no caching layer enforces it:** a cache key must
cover every input that can change the number, not just the ones that are convenient to
hash. Any future baseline cache must include the `--set` overrides and the environment in
its key, or refuse to serve rows that use them.

Two smaller guards went with it, recorded for the same reason:

- The starlark aggregator compiled the ticket into its own source text, so it could not
  accept a graph *option* for `ticket` — only a plain string. The python aggregator reads
  the ticket at run time via `${global["ticket"]}`, which is why every entry point can
  expose `ticket` as a Nirvana option. That property survives; only the backend that
  lacked it is gone.
- The starlark comparison hardcoded a base-vs-finetuned pair read as `input[0]`/`input[1]`,
  so it could never render eval-only mode's N columns, and it required `with_baseline=True`
  because it read the baseline as `input[0]`. The python report walks the payload instead
  of indexing into it, so one seed and N models render through the same code.

`json_process` itself is **not** gone: `_collect` still uses a one-line `return input`
Starlark to gather more than one eval payload, because the `Make DataFrame` block accepts a
single connection on `inp`. Only the *reporting* Starlark was deleted.

---

## 2026-08-08 — **`kilt_nq` and `nq` loaders deleted** (VIBEQUEUE-16 P4)

`ddi/data/kilt_nq.py` and `ddi/data/nq.py` are gone, with their registrations, their
config references and their tests. Neither could produce a usable measurement, and both
were still wired into `experiments/cache_restart.yaml` — one aborting its cell, the other
burning record-passes on a column the aggregator then dropped.

The loaders' post-mortems are the valuable part, so they are copied here verbatim in
substance before the code goes. Both datasets stay in
`ddi.aggregate._QUALITY_DROPPED_DATASETS`, which is what keeps a store still holding their
rows from resurrecting them as columns.

### `kilt_nq` — its documents were Wikipedia titles, not passages (review R1)

`ddi/data/kilt_nq.py::_passage_text` fell back to a `"[<title>]"` placeholder when a KILT
provenance entry carried no `text`, with a comment saying this was "so downstream code can
still count passages while a follow-up wires a real wiki lookup". The follow-up never
landed, and `facebook/kilt_tasks` provenance entries carry **no** inline `text` at all —
the passage bodies live in the separate KILT knowledge source (`facebook/kilt_wikipedia`)
and have to be joined in by `wikipedia_id` / `paragraph_id`. So every `kilt_nq` "document"
the harness ever saw was a bare Wikipedia title of 6–61 characters: `[Malware]`,
`[South Asia]`, `[2017 Wimbledon Championships – Women's Singles]`.

Why that voids the dataset rather than merely weakening it: with no passage text, `full`
and `blind` see the same information. The `full − blind` headroom — the thing the whole DDI
question is measured against — collapses, and the measured DDI gap is pinned near zero for
reasons that have nothing to do with the mask. It also defeated `_normalize`'s
"drop entries with no extractable text" filter, since `_passage_text` never returned empty
for a row that had a title. `kilt_nq` was a closed-book dataset wearing a retrieval
dataset's clothes, and every number ever reported in a `kilt_nq` column is void.

Three guards were added in response and are recorded here because they encode what "fixed"
had to mean:

- `_check_passage_coverage` raised when fewer than 50% of a split's records carried any
  passage text — a hard stop with a pointer to the missing knowledge-source join, rather
  than a dataset's worth of confident zeroes.
- `_is_placeholder_doc` recognised a fabricated document by its signature (a string that
  starts with `[` and ends with `]`; a genuine passage body does not). Needed because a
  *fresh* load yielded empty `info_contexts` for a textless row while every
  **already-written cache** held the non-empty placeholders instead — an emptiness-only
  check would have waved those straight through.
- The coverage check therefore ran on the **cached** path too, not just the fresh load.
  Every existing `kilt_nq` cache — local and the YT mirror — was written by the pre-review
  loader. Guarding only `_load_hf_split` would have left every current user reading
  fabricated documents.

Consequences that were paid before the guards existed: six graph runs were stopped and
relaunched once the loader started raising (`nirvana/common/constant.py`, `d1e2f6662c`);
four graph notes spent a day advertising a `kilt_nq` column no run produced (`4867c1f572`);
rows 22/23 published tables carrying a `kilt_nq` column built from 41,304 stale
broken-harness rows (`91f63399f1`); and graph row 5 (`DDI_KILT_N_PASSAGES=2`) was
defined-but-unlaunchable, because the `kilt_nq_n2` cache it was built for was derived by
truncating `kilt_nq_n5` and so held exactly those placeholders.

`DDI_KILT_TITLE_PLACEHOLDER=1` used to restore the old behaviour for reproducing
historical store rows. It goes with the loader. The `kilt_nq_n*` caches are still on YT if
a real passage-body join ever arrives; restoring the dataset means doing that join first,
then removing `kilt_nq` from `_QUALITY_DROPPED_DATASETS`.

### `nq` — measured nothing by construction

`ddi/data/nq.py` loaded NQ Open (`google-research-datasets/nq_open`), a question–answer
dataset with **no retrieved passages**, so `info_contexts` was always `[]`. Its module
docstring said as much and promised "a retrieval step (Phase 2.1) must run before this
loader is useful for actual DDI eval; it exists now so Phase 1.4's `full` baseline can run
against a real dataset." Phase 1.4 is long gone; the retrieval step never arrived.

With zero documents there is no block structure to mask, so `build_mask` collapses every
DDI variant to plain causal and the runners log `coerced variant ... -> 'full'` on every
record. The aggregator then dropped the rows again on the way out. A `kind: quality` or
`restart_quality` cell naming `nq` therefore spent ~100 record-passes of generation to
produce three arms of the same number, and then reported none of them. It was dropped from
`quality.yaml` in #105 for exactly this reason and survived in `cache_restart.yaml` only
because no test covered that file.

---

## 2026-07-27 — **Roadmap complete for all experimental phases**

Phases 0 → 5 all closed; only #29 (writeup) and the debt-tracker #30 (transformers-4.53 API
migration) remained after Phase 5, and #29 lands in this cycle. Every C1–C6 claim in
RESEARCH_PLAN §4 has an executable runner path and a rendered result artifact behind it.

## 2026-07-27 — [6] preprint scaffold — Phase 6 done

arxiv-submittable LaTeX skeleton in `paper/preprint.tex` + `paper/sections/00_intro.tex` through
`08_limitations_conclusion.tex`, drafted from RESEARCH_PLAN in paper voice. `paper/build.sh`
regenerates every table (markdown via `ddi-report --name X` piped through `pandoc`) and figure
(pdf via `ddi-report --name X --output ...`), stubs on empty store, then runs
`pdflatex → bibtex → pdflatex → pdflatex` when a TeX toolchain is available. `\todoreport{...}`
macro flags every cell that awaits data. Ground rule 3 enforced by construction: no hand-typed
numbers in the .tex sources. Bibliography wired to `paper/references.bib`. Root `Makefile`
gains `make paper`. Closes #29.

Follow-ups landed in the same cycle:
- `ddi-report` script entry added to `[project.scripts]` (agent flagged this — the scaffold's
  `build.sh` used the `python -m ddi.report_cli` form; both now work).
- **T2 renderer** (`ddi.reports.phase2_quality`) — the Phase 2 quality summary table referenced
  by `sections/04_results_quality.tex`. Rows per `(model, dataset, variant)`, columns for EM /
  F1 / presence / loss (median ± σ across seeds).

## 2026-07-27 — **Phase 5 complete**

All three Phase 5 sub-issues closed: #26 hop-split renderers (T5, F4), #27 PCW baseline, #28
composition probe (dataset + runner + T6 + F7). Every experiment described in RESEARCH_PLAN
§4's C1–C6 now has an executable path. Remaining on the roadmap: **Phase 6** (#29 writeup) and
the transformers-4.53 migration debt (#30). All eight prior phases' sub-issues (0.1 through 5.3)
are closed on GitHub.

## 2026-07-27 — [5.3] composition probe (dataset + runner + T6 + F7)

New dataset `composition` (`ddi.data.composition`): each record has entity `X` living in `<CITY>`
(doc_A) and `<CITY> is in <COUNTRY>` (doc_B) plus `N−2` leakage-checked distractors placed at
controlled `distance` and `pair_position`. Answer = `<COUNTRY>`. `hop_type="multi"` by
construction. `phase5_composition` runner mirrors `phase4_restart_quality` and encodes the
probe knobs in the `run_id` (`...:d{distance}:p{pair_position}:...`) so downstream slicing is
free. `T6` markdown groups by `(variant, layout, distance)` with EM mean/median/IQR/n; `F7`
plots median EM vs distance, one line per `(variant, layout)` — the picture that shows where
`ddi_qf` starts to fail as gold docs move apart. Closes #28.

## 2026-07-27 — [5.1] hop-split renderers T5 + F4

`ddi.reports.phase5_hop_split`: joins `phase2_train_free` EM rows with the record's `hop_type`
(fetched from `ddi.data.load` and cached per `(dataset, seed)` since the full split re-materialises
for each row). T5 markdown: per `(dataset, variant)`, columns for EM on single / multi / all,
plus `Δ EM (multi − single)`; adds an `EM (unknown)` column and an inline `<!-- N rows dropped -->`
note whenever any row falls to `hop_type=unknown`. F4 figure: grouped bar chart, median height,
IQR error bars, matplotlib lazy-imported. Both raise loudly on entirely-empty stores; missing
splits render `n/a`, and the degradation column is `n/a` unless both hop buckets have data —
numbers are never fabricated. Report-only; no new runner or store row. Closes #26.

## 2026-07-27 — [5.2] Parallel Context Windows baseline

`ddi.runners.phase5_pcw` — registered as `task=phase5_pcw`. Reproduces Ratner et al. (ACL 2023):
each doc is an independent context window, positions restart at 0, query attends to all
windows. In DDI's machinery this is exactly `variant=ddi_qf` + `layout=restart`; the runner
hard-codes that combo and writes rows under a distinct `phase5_pcw:{model}:{dataset}:...`
prefix so the paper's PCW column is a first-class citizen in tables (rather than getting mixed
into phase2_train_free or phase4_restart rows). No mask/layout knobs in the config — the paper
defines the setup. Greedy prefill+cached-KV generation helper copied from
`phase4_restart_quality`; a follow-up should factor both into `_shared.py`. Closes #27.

## 2026-07-27 — **Phase 3 and Phase 4 complete**

Nine sub-issues closed: #17 timing harness, #18/#19/#20 timing sweeps + configs, #21
realization ratio (T3/F3) + F6 Pareto pairing, #22 position restart, #23 per-doc KV cache,
#24 order invariance, #25 throughput (F5). The harness can now run every efficiency
measurement (FTT/TPOT with median+IQR, realized-vs-analytical per cell, backend cross-check),
route DDI's masked prefill through a content-keyed KV cache, and measure end-to-end QPS under
three reuse regimes — plus GPU is required by default so a broken container fails at boot.
Remaining sub-issues: Phase 5 (#26/#27/#28 failure envelope) and Phase 6 (#29 writeup).

## 2026-07-27 — [4.3/4.4] order invariance and throughput under reuse (F5)

`phase4_order_invariance`: prefills a record's prompt under `(variant, layout)`, then compares
last-token query logits against `n_permutations − 1` deterministic random doc-order permutations
via `torch.allclose(atol=tolerance)`. Writes per-`(record, layout)` `invariance` (1.0/0.0) and
`max_delta`. Restricted to `ddi_qf` / `ddi_bridge_k` where the invariance claim is well-defined.
Hypothesis (restart → invariant, sequential → not) is left to the report layer.

`phase4_throughput`: three modes (`ddi_cache`, `ddi_recompute`, `full_causal`) over a synthetic
doc universe with `zipf` / `uniform` / `fixed` reuse. First-pass cache path skips doc prefill on
hit; full KV-injection is a follow-up. Writes `qps`, `hit_rate`, `n_requests`, `total_elapsed_s`
per run and `per_request_latency_ms` per request. `F5` renderer joins qps + hit_rate per base
run_id and plots log-scale QPS vs hit rate with one series per mode. Closes #24, #25.

## 2026-07-27 — GPU-required by default at CLI startup

`ddi.cli.main` now calls `gpu.require_cuda()` after `gpu.configure()` when the config's
`require_gpu` key is truthy (default `true`). Silently running a real DDI sweep on CPU would
burn hours and produce meaningless numbers, so misconfigured serverless containers fail at boot
with a clear "CUDA not available" message. Opt-out via `require_gpu: false` in the config;
`ddi/configs/example.yaml` and the CLI smoke test set that opt-out because they don't need a
real model.

## 2026-07-27 — GPU usage + [3.5] realization ratio + [2.3 pairing] Pareto

**GPU:** new `ddi/gpu.py` — `is_available()`, `require_cuda()`, `gpu_info()` (device name,
compute capability, VRAM, bf16 support), `configure()` (prints one-line stderr summary,
enables TF32 on Ampere+, idempotent). `ddi.cli.main` calls `configure()` between `set_seed`
and `dispatch` so every runner prints device facts and gets TF32 without any per-runner change.
`make gpu-check` runs the check as a standalone preflight; `make setup-gpu` chains it so a
missed CUDA install fails at container-boot time, not mid-sweep.

**[3.5] realization ratio (T3, F3).** `ddi.reports.phase3_realization`: T3 tabulates
analytical vs realized vs realization ratio per N (RESEARCH_PLAN §7 shape) — the "9× gap"
story per cell. F3 plots realization ratio vs N with one line per L. Both read the store
only; `n/a` when a cell's arms are missing; entirely-empty store raises. Closes #21.

**[2.3 pairing] F6 Pareto frontier.** `ddi.reports.phase2_pareto`: pairs Phase 2.3 bridge EM
rows (`ddi_bridge_k_k{K}`) with Phase 3.2 timing rows at the same K; x = realized speedup,
y = median EM; annotates points with `k`, draws dashed lines at the `full` baseline and at
`full − Δ` when `delta_em` is in the store. Paper centerpiece.

## 2026-07-27 — [4.2] per-doc KV cache

`ddi.cache.DocCache`: disk-backed cache keyed by `sha256(model_id\0dtype\0doc_text)[:16]`; atomic
`.tmp` + `os.replace` writes; per-key `fcntl.flock` sentinel prevents concurrent-write races.
`compute_doc_kv(model, tokenizer, doc_text, layout)` runs a single-doc `variant="full"` prefill
with `build_positions(..., layout)` and extracts per-layer K/V from HF's `past_key_values`
(handles both `DynamicCache` and legacy tuple formats). Runner `phase4_cache_correctness`: for
each retrieved doc, asserts (a) two cold computes agree within tolerance and (b) the KV block
round-trips through the cache. End-to-end injection into a full prompt lives in Phase 4.3 (#24)
and 4.4 (#25). Closes #23.

## 2026-07-27 — [3.2–3.4] Phase 3 timing sweeps

`ddi.runners.phase3_timing` covers issues #18, #19, #20 with one runner. One invocation =
one grid cell `(N, L, Q, mask variant, backend, seed)`. Every row carries the shape in `run_id`,
the analytical speedup target from RESEARCH_PLAN §7 (`T²/2` for full; `Nℓ²/2 + Q·Nℓ + Q²/2` for
`ddi_qf`; `Nℓ²/2 + Q²/2 + T` for `ddi_strict`), and mask sparsity (`1 − allow_pairs / causal_pairs`
— matches §7's 90.6% at N=13/L=512/Q=64), so Phase 3.5's realized-vs-analytical is a store query,
not a re-derivation. Backend selection routes via `load_model(..., attn_implementation=...)`;
FA2-on-CPU fails fast before any download. Three config templates (`phase3_timing_N.yaml`,
`phase3_timing_L.yaml`, `phase3_backends_NxL.yaml`) that the orchestrator overrides per cell.
Closes #18, #19, #20.

## 2026-07-27 — [4.1] position restart layout + phase4_restart_quality

`ddi.positions.build_positions` returns a `[T]` long tensor: `sequential` = `arange(T)`,
`restart` = each doc's tokens start at 0, query+answer block starts at `max(doc_lengths)`. The
`max` offset keeps query→doc RoPE distances well-defined without any RoPE collision into doc
rows. `ddi.model.forward_with_mask` gains an optional `position_ids` kwarg (2D `[batch, T]` long,
default `None` preserves HF's default). Runner `phase4_restart_quality` runs both layouts on the
same records and writes rows carrying the layout in the run_id (`...:{variant_tag}:{layout}:...`),
so downstream analysis can compare directly whether restart costs anything at eval time —
ground-truth for whether Phase 4.2's cache is a win at any quality level. Closes #22.

## 2026-07-27 — [3.1] timing harness

`ddi.timing.measure`: `warmup + n_repeats` calls; CUDA-event timing when `torch.cuda.is_available()`
else `time.perf_counter`; `torch.cuda.empty_cache()` before every iteration (warmup included) so
each repeat starts cold, `torch.cuda.synchronize()` after so timing captures full compute.
`Timing` dataclass carries `median_ms`, IQR bounds, and every post-warmup sample for auditability.
Thin wrappers `measure_prefill` (FTT via `forward_with_mask`) and `measure_decode` (TPOT — full
prefill+decode time divided by `n_new_tokens`). `append_timing` writes four rows to the store
(`{metric}_median_ms`, `_iqr_low_ms`, `_iqr_high_ms`, `_repeats_ms` as JSON-encoded list per
ground rule 4). CPU fallback keeps CI green without a GPU. Reuses `ddi.report._quantile`. Closes #17.

## 2026-07-27 — **Phase 2 complete**

All four Phase 2 sub-issues closed: #13 train-free primary result, #14 conditional finetuned
sweep, #15 bridge sweep centerpiece, #16 base-vs-instruct probe. The harness can now: evaluate
one cell of the `{model} × {mask} × {dataset}` sweep on stock weights, fine-tune with the mask
present and save a reloadable checkpoint, sweep bridge widths for the F2 curve, and produce a
base-vs-instruct effect-size table (T4_probe_instruct). All quality metrics share the same
`build_mask → to_additive → forward_with_mask` code path as the baseline (Phase 1.4) so any
gap is a semantic effect, not implementation drift. Ready for Phase 3 (efficiency + realization
gap) and Phase 4 (per-document caching) in parallel.

## 2026-07-27 — [2.4] base-vs-instruct probe

`ddi.reports.phase2_base_vs_instruct`: registered as `T4_probe_instruct`. Parses
phase2_train_free run_ids to identify (model, dataset, variant); pairs base and instruct
versions of each family (strip `-Instruct` / `-instruct` suffix); reports per-family
`gap = mean(full) − mean(ddi_qf)` for base and instruct with per-seed σ, plus the effect
(base − instruct). Hypothesis in RESEARCH_PLAN §2.4: negative effect means instruct is more
DDI-friendly. Closes #16.

## 2026-07-27 — [2.3] bridge sweep + F2

`ddi/reports/` package introduced (mirrors `ddi/runners/` — side-effect registration).
`ddi/report.py` imports it at module bottom so every renderer is discoverable via
`list_reports()`. `phase2_bridge.yaml` matches the phase2_train_free schema with
`variant=ddi_bridge_k` and a `bridge_k` placeholder that the orchestrator overrides per
invocation across `{1, 2, 4, 8, 16, 32, 64, query_len}`. `F2` renderer (registered by
`ddi.reports.phase2_bridge`): reads bridge rows from the store, groups by k, plots median EM
with IQR error bars on a log-2 x-axis. Lazy matplotlib import; parent path is created on demand.
Closes #15.

## 2026-07-27 — [2.2] SFT training loop + phase2_finetune_train

`ddi.train`: from-scratch SFT loop (no `Trainer`/`trl`/`peft`) — AdamW, linear warmup then
constant LR, configurable gradient accumulation, gradient checkpointing, non-finite-loss step
skip that keeps the LR schedule honest, periodic validation loss logged as `train_val_loss` to
the store. Deterministic dataloader by re-invoking the loader per pass (no `itertools.cycle`
buffering). Checkpoint saved via `model.save_pretrained` under a hashed `_run_tag` that
sanitises model/dataset slashes. Runner `phase2_finetune_train` wraps it; eval reuses
`phase2_train_free` by pointing its `model:` at the checkpoint dir. LoRA / light-adaptation
rung deferred with a `TODO` in the config. Closes #14.

## 2026-07-27 — [2.1] train-free mask sweep runner

`ddi.runners.phase2_train_free`: reads mask variant + bridge_k from config so the orchestrator
fills the sweep grid one cell per invocation. `ddi/runners/_shared.py` factors out
`shifted_cross_entropy` so Phase 1.4 (`phase1_baseline`) and Phase 2.1 (`phase2_train_free`)
use bit-identical loss — Δ is only meaningful if the formula is shared. `_generate_with_prefill_mask`:
DDI-masked prefill via `forward_with_mask`, then greedy cached-KV argmax steps. Records with
empty `info_contexts` coerce to `variant="full"` with a per-run summary count. Config validation
fails loudly on missing keys and on `bridge_k` / variant mismatch before touching the model.
Closes #13.

## 2026-07-27 — Makefile refocused on serverless surface

`Makefile` narrowed to what a serverless container actually needs: install (`setup`, `setup-gpu`),
tests (`test`, `gates`), **datasets upload** (`upload-dataset` — pushes frozen split manifest to
HF hub), **models upload** (`upload-model` — pushes a checkpoint to HF hub), experiments (Phase
1–5 targets), results collection (`report`). Dropped: `test-fast`, `lint`, `format`, `clean` —
not part of experiment orchestration.

## 2026-07-27 — **Phase 1 complete**

All four Phase 1 sub-issues closed: #9 datasets, #11 metrics, #10 hop labels, #12 baseline
runner + Δ. The harness can now: load a frozen dataset with verifiable split hashes, tokenize
and score model output against SQuAD-style metrics, and run a `phase1_full_baseline` that writes
per-record loss + EM/F1/presence to `results/runs.jsonl` under a `full` mask routed through the
same `build_mask → to_additive → forward_with_mask` path every DDI variant will use (no
baseline-vs-variant implementation drift). `compute_delta` reads those rows and returns
`Δ = k × std_across_seeds(per_seed_mean(metric))` — this is what makes ground rule 9 real. Ready
for Phase 2 (issues #13 train-free primary, #14 conditional finetuned, #15 bridge sweep
centerpiece, #16 base-vs-instruct probe).

## 2026-07-27 — [1.4] baseline runner + Δ

`ddi.runners.phase1_baseline`: registers `task=phase1_full_baseline`. For each record: builds a
`variant="full"` mask via `build_mask` + `to_additive`, runs it through `forward_with_mask` for
answer-only cross-entropy (next-token-shifted), then `model.generate` (greedy) for EM/F1/presence.
Every metric writes a row to `results/runs.jsonl` with full provenance. Skips records with empty
`possible_answers` (loud print). The baseline shares the mask-injection path with future DDI
variants, so any drift between "baseline" and "DDI" numbers is a code bug, not an implementation
difference. `ddi.delta`: `compute_delta` filters by `phase1_baseline:{model}:{dataset}:` +
metric, groups by `provenance.seed`, computes sample std (ddof=1), returns `Δ = k × std`
(default `k=2`, requires ≥3 seeds). `record_delta` writes per-seed-mean rows AND a `delta_{metric}`
row so the whole computation is auditable per ground rule 3. `ddi-delta` CLI wraps it. Makefile
gains `phase1-baseline` and `delta` targets. Closes #12.

## 2026-07-27 — [1.2] hop labels + ddi-audit CLI

`ddi.hop_labels`: `hop_type(record)` reads the native field; `register_heuristic(dataset)` lets
later loaders (HotpotQA, Open RAG Bench) plug in a labeler for `hop_type == "unknown"` records
without touching the loaders themselves; `apply_heuristic` yields shallow-copied dicts so
callers can iterate loaders directly. `sample_for_audit` + `load_audit` + `agreement_rate` wire
the hand-audit workflow. `ddi-audit sample` writes a JSONL for review; `ddi-audit score`
re-samples, applies the heuristic, loads human labels, prints the rate + disagreements +
skipped count. Closes #10.

## 2026-07-27 — [1.1] dataset framework + NQ + synthetic

`ddi.data`: registry-based loaders yielding a common record shape (`info_contexts`, `dialog`,
`possible_answers`, `hop_type`, `gold_position`, `meta`). `splits.py` provides deterministic
`split_ids` + order-invariant `hash_split`; `manifest.py` reads/writes/validates a JSON manifest
at `results/splits/<name>.json` per ground rule 1 — mismatch raises `RuntimeError` with a diff.
Two initial loaders: `synthetic_qa` (deterministic, offline, 100 records / 60·20·20 split) for
tests and smoke runs, and `nq` (via `google-research-datasets/nq_open`) as the first real
dataset. `pyproject.toml` gains `datasets>=2.14`. Closes #9.

## 2026-07-27 — [1.3] metrics

`ddi.metrics`: SQuAD-style `normalize` (lowercase → punctuation strip → whole-token article
removal to avoid eating `"the"` inside `"theatre"` → whitespace collapse), then `exact_match`,
`token_f1` (Counter-multiset overlap, canonical SQuAD formula, max across candidates), and
`answer_presence` (refuses trivial empty-candidate hits). `make_answer_labels` returns a
`[batch, seq]` long tensor that is `input_ids` inside `[answer_start:answer_end)` and `-100`
elsewhere — feeds HF's own answer-only cross-entropy path. Closes #11.

## 2026-07-27 — **Phase 0 complete**

All seven Phase 0 sub-issues closed: #2 skeleton, #3 masks, #6 attention backends, #5 HF model
wrapper, #4 correctness gates, #7 provenance + store, #8 report generation. The DDI harness now
has a single mask definition (ground rule 8), three attention backends verified equivalent by the
Phase 0.5 gates (ground rule 6), mask injection into HF's own forward (no hand-rolled decoder),
run identity + append-only store (ground rules 1, 2), and idempotent report rendering from the
store (ground rule 3). Ready for Phase 1 (datasets + `full` baseline + fix `Δ`).

## 2026-07-27 — [0.5] correctness gates

`tests/test_gates.py`: three parametrized gates, all `@pytest.mark.gate`. Gate 1 — N=1 mask
equivalence + output equivalence over available backends. Gate 2 — SDPA-referenced backend
cross-check on doc-only fixed input; skips (never passes vacuously) when only SDPA is available.
Gate 3 — mask shape sanity plus the `len()`-vs-`.shape[-1]` regression encoded permanently. Meta
test verifies every `test_gate*` carries the marker. `gate` marker registered in `pyproject.toml`.
`make gates` runs just the gate suite. Closes #4.

## 2026-07-27 — Makefile as experiment shortcuts

Reframed root `Makefile` around experiment shortcuts. Setup collapsed to two targets (`setup`,
`setup-gpu`); every phase gets its own labeled targets (`sweep-train-free`, `sweep-bridge`,
`timing-N`, `realization`, `cache-throughput`, `pcw-baseline`, ...), each pointing at a config
under `ddi/configs/phase*.yaml`. `help` output is sectioned. `run-seeds` added for one-VM seed
loops. `report` target wraps `ddi.report_cli` (new module). Two config stubs seeded
(`phase2_train_free.yaml`, `phase2_bridge.yaml`) to demonstrate the shape.

## 2026-07-27 — [0.4] HF model wrapper with mask-preservation patch

`ddi.model.load_model` returns `(model, tokenizer)` and applies `patch_model` which installs an
instance-level `_update_causal_mask` (via `types.MethodType` on `model` or `model.model`) that
returns a supplied **4D float** mask unchanged and delegates every other shape/dtype to the
original. `unpatch_model` deletes the instance attribute to unshadow the class descriptor cleanly.
`forward_with_mask` validates 4D + floating-point and calls HF's own forward with `use_cache=False`.
No hand-rolled decoder loop. Integration test gated on `DDI_INTEGRATION=1`; unit tests use a mock.
Closes #5.

## 2026-07-27 — Makefile as serverless entry point + CI

`Makefile` reworked to use `uv` throughout (`uv pip install --system`, `uv run <cmd>`) and
reframed as the single interface for serverless experiment runs: fresh container → `make setup`
→ `make <phase-target>`. GitHub Actions workflow at `.github/workflows/test.yml` installs via
`make setup`, runs `make lint test` on Python 3.11 and 3.12, matches the serverless first-run
sequence so CI green ≡ the container will boot cleanly.

## 2026-07-27 — [0.3] attention backends

`ddi.attention.get_backend` dispatches to three implementations of one mathematical object: SDPA
with explicit additive mask (**reference** — defines correct), FlashAttention-2 varlen scoped to
doc-only prefill (uses `cu_seqlens`, CUDA-only, lazy-imported), and Triton deferred as
`NotImplementedError`. Cross-backend equivalence is verified by the Phase 0.5 gate. Closes #6.

## 2026-07-27 — [0.7] report generation

`ddi.report` provides the `@register` decorator (mirrors `ddi.cli.register_runner`), sorted
`list_reports`, and `render(name, store, output=None)` dispatching tables (markdown) vs figures
(write to output) by `T*` / `F*` naming. Store reader `iter_rows` in file order plus
`filter_rows` (metric / run_id / provenance subset). Helpers: `median_with_iqr` with linear
interpolation, `group_by` with dotted-path lookup. Idempotent by construction. Renderers for
T2/T3/F2–F6 register from later-phase modules as their data lands. Closes #8.

## 2026-07-27 — [0.2] masks.py

`ddi.masks.build_mask` returns a boolean `[T, T]` allow-matrix for all five variants (`full`,
`ddi_qf`, `ddi_strict`, `ddi_bridge_k`, `blind`) via vectorised boolean-tensor ops over row/col
index grids. `to_additive` converts to `-inf` form preserving bf16 representability. N=1 collapse
to `full` for the qf/strict/bridge_k family is explicit and matches the correctness-gate semantics
in issue #1 §0.5. Independent triple-loop reference in `tests/test_masks.py`. Closes #3.

## 2026-07-27 — [0.6] provenance and append-only store

`ddi.provenance.capture` returns a `Provenance` (git SHA/dirty, config hash, seed, model/dataset
revisions, GPU model, driver/CUDA/torch versions, UTC timestamp). `ddi.store.append` writes one
`fcntl.flock`-protected JSONL row per metric. Two-process concurrent-write test verifies parallel
sweeps do not interleave. Closes #7.

## 2026-07-27 — Makefile added

Root `Makefile` collects install / test / lint / format / run / clean targets and pins overridable
variables (`PYTHON`, `CONFIG`, `SEED`). Placeholder comments mark where phase-specific sweep
targets will land as their sub-issues close.

## 2026-07-27 — [0.1] package skeleton

`pyproject.toml`, `ddi/` package with stub modules (`masks`, `attention`, `model`, `provenance`,
`store`, `report`), working `ddi.cli` with runner registry, `ddi/configs/example.yaml`, `tests/`
with CLI smoke tests, `results/` directory. Closes #2.

## 2026-07-27 — CHANGELOG created

Repo scaffolded for the clean-room rebuild. Awaiting Phase 0.
