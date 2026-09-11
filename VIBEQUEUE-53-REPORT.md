# V53 Outcome: Throughput Claim Withdrawn

## Decision

V53 implementation, full local inference campaign, artifact audit, and paper
withdrawal are complete. This is **not clean-revision deciding evidence** and
does not establish correctness-preserving cache throughput. No Tracker update,
commit, push, V54/V55 work, training, or Flex run was performed.

The new diagnostic leaves `ddi/run.py`, `ddi/cache.py`, and
`ddi/inference.py` production behavior unchanged. Historical V51 JSONL and
summary files retain their original hashes. The user's untracked
`experiments/finetune_qwen_r5b_seed1.yaml` and `wget-log` were not modified.

## Root Cause and Reproduction

The legacy benchmark lacks an explicit token/work contract. It decodes sampled
IDs into text and retokenizes while labeling rows with requested lengths; its
full-causal path always executes A decode forwards, while DDI stops on EOS,
and output counts are discarded. The small behavioral test was not the timed
workload, and cache identity did not include checkpoint/tokenizer/layout content.

The local Qwen tokenizer's seed-0 nominal L=500 universe realizes these lengths:

```text
555 521 528 525 534 524 525 516 530 534 524 517 513 526 515 519
523 536 521 528 525 528 523 519 517 524 521 532 535 521 516 532
```

Q=32 happened to remain 32 for this seed. The CPU regression also reproduces
the legacy EOS asymmetry: with budget 4 and first-token EOS, full causal makes
5 total forwards (prefill + 4), while shipped DDI makes only the prefill.
This demonstrates the potential mismatch, not that historical V51 requests
necessarily stopped early. The exact-ID V53 stream had no early EOS.

## Runtime and Protocol

The full prespecified campaign used one NVIDIA A100 80GB PCIe, driver 580.65.06,
torch 2.5.1+cu121, transformers 4.52.4, bf16 SDPA, and the existing local base
`Qwen/Qwen2.5-7B-Instruct` snapshot:

```text
/home/alexkay28/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/yt-cache
```

All snapshot files (weights, config, tokenizer) are content-hashed in the run
manifest. V51's raw rows name that base model and have no tuned checkpoint;
they do not retain weight-content hashes, so byte identity with historical V51
cannot independently be proved. V51 also used torch 2.13.0+cu126, not this
available runtime. This is a newly bounded diagnostic, not an identical-runtime
V51 replication.

Exact N=5,L=500,Q=32,A=16, top=32, Zipf a=1.5; seeds 0/1/2; 10 warmups and
50 measured requests per seed/mode. One prefill yields token 1; exactly 15
decode forwards yield tokens 2--16. EOS is ignored only in fixed-work timing.
Both full and DDI modes use the same decode convention, exact input IDs and
request stream. No production defaults were changed.

The mode orders are cache/recompute/full, recompute/full/cache, and
full/cache/recompute. Synchronized wall time includes masks, disk lookup and
write, cold KV computation, assembly/transfer, query prefill and decode, but
excludes sampling, diagnostic logit copies and trace serialization.

Before timing each seed, the separate validation namespace runs cold, warm,
permuted, partial-reuse and all 60 actual stream requests. Natural generation
uses the shipped tokenizer-EOS rule with a 16-token cap. Natural cache checks
reuse entries already populated by fixed-work validation; they are not cold
latency measurements. Teacher-forced cache comparisons follow fresh DDI IDs
so numerical errors remain meaningful even after greedy trajectories diverge.
Full decision logits are retained for fresh, cache-greedy and cache-forced paths.

## Behavioral Results

| Scope | Greedy ID failures | Natural-EOS ID failures | Numerical gate failures |
| --- | ---: | ---: | ---: |
| Seed 0, 64 validation comparisons | 0 | 0 | 64 |
| Seed 1, 64 validation comparisons | 6 | 6 | 64 |
| Seed 2, 64 validation comparisons | 19 | 19 | 64 |
| Total, including explicit repeated cases | 25 / 192 | 25 / 192 | 192 / 192 |

The unchanged numerical gate is `atol=rtol=1e-4`. Every compared logit is
finite; maximum same-prefix absolute error is **1.1416015625**. Cold/warm/
partial cases pass ID agreement for all three seeds; seed 2's explicit
permutation fails. Among the 150 measured-stream requests, 19 differ between
cache and fresh DDI (seed counts 0,4,15). Five additional failures are warmup
stream requests, and one is the explicit permutation.

All 384 natural-EOS outputs reach the 16-token cap; none stops early.
The fixed-work IDs from timing match their corresponding validation IDs for
both DDI modes on every request. Thus the disagreement is reproduced outside
the diagnostic logit-copy path, not hidden by timing.

Example: seed 1, `stream_7`, indices `[0,2,3,7,5]`, differs at the first
generated token: fresh chooses 220, cache chooses 198. Fresh top-two margin is
0.125, cache margin is 0, and the first-step maximum logit difference is
0.21875. Seed 1 `stream_5` diverges on token 2 with a zero fresh margin. These
are consistent with bf16 path-dependent numerical differences changing greedy
argmax near a tie. No kernel-level root cause or general numerical fix is
claimed. CPU fp32 tiny-Qwen agreement does not establish full-model bf16 parity.

## Timing Diagnostics Only

| Mode / measurement | Seed 0 | Seed 1 | Seed 2 |
| --- | ---: | ---: | ---: |
| Cache QPS | 1.972357 | 1.890489 | 1.847594 |
| Fresh DDI QPS | 1.595253 | 1.593203 | 1.599303 |
| Full causal QPS | 1.613400 | 1.600539 | 1.631006 |
| Cache/fresh ratio | 1.236391 | 1.186596 | 1.155249 |
| Cache/full ratio | 1.222485 | 1.181157 | 1.132794 |
| Measured cache hit rate | 0.948 | 0.944 | 0.944 |
| Warmup hits / 50 lookups | 32 | 32 | 35 |

Paired seed-ratio mean cache/fresh: **1.192746**, 95% df=2 t-interval
**[1.091097,1.294395]**. Cache/full: **1.178812 [1.067295,1.290329]**.
These are diagnostic latency ratios for potentially different output tokens,
not correctness-preserving speedups. Three seeds give weak replication; no
request-level independence or serving-wide inference is asserted. Numerical
and behavioral gate failures were retained, not relaxed or selected away.

## Provenance and Artifacts

Artifact root: `results/cache_v53_local_20260907/` (local ignored results).
Log: `results/logs/cache_v53_local_20260907.log`.

| Artifact | Purpose |
| --- | --- |
| `manifest.json` | Effective config, checkpoint/tokenizer hashes, runtime/GPU, source identity |
| `source/` | Exact executing DDI Python sources and diagnostic script |
| `working.patch` | Tracked working diff at launch |
| `inputs_seed{0,1,2}.json` | Exact document/query token IDs and 60-request streams |
| `timing.jsonl` | 540 rows: 450 measured + 90 warmup, IDs, work counts, hits, timings |
| `validation.jsonl` | 192 comparisons, errors, argmax agreements, margins, natural output IDs |
| `logits_{seed}_{case}.pt` | Full decision-logit tensors, three paths per comparison |
| `validation_cache_*`, `timing_cache_*` | Separate disk KV namespaces with identity+token keys |
| `summary.json` | Paired seed estimates, failure counts, compute usage |
| `audit.json`, `audit_source.py` | Verified trace consistency, source/data hashes, audit source |

Base SHA: `16dd67d7e53a8168cf6580e6d0197c219e6ba198`.
Executing-source map SHA-256:
`56bf338e5f7cc062c9aa69bfc89b8f10c298c1a8abd3a12a5aa90b706d5256da`.
Launch diff SHA-256:
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
That is an empty **tracked** diff, not a clean executing revision: the new
diagnostic was untracked and is preserved under `source/`. Later paper edits
are not misrepresented as the launch diff. Both manifest and summary explicitly
set `admissible_paper_evidence=false`; manifest sets `clean_revision=false`.

Timing JSONL SHA-256:
`055cf47312269ca147c6d4517d26b82aabc910b11fd6ca07f5eba5e3f54153a8`.
Validation JSONL SHA-256:
`8597c0b5394b4b17248065ad1ebc386a0a2d864f089f8e65d51d5e293f7b8d4a`.
All compact artifact hashes and per-source hashes are retained in the audit
and manifest. No clean SHA was fabricated; no commit or push was made.

## Commands and Verification

Commands run from the repository root:

```bash
.venv/bin/python -m pytest tests/test_cache_workload.py -q
PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python scripts/validate_cache_workload.py --checkpoint /home/alexkay28/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/yt-cache --output results/cache_v53_local_20260907 > results/logs/cache_v53_local_20260907.log 2>&1 &
.venv/bin/python scripts/audit_cache_workload.py results/cache_v53_local_20260907
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q
.venv/bin/ruff check scripts/validate_cache_workload.py scripts/audit_cache_workload.py tests/test_cache_workload.py paper/render.py tests/test_paper_render.py
.venv/bin/python paper/render.py
.venv/bin/python paper/render.py --check --require-all-sources
git diff --check
nvidia-smi
```

The launch command is historical: the output directory is exclusive and a
rerun must choose a new directory, never overwrite these artifacts. The GPU
process was waited for through completion and its summary/audit inspected.

Prelaunch small-shape gate: **3 passed**. Final full suite: **546 passed,
28 skipped, 4 multiprocessing-fork deprecation warnings**, 6.52 seconds.
Skipped tests require opt-in network/GPU integrations or flash attention;
the full V53 A100 campaign is separate and its correctness failures are not
represented as passing tests. Relevant Ruff checks and paper source/hash/drift
validation pass. `git diff --check` passes. PDF compilation was not performed;
`latexmk` is not installed.

The first full suite caught a new renderer regression: early return for
withdrawal bypassed the existing malformed-seed check. The renderer now validates
historical rows before rendering withdrawal. Tests cover withdrawal (no bars),
historical audit rendering, source hashes, and the existing malformed-row gate.

Compute: **871.908 seconds (14.532 minutes, about 0.242 A100 GPU-hours)** for
the campaign's recorded wall interval including load/validation/timing/output.
Checkpoint hashing occurs before that recorded interval. Peak PyTorch allocated
memory: **16,251,998,208 bytes (15.136 GiB)**. No energy measurement was collected.
Final `nvidia-smi` reports **0 MiB used, 0% utilization**, no remaining GPU work.

## Parent Tracker Handoff

| V53 checklist item | Status for parent to sync |
| --- | --- |
| Reproduce mismatches; define fixed-work and behavioral checks | Complete; protocol plus tokenizer and CPU EOS reproductions |
| Pin checkpoint/tokenizer/revision/config/namespaces; memory and tests | **Incomplete**: hashes, isolation, memory and tests complete, but no clean executing revision is available without authorized commit |
| Run exact-point cold/warm/reordered/partial correctness with traces | Execution complete; correctness fails, including seed-2 permutation; not a passing gate |
| Counterbalanced three-seed equal-work timing with traces | Complete as local diagnostics only; 540 audited rows |
| Paired summaries; update or withdraw paper claim and artifacts | Complete via withdrawal, not admission of the diagnostic ratios |

The remaining clean-revision item must stay unchecked. Do not describe V53 as
fully closed under that policy. No additional experiment is needed to justify
the withdrawal, but a future admitted replacement would need clean source
provenance and a separately justified correctness result. This handoff makes
no changes to Tracker and starts neither V54 nor V55.
