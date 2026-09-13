# V70 Preregistration

**Frozen:** 2026-09-11T09:11:05Z, before any 20-step or full V70 training run  
**Machine-readable protocol:** `experiments/v70_preregistration.json`

## Treatment And Hypothesis

V70 compares k0 with k5 fixed-marker conditioning. The five marker embeddings are fixed deterministic vectors, not learned links. No document is deleted or dropped. Documents remain in original dataset order; document shuffle is explicitly disabled because it is not implemented as a common arm-independent treatment.

The hypothesis is that k5 improves persisted-cache C quality relative to k0 while C remains non-inferior to fresh independent B and full-reference D under the frozen gates.

The teacher is identical across arms: no-link, sequential, full-causal targets. Student training uses a stable arm-independent SHA-256 schedule keyed by seed, global presentation index, and record ID. Exactly 15% of presentations use the differentiable independently encoded bf16 document K/V path and 85% use monolithic encoding.

## Training

- Seed: 47.
- Trainable scope: top 8 decoder layers.
- Optimizer: `bitsandbytes.optim.AdamW8bit`, required; no fallback.
- Effective batch: 4 record presentations and backwards per optimizer step. Each loss is divided by 4 before backward; one optimizer step follows the four accumulated backwards.
- LR: linear warmup on optimizer steps 1 through 20, reaching `1e-5` at step 20, then constant `1e-5`.
- Requested run: 1,000 optimizer steps per arm, hence 4,000 presentations per arm and exactly 5 epochs over the frozen 800-record train set.
- `--steps` always means optimizer steps.

Both arms first run 20 optimizer steps. Let `s` be total measured training elapsed across both arms divided by 40 optimizer steps. If `2 * 1000 * s <= 21600` seconds, both full arms use 1,000 steps. Otherwise the common count is `floor(floor((21600 / s / 2) / 50) * 50)`, with a declared minimum of 300. Timings, projection, and any adjusted count must be written as an amendment before either full run. The 6-hour total GPU-training cap remains hard; if the minimum cannot fit, stop rather than exceed it.

## Frozen Data

- Train: V62 train 800, ID SHA `a3b1f5b6b1fcd9e099969b62d7ee2922c1e2e1e62397d9bbd4c1c6b42630bcbb`.
- Development: V67 fresh dev 100, ID SHA `9f5975606844e805afbf6449b5b03d020762e3adef0dd5862a032318291544ac`.
- Confirmation: `experiments/v70_confirmation300.json`, 300 IDs, SHA `ba7263ae886d929edb63595bfb693e1579269c1814969dffb934d55cf13051d1`.
- Original V62 confirmation100 is frozen as confirmation positions 1-100, SHA `529f0109eefbf2ef42ede00be146d95cf19876e35859bc1358fbceb1b1bf04d7`.

The confirmation extension was reconstructed from the same seed-0 HotpotQA train stream and V62 exclusion rule. The frozen V62 first 1,000 eligible IDs were reproduced exactly before selecting the next 200 after additionally excluding V67 fresh dev. Selection read IDs only, not labels, answers, predictions, or metrics.

## Selection And Gates

Development selection lexicographically maximizes persisted C EM, then persisted C F1, between k0 and k5. Selection is valid only if C is demonstrated to be a true hit from a reopened persisted store and C-B persistence sanity passes.

Confirmation has two primary paired non-inferiority gates: C-B and C-D. For both EM and token F1, the paired 95% percentile-bootstrap lower bound must be at least `-0.02`. Use 10,000 paired resamples and seed 47. A and B are diagnostic. V68 confirmation labels or predictions cannot be used for selection; the divergent population is reported post hoc only.

## Connectivity Evidence

The passed connectivity smoke is infrastructure validation, never quality selection:

- Result: `results/v70_harness_smoke_20260911T085908Z/result.jsonl`, SHA `2f0c564b2c4fa56c21a16ec58f360043f58c56fc9eb4ac337ecddf3afe1d2ea2`.
- Manifest: `results/v70_harness_smoke_20260911T085908Z/manifest.json`, SHA `4a01ded0bbd205250b3ac9333d8158b4162aedcceab9718ced6c4658d40b66df`.
- All 15 assertions passed; k0/k5 logit delta 3.5703125; independent-versus-monolithic deltas 1.25/1.75; peak allocated memory 49.718 GiB.
- The exact smoke-time source hashes remain pinned in the JSON protocol; later training-semantics edits do not rewrite that historical evidence.

## Stop Rules

Stop on a smoke-budget failure, NaN/nonfinite loss/gradient/weight, treatment disconnect or common-schedule mismatch, checkpoint restore mismatch, failure to prove C is a true persisted hit, or the 6-hour cap.

## Reproduction

Next command, run from repository root:

```bash
for arm in k0 k5; do CUDA_VISIBLE_DEVICES=0 python scripts/v70_run.py train --arm $arm --steps 20 --checkpoint-dir results/v70_20step_seed47/checkpoint_$arm --seed 47; done
```

CPU verification:

```bash
pytest -q tests/test_v70_harness.py
ruff check scripts/v70_run.py scripts/v70_harness.py scripts/v70_schedule.py tests/test_v70_harness.py
sha256sum -c experiments/v70_SHA256SUMS
```

Rollback, limited to this cycle's V70 files: delete `experiments/v70_confirmation300.json`, `experiments/v70_preregistration.json`, `experiments/v70_preregistration.md`, and `experiments/v70_SHA256SUMS`; restore only `scripts/v70_run.py` and `tests/test_v70_harness.py` from their pre-cycle copies. Do not alter V67 artifacts or production modules.

## Pre-Full-Run Amendment 1 (2026-09-11T09:24:16Z)

The first budget-smoke attempt failed before optimizer step 1 because the new output parent did not exist. The only runner change creates `checkpoint_dir.parent` before opening its append-only loss log; the existing accumulation test now uses a nested absent parent as a regression case. This is infrastructure repair, not a treatment or analysis change.

The completed 20-step smoke measured 59.9812 seconds across k0's 20 optimizer steps and 62.4252 seconds across k5's 20 steps. The frozen mean-step formula gives 3.06016 seconds and projects 6,120.32 seconds (1.7001 hours) for two 1,000-step arms. The budget gate therefore passes and the full count is frozen at **1,000 optimizer steps per arm**. All 160 presentations had finite gradients; each arm received 12 independently encoded presentations. Smoke losses are stability diagnostics only and cannot select an arm.

Authoritative summary: `results/v70_20step_seed47/smoke_summary.json`. The 20-step checkpoint weights are disposable and are deleted before full training; the append-only loss logs and summary are retained.

## Pre-Evaluation Amendment 2 (2026-09-11T11:26:43Z)

This amendment was written before any full development evaluation. A real one-record persisted evaluation reproduced a representation-only B/C comparison failure: fresh B K/V tensors were GPU-resident while reopened persisted C tensors were CPU-resident. `_kv_equal` now first requires identical shape and dtype, then compares `detach().cpu()` tensors. This is an infrastructure-only equality fix; it does not alter model execution, generated IDs, treatment, gates, margins, bootstrap, frozen IDs, selection, or filtering. The repeated one-record persisted evaluations then passed for both arms, with B and C generated ID lists exactly equal and C still enforced as a hit-only read from a reopened store.

Both full training arms completed the frozen 1,000 optimizer steps with effective batch 4 and 4,000 presentations. Their record and encoding schedules were identical; each used 595 independent and 3,405 monolithic presentations. All logged losses and gradients and both final checkpoints were finite. The k0 first-50/last-50 mean losses were 1.516735762801254 and 0.2669688443746418; k5 values were 1.6901578945468645 and 0.27664729200303556. These are convergence diagnostics only and are not quality outcomes or selection evidence. The descriptive artifact is `results/v70_arms_valid_seed47/training_summary.json`, SHA `6790335d153fbe5c7eb5fead81bdf589db8b04a55b51109dbeb6bb4154b46b2f`.

Exact pre-development-evaluation pins:

- Code: `scripts/v70_harness.py` `8c3483649b928048cfc45774c95764cc0360ecbfb5d560126b10b294eb696349`; `scripts/v70_schedule.py` `bf812a8ffa85113a68b60e573c20eaecfcf9a53cec80b78e0fe3024695e7f04a`; `scripts/v70_run.py` `93d52a1f382805d0bdb98de20f374fb9f4bcedfa3c718ba8a982f190b9ee7208`; `tests/test_v70_harness.py` `5ba28c4965e1e4e26c7661bbc6480e59a873dd435131bbecc79f20f997ade42f`.
- k0 checkpoint aggregate SHA: `f3f1e91ce5beb30ddd8c4200b3365487511ac6c34dc04de125ac660d56c2dc88`; metadata SHA `315403dacf4ed7a3900a24322605e1a9812f52e54933bc32cc26084f4b9ae1de`; config SHA `9f68052c645602b88a3452048dd6e59add015a4e37114ba1bf83bd785057fed5`; loss-log SHA `0915a268f8da61e3fab5592ff18479209c5774ee337020d9b48ea88ca548efb1`.
- k5 checkpoint aggregate SHA: `04fe4dcb3c70f0ef75f0cdfbf1f99164a90ab54df0833cb4b14cc56bd723ba87`; metadata SHA `f2d218b6a6692442bc590bc66c96030426ad8e9aab93ba28d14c3ae51ae276f0`; config SHA `9f68052c645602b88a3452048dd6e59add015a4e37114ba1bf83bd785057fed5`; loss-log SHA `ff04ebdf63391ef9ee92b4dd84742a2fed19e9a5b040c2a6893e038a0fdf6661`.
- One-record persisted evaluation: k0 `results/v70_eval_smoke_k0.json` SHA `402dbdb5f193558d0c4ea351d8306ca7fbde633fd8fb295c84b1540c744f8413`; k5 `results/v70_eval_smoke_k5.json` SHA `1e58794bb36d40965273a0749f27ed66aa443e81843e53bc34c11fb911a73fe1`.

The checkpoint aggregate is SHA-256 over the sorted JSON mapping of every checkpoint-relative file path to its SHA-256, using the runner's default JSON separators. Thus it covers all four safetensor shards plus configuration and tokenizer files.

## Pre-Confirmation Amendment 3 (2026-09-11T12:25:48Z)

This amendment was written after development evaluation and selection but before opening any confirmation data or running confirmation. No confirmation labels, predictions, or metrics were inspected. There are no treatment, data, filtering, gate, bootstrap, selection-rule, or confirmation-protocol changes.

Independent parsing verified that both development artifacts contain 100 records with the same IDs in the same order. B and C generated IDs are identical on every row in both arms. Both artifacts record that C read hit-only from a reopened persisted store, failed closed on any encode attempt, retained the same entry count and keys, and matched fresh B K/V after shape/dtype checks and detached CPU normalization. Both C-B and C-D gates pass for both arms.

The preregistered lexicographic selection values are:

- k0 persisted C: EM `0.54`, F1 `0.7011170919723552`.
- k5 persisted C: EM `0.54`, F1 `0.7053563956368835`.

The C EM values tie, and k5 has the higher C F1, so **k5 is frozen as the selected arm**. The selected checkpoint is `results/v70_arms_valid_seed47/checkpoint_k5`: aggregate SHA `04fe4dcb3c70f0ef75f0cdfbf1f99164a90ab54df0833cb4b14cc56bd723ba87`, config SHA `9f68052c645602b88a3452048dd6e59add015a4e37114ba1bf83bd785057fed5`, metadata SHA `f2d218b6a6692442bc590bc66c96030426ad8e9aab93ba28d14c3ae51ae276f0`.

Frozen development evidence:

- k0: `results/v70_dev_k0.json`, SHA `381db1e15cb711b7574e1c4d3b15812c074ea5863b78b0f9a946ee5144824ba5`.
- k5: `results/v70_dev_k5.json`, SHA `73060dfe0caf62ef6963109e2c8663cf6a025ca195430f0c901faaf18eac6936`.
- Selection JSON: `results/v70_dev_selection.json`, SHA `0b10dba778ea2413a98da44a80201a9a477903a1caac94aac8c926c5e41d614a`.
- Selection Markdown: `results/v70_dev_selection.md`, SHA `bcbf67833d3b228b959941d2cc8927b8fcf6191094df85a9009c3b8e14a336c1`.

## Amendment 3 — Development Selection Frozen (2026-09-11T14:55:00Z)

Development evaluation ran on the frozen V67 dev 100 for both arms with identical protocol (64 new tokens, preregistered gates, true reopened hit-only persisted C):

- k0: C EM 0.54, C F1 0.7011170919723552; C-B exact on 100/100 rows (CI [0,0]); C-D pass.
- k5: C EM 0.54, C F1 0.7053563956368835; C-B exact on 100/100 rows (CI [0,0]); C-D pass.

Selection rule (C EM, then C F1): **k5 selected** (EM tie, F1 tie-break). Selected checkpoint `results/v70_arms_valid_seed47/checkpoint_k5`, aggregate sha `04fe4dcb3c70f0ef75f0cdfbf1f99164a90ab54df0833cb4b14cc56bd723ba87`.

Artifacts: `results/v70_dev_selection.json` (sha `2481d71a88a6af2d0a7db6858933ab6b653b13b3f61382a6d2189dbc18cf4f85`), dev artifacts sha `381db1e1…` (k0) / `73060dfe…` (k5). No confirmation data opened; no protocol changes.
