# V70 Preflight Blocker Audit

**Ticket:** VIBEQUEUE-70  
**Created:** 2026-09-11T07:07:26Z  
**Decision:** STOP before preregistration, confirmation construction, or GPU work

## Stop Reason

The V67 code performs real model forwards, backwards, optimizer steps, and greedy generation. Its outputs are not merely projected metrics. However, the required V67 harness does **not** implement the registered experiment: it does not create distinct k arms, does not apply a bf16 K/V perturbation, and does not measure cache-vs-fresh behavior. Extending it to 1000 steps would extend invalid evidence, triggering the explicit V70 stop instruction.

No partial V70 files existed at preflight. No GPU command was run, no confirmation IDs were exposed or frozen, and no existing V67/V62/V68 artifact was modified.

## Blocking Findings

### 1. k0, k1, and k5 are labels, not treatments

- `scripts/v67_allinone.py:114-290` accepts `arm_name`, but uses it only in logs, result metadata, and an error label.
- `scripts/v67_allinone.py:442-454` loops over k0/k1/k5 without passing k into token construction, masks, positions, model inputs, or loss.
- `scripts/v67_allinone.py:608-634` first converts an arm label to k in a post-hoc latency estimate, after all training and evaluation.
- `scripts/v67_train.py:149-402` has the same defect: `arm` is reported but never changes model computation.
- The link-layout helper tested by `tests/test_v67_link_masks.py` is never imported or called by either V67 training script.

Therefore the reported arm differences cannot be attributed to link tokens. Since torch RNG is not reset per nominal arm, ordinary training-mode dropout is one plausible source of the differences between otherwise identical computations.

### 2. The “bf16 re-encode” perturbation never touches bf16 or K/V

`scripts/v67_allinone.py:62-77` and `scripts/v67_train.py:76-98` run `tokenizer.encode` followed by `tokenizer.decode` on document strings. They do not cast tensors to bfloat16, independently encode document K/V, or perturb a cache payload. This can alter text serialization, but cannot reproduce the registered independent-encoding bf16 numerical perturbation.

### 3. Cache-vs-fresh is not evaluated

`scripts/v67_allinone.py:315-338` creates one monolithic `inference.encode` batch and generates with `ddi_qf` (A) and `full` (D) masks. Although `CanonicalCache` and `DocCache` are imported, neither is instantiated or called. The genuine cache generation path, `ddi.cache.generate_tokens_with_doc_cache`, is not used.

The artifact consequently has an A-versus-D DDI-mask/full-reference comparison, but no B-versus-C fresh-independent-encoding/persisted-cache comparison. One of the two frozen non-inferiority gates has no observations.

### 4. Frozen statistics and reproducibility are incomplete

- `paired_bootstrap` is imported at `scripts/v67_allinone.py:518` but never called.
- `_compute_verdict` at `scripts/v67_allinone.py:603-636` computes a latency estimate only, not EM/F1 confidence intervals or gates.
- Perturbation seeding uses `hash(record["id"])`, which varies across processes unless `PYTHONHASHSEED` is fixed before startup.
- V67 preregistered 500 steps per arm, while the supplied result reports 300.
- `results/v67_arms_20260910T191742Z/` contains no arm checkpoints.
- The stored “loss curve” records step 1 and every 25th step, not the full optimization curve.
- The optimizer silently falls back from frozen AdamW8bit to ordinary AdamW if bitsandbytes is unavailable.

## Interpretation

| Question | Finding |
|---|---|
| Did real neural training occur? | Yes, according to the code path. |
| Are predictions real generated strings? | Yes, the supplied result contains generated predictions. |
| Were k0/k1/k5 distinct link-token treatments? | No. |
| Was the frozen bf16 cache perturbation implemented? | No. |
| Was cache-vs-fresh measured? | No. |
| Is V67 valid evidence to extend with longer training? | No. |

This is a disconnected-treatment and mislabeled-estimand failure, not a claim that every numeric value was synthetically generated.

## Stages Not Run

- V70 preregistration and 300-ID confirmation freeze
- V70 script and helper tests
- `nvidia-smi` or any GPU run
- 20-step smoke
- 1000-step k0/k5 training and checkpointing
- dev selection
- n=100/n=300 confirmation, confidence intervals, and gates
- V68 divergent-ID tracking
- V65/V66 latency tradeoff and final V70 verdict

These outputs are intentionally absent rather than fabricated.

## Required Remediation

1. Wire k into the actual token stream, masks, positions, training loss, and generation path; add an integration test proving k0 and k5 execute different model inputs/computations.
2. Apply the 0.15 perturbation to independently encoded bf16 K/V payloads, or revise the hypothesis to match a text perturbation before preregistration.
3. Generate per-record predictions for real fresh independently encoded K/V, persisted/reloaded K/V, and full reference.
4. Implement both paired EM/F1 gates with 10,000 resamples and seed 47.
5. Replace Python `hash()` seeding, reset all RNGs per arm, require AdamW8bit, save every-step losses, and save checkpoints.
6. Re-establish a valid V67-equivalent result before interpreting V70 as a longer-training retry.

## Pinned Inputs

| Path | SHA-256 |
|---|---|
| `experiments/v67_preregistration.md` | `24aa491c7947440bb62b77d2c7e0dca5dd445e0e16d0b7f588cb4829660b45c5` |
| `experiments/v67_hypothesis_draft.md` | `47aedad3fe57b31d3b3436a256e3384ff8444f442553dd3af5a319c8e0fe8760` |
| `experiments/v67_fresh_dev_split.json` | `eb187bb32a0d5bb14db430bfea3e2b77204cc76d222ab61afd7b0fc535fc4f35` |
| `experiments/v62_split_manifest.json` | `5ad3761106ecaed967b28d21c3919fa1869ebf70bf0e178c1e773a14bf8ee098` |
| `results/v67_arms_20260910T191742Z/result.json` | `5e48d0fb317ba93edea9a228c92764d5a8b3fa772077ea7036bbcc12abbef368` |
| `results/v67_verdict_20260910T201000Z/verdict.json` | `500ac42853964905c49ef2a6fd3fdbf199b32212cdc504d7f9aace908e9eb082` |
| `results/v68_anatomy_20260910T131036Z/differing_records.jsonl` | `c7184d0ce3ef323efb8d61c472d016dfbe4640a9dfae96c1fde1eeaca5983928` |
| `scripts/v67_allinone.py` | `d89656eec00bf8d6488feb4df5c498dc593dfcb4259b6c51f9183957d681a448` |
| `scripts/v67_train.py` | `d88f75b6d25826b86acb0c67abf84cedd0fcb877c9186de85773213742643d1b` |
| `tests/test_v67_helpers.py` | `31c9d5f73b2ad11b0901be1814f4f8d9e002e38250e25e0ca802b59aa42c07ff` |
| `tests/test_v67_link_masks.py` | `4022e93504ae5b2b9f0fd267465a9a4658d6df9d0853a92d35d44a0ef5e983d5` |

The machine-readable companion is `experiments/v70_blocker_audit.json`.
