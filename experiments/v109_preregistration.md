# V109 Preregistration — Original-Model Full-Causal QA Reference on confirmation300

**Frozen:** 2026-09-15, before any evaluation run. **Ticket:** VIBEQUEUE-109.
**Motivation:** VIBEQUEUE-90 — no arm at n=300 in V70/V72 was the original
untrained model, so the paper makes no QA-preservation claim. This experiment
supplies the matched reference.

## Question

How close is the trained-contract persisted-cache path (V70 arm C, pinned EM
0.423333 / F1 0.556105) to the original Qwen2.5-7B-Instruct under standard
sequential full-causal inference on the same records?

## Arm

- **R (base-std):** the original Qwen/Qwen2.5-7B-Instruct hub snapshot (bf16,
  eager backend), on the byte-identical token stream used by the V70 arms:
  documents tokenized individually (`add_special_tokens=False`, dataset order,
  `info_contexts`), then the chat-templated query (`build_query_text`);
  no BOS, no link/marker tokens. Sequential positions (`arange`), full-causal
  attention (variant `full` of `ddi.masks`, prefill and every decode step via
  the `ddi.inference.generate` construction). Greedy decode,
  `max_new_tokens=64`, stop on `tokenizer.eos_token_id`.

## Data (frozen)

Records = the 300 IDs of `experiments/v70_confirmation300.json` in manifest
order, loaded via `ddi.data.load('hotpot_qa', 'train', 0)`. Gold =
`possible_answers`. The split is reused, not resampled; this experiment reads
labels only for scoring, changes no gate, and is diagnostic (no preregistered
pass/fail).

## Metrics and analysis (frozen)

- EM / F1 / presence via `ddi.metrics` on decoded predictions — the same
  functions the V70 harness used.
- Primary readout: paired per-record deltas C−R (EM and F1) against the pinned
  V70 rows in `results/v70_confirmation300_k5.json` (same record order,
  asserted), with paired percentile bootstrap: 10,000 resamples, seed 47.
- No gates, no materiality rule: this arm is a reference measurement.

## Implementation pins

- Runner: `scripts/run_v109_v110_references.py --arm base-std`.
- Tokenizer/model: hub snapshot resolved at run time; `config.json` and
  `model.safetensors` SHA-256 recorded into the result.
- Runtime: bf16, eager attention, batch 1, greedy, CUDA device recorded.
- A smoke pass on 2 train-split records (not confirmation) precedes the real
  run; smoke results are not part of the analysis.

## Deviations

Any deviation (record mismatch, decode change, metric change) must be recorded
in this file as an amendment with a UTC timestamp before the affected results
are interpreted.
