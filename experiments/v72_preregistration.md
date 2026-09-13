# V72 Preregistration — Seed-1 Replication of the V70 Trained Contract

**Frozen:** 2026-09-12, before any seed-1 smoke or training run.
**Ticket:** VIBEQUEUE-72 (stage 1). **Inherited protocol:** the V70
preregistration `experiments/v70_preregistration.{md,json}` applies verbatim —
data splits, schedule, optimizer, gates, bootstrap, selection rule and stop
rules are all inherited unchanged. Only the deltas below are frozen here.

## Deltas from V70

1. **Seed: 1** (V70 used 47). The arm-independent SHA-256 presentation
   schedule is keyed by this seed via `scripts/v70_schedule.py`, unchanged.
2. **Checkpoint directories:** `results/v72_seed1_smoke/checkpoint_{k0,k5}`
   (20-step smoke, disposable weights) and
   `results/v72_arms_seed1/checkpoint_{k0,k5}` (full arms). Model_name/dir
   uniqueness is guaranteed by the `v72_` prefix; no V70 directory is touched.
3. **Budget:** identical 6-hour total hard cap and identical projection
   formula. Seed-47 measurement projected 1.70 hours; the smoke re-derives
   `s` under seed 1 before the full run, per the inherited rule.
4. **Success criteria (inherited, restated):** confirmation on the frozen
   300-record extension with the two paired non-inferiority gates C−B and
   C−D (margin −0.02, 10k paired resamples, seed 47 bootstrap seed — the
   bootstrap seed is part of the inherited analysis protocol and does not
   change). Gate C−B is expected to reproduce exact zero; C−D margin and
   sign are recorded as characterization, never as a pass/fail criterion
   beyond the inherited gate.
5. **Out of scope:** any serving-timing claim (VIBEQUEUE-71); Llama transfer
   (V72 stage 2, separately preregistered later).

## Environment pins (measured at freeze time)

- `scripts/v70_run.py` `93d52a1f382805d0bdb98de20f374fb9f4bcedfa3c718ba8a982f190b9ee7208`
- `scripts/v70_harness.py` `8c3483649b928048cfc45774c95764cc0360ecbfb5d560126b10b294eb696349`
- `scripts/v70_schedule.py` `bf812a8ffa85113a68b60e573c20eaecfcf9a53cec80b78e0fe3024695e7f04a`
- `tests/test_v70_harness.py` `5ba28c4965e1e4e26c7661bbc6480e59a873dd435131bbecc79f20f997ade42f`
- `experiments/v70_preregistration.json` `e06d9d6b7448c9ce200116d2691cd32cad5cee1c832c3a92bc5afa8459b148a6`
- `experiments/v70_confirmation300.json` `0df9b0eb54487fd000c0bd82d5dfa51456f39da118f82ae8fe07297f663b2568`

All six match the hashes recorded in the V70 prereg amendments — the harness
is byte-identical to the code that produced the seed-47 verdict. Pins are
re-checked by `sha256sum -c experiments/v72_SHA256SUMS` before launch.

## Amendments

To be appended below, timestamped, only between stages (after smoke, after
training, after dev selection, before confirmation), per the inherited rule.

## Amendment 1 — Budget Smoke Passed, Full Count Frozen (2026-09-12, pre-full-run)

The 20-step seed-1 smoke completed for both arms with all 40 steps finite in
loss, gradient and weights (`results/v72_seed1_smoke/checkpoint_{k0,k5}.losses.jsonl`).
Measured elapsed: k0 61.725 s, k5 63.154 s over 20 optimizer steps each.
The inherited formula gives mean step `s = 3.12197 s` and projects
`2 × 1000 × s = 6243.9 s` (1.734 h) ≤ 21600 s, so **both full arms are frozen
at 1,000 optimizer steps**. Smoke losses are stability diagnostics only and
select nothing. Per inherited convention the smoke checkpoint weights are
deleted before full training; the append-only loss logs are retained.

## Amendment 2 — Infrastructure Incident During Full Training; No Rerun Required (2026-09-12, post-training)

Both full seed-1 arms completed their frozen 1,000 optimizer steps with finite
losses and gradients (k0 1000/1000, k5 1000/1000; append-only loss logs
complete). During the final checkpoint write the `/home` filesystem reached
100% utilization and the arcadia FUSE mount crashed
("Transport endpoint is not connected"), killing the follow-on chain process
and the interrupted first dev evaluation. The arcadia mount was recovered;
integrity verification of the checkpoints:

- k0 aggregate SHA: `44abe1b8a3af2022bc3baaae564a8b4c20401beb5e0163b2d2237e809cf16fbd` (15 files, complete).
- k5 aggregate SHA: `ce8adf82548294ff07d2461dc5dbc736595c2c3fc765c08abcf036602f2531dd` (15 files, complete; `v70_metadata.json` records arm k5, seed 1, step 1000; shard bytes match the safetensors index).

Both checkpoints are byte-complete; no training rerun is required. The chain
restarts from the development-evaluation stage. This amendment records an
infrastructure incident only; no treatment, data, gate, or analysis change.
The dev-evaluation stage produced no persisted output before the crash and is
rerun from scratch.
