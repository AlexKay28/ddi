# GOAL

Document the bounded DDI evidence without converting single-run controls or
raw-kernel observations into parity or serving claims.

## Current state (2026-09-06) - end-to-end cache path corrected

- C1 quality: the matched three-seed Qwen/Llama transfer study reaches no
  natural-QA parity gate. Its sole valid architecture contrast is musique F1:
  Llama minus Qwen normalized closure is 0.028 with a 95% equal-seed bootstrap
  interval of [-0.085, 0.179], which establishes no practical architecture
  difference. Older Qwen cells remain single-run estimates.
- C3 compute: SDPA shows no sparsity benefit from a 4D mask. The old numerical
  Flex timing table is removed because its cited historical log is unavailable;
  no Flex model-throughput claim is admitted.
- C4 cache behavior: exact 16-token greedy text matches on one A100 diagnostic for
  original, reordered, and 2-hit/1-miss partial-reuse requests. Numerical logit identity
  fails the old tolerance (`max|delta|=0.3125` at `N=2,L=32,Q=8`).
- C4 cache throughput: the historical 1.72x result is withdrawn because its runner stored
  empty cache sentinels and did not consume document KV. The corrected three-seed study at
  `top=32,N=5,L=500,Q=32,A=16` reports 1.219x cache/recompute throughput (95% t-CI
  [1.152, 1.286]); this is a one-shape systems result, not a general serving claim.
- C4 restart control: the local 16-layer single-run control with explicit
  cached-decode positions is restart EM=0.47 versus sequential EM=0.45 on 100
  hotpot_qa records. It invalidates the legacy restart wall but does not establish
  parity. Dynamic NTK scaling is inactive: maximum position 4,720 is
  below the 32,768 threshold.
- Follow-up remediation: the preregistered one-seed VIBEQUEUE-52 pilot stopped
  after DDI F1 regressed by 0.121 on hotpot_qa and 0.041 on musique relative to
  the matched V46 seed-0 checkpoint. It is campaign bookkeeping, not a deciding
  quality cell: the run used a dirty working tree, the checkpoint was removed,
  and the planned three-seed expansion and paired bootstrap did not run.
- Research completion: VIBEQUEUE-48 (quality breadth/power) and VIBEQUEUE-49
  (validated compiled timing) remain open. VIBEQUEUE-50 now has deterministic
  rendering from hash-pinned preserved evidence. The research program is still
  not complete.

## Results summary

### Quality (EM, 100 records per cell)

| mask | dataset | base | seq-tuned | mixed-tuned | restart-tuned |
|:---|:---|:---|:---|:---|:---|
| full | hotpot | 0.57 | 0.46 | 0.40 | 0.39 |
| full | musique | 0.29 | 0.23 | 0.22 | 0.21 |
| ddi_qf | hotpot | 0.31 | 0.46 | 0.39 | 0.42 |
| ddi_qf | musique | 0.05 | 0.13 | 0.12 | 0.12 |

The listed quality values are single-run estimates and do not establish parity
across checkpoints.

### Restart control (EM, 100 records)

| evaluator | sequential hotpot | restart hotpot | restart - sequential |
|:---|:---|:---|:---|
| corrected explicit decode positions | 0.45 | 0.47 | +0.02 |

The legacy evaluator supplied restart positions during prefill but let HF infer
cached-decode positions from cache length. Its unmatched numerical row is not
retained. The claimed hard wall, exposure-bias explanation, and RoPE-theta
interpretation are invalidated. The corrected control is bounded to this
checkpoint, dataset, and 100-record subset; it does not establish a broader
failure, success, or parity claim.

## Steps

- [x] Step 0: Quality, timing, and cache experiments on untrained 7B.
- [x] Step 1: Fix `-inf` to `torch.finfo(dtype).min` in `ddi/masks.py`.
- [x] Step 2-3: Run sequential finetune and record hotpot quality estimates.
- [x] Step 4-7: Audit and invalidate legacy restart evaluations because cached
  decode omitted explicit position IDs.
- [x] Step 8: Corrected 16-layer restart control - 0.47 restart versus 0.45
  sequential EM on 100 hotpot_qa records.

## Success criteria

- [x] Single-run quality and corrected restart-control values are recorded.
- No residual restart limitation is claimed without a correctly controlled
  reproduction.

## Constraints

- HF_HUB_OFFLINE=1 everywhere.
- All results are logged to results/runs.jsonl when present.
- No bitsandbytes installed - trainable_layers=8 (top 8/28), with fallback to
  torch.optim.AdamW.
