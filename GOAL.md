# GOAL

Document the bounded DDI evidence without converting single-run controls or
raw-kernel observations into parity or serving claims.

## Current state (2026-08-31) - restart control corrected

- C1 quality: Qwen quality values are single-run estimates. The observed hotpot
  cells include 0.46 versus 0.46 (sequential) and 0.39 versus 0.40 (mixed),
  but neither is a parity conclusion.
- C3 compute: archived batch-1 flex prefill timings show a speed signal, but
  no current flex model-throughput claim is admitted: the full-model parity
  gate fails at `N=2,L=128` before the direct sparse or batched path is timed.
- C4 cache correctness: PASSES (ddi_cache equals ddi_recompute).
- C4 cache throughput: 1.72x QPS (2.53 vs 1.47) remains a separate throughput
  result.
- C4 restart control: the local 16-layer single-run control with explicit
  cached-decode positions is restart EM=0.47 versus sequential EM=0.45 on 100
  hotpot_qa records. It invalidates the legacy restart wall but does not establish
  parity. Dynamic NTK scaling is inactive: maximum position 4,720 is
  below the 32,768 threshold.

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
| legacy, invalid cached decode | 0.45 | 0.06 | -0.39 |
| corrected explicit decode positions | 0.45 | 0.47 | +0.02 |

The legacy evaluator supplied restart positions during prefill but let HF infer
cached-decode positions from cache length. Its restart deficits, the claimed
hard wall, exposure-bias explanation, and RoPE-theta interpretation are
invalidated. The corrected control is bounded to this checkpoint, dataset, and
100-record subset; it does not establish a broader failure, success, or parity claim.

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
