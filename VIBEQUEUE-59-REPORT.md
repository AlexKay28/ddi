# VIBEQUEUE-59: Persisted Cache Cost, TTFT, and Crossover

## Decision

Persisted document-KV reuse has a sharp geometry crossover. It is not beneficial for 32-token
documents, is marginal or demanding at 128 tokens, and is materially faster at 512 tokens.
This performance result does not reopen V58's failed exact-equivalence and QA gates.

## Primary result

- Attempt: `results/v59_ttft_grid_20260908T131148Z/`
- Model: restart-trained Qwen2.5-7B, bf16 SDPA on one A100 80GB.
- Grid: `N={2,5,10}`, `L={32,128,512}`, `Q=64`, three seeds, two warmups and ten
  retained repeats per arm and cell.
- Scope: time to first token through host-materialized argmax; decode excluded.
- Audit: pass for all 27 cells, exact input reconstruction, stage sums, hit states, formulas,
  source/patch digests, and 1,968,387,534 retained cache bytes.
- Fresh and warm first-token IDs agreed in every cell. This is diagnostic only and does not
  supersede V58's natural-QA failures.

| N | L | Fresh ms | Warm ms | Cold ms | Warm speedup mean [95% CI] | Hit-rate crossover | Reuse crossover |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 32 | 23.3 | 32.8 | 89.9 | 0.716 [0.706, 0.731] | none | none |
| 2 | 128 | 38.7 | 37.6 | 110.8 | 1.017 [0.994, 1.032] | not robust | not robust |
| 2 | 512 | 103.1 | 55.8 | 228.0 | 1.850 [1.824, 1.864] | 0.733 [0.720, 0.757] | 3.61 [3.55, 3.73] |
| 5 | 32 | 29.2 | 46.0 | 191.3 | 0.633 [0.604, 0.652] | none | none |
| 5 | 128 | 67.4 | 59.5 | 252.1 | 1.147 [1.107, 1.200] | 0.955 [0.939, 0.966] | 23.59 [16.46, 29.93] |
| 5 | 512 | 253.9 | 110.0 | 579.3 | 2.254 [2.138, 2.349] | 0.700 [0.691, 0.711] | 3.09 [3.02, 3.21] |
| 10 | 32 | 40.2 | 67.9 | 378.5 | 0.590 [0.579, 0.600] | none | none |
| 10 | 128 | 122.7 | 91.2 | 470.5 | 1.363 [1.328, 1.406] | 0.913 [0.903, 0.922] | 11.66 [10.37, 12.84] |
| 10 | 512 | 591.5 | 223.5 | 1176.0 | 2.635 [2.604, 2.662] | 0.616 [0.613, 0.622] | 2.37 [2.35, 2.41] |

Fresh, warm, and cold columns are medians across the three per-seed medians. Speedup and
crossover intervals are paired seed bootstraps. With only three seeds, the intervals describe
run-to-run stability on this host rather than broad hardware uncertainty.

## Cost breakdown

Warm cache cost has four material components: persisted `torch.load`, H2D plus layer-wise K/V
concatenation, query mask/position preparation, and the query forward. Mean stage milliseconds:

| N | L | Disk load | H2D + assembly | Query prep | Query forward |
|---:|---:|---:|---:|---:|---:|
| 2 | 32 | 5.2 | 3.0 | 0.7 | 23.9 |
| 2 | 512 | 17.5 | 11.5 | 2.2 | 25.3 |
| 5 | 128 | 19.9 | 11.8 | 1.7 | 26.4 |
| 5 | 512 | 48.2 | 29.0 | 11.9 | 26.2 |
| 10 | 128 | 38.9 | 22.7 | 3.1 | 25.8 |
| 10 | 512 | 94.6 | 54.5 | 48.5 | 31.8 |

The fixed query forward is about 24-32 ms. At `L=32`, disk and assembly overhead exceed the
document projection saved by reuse. At `L=512`, fresh document projection dominates and warm
reuse wins despite 29-149 ms of load and assembly overhead. Query preparation itself becomes
material at `N=10,L=512` because the dense additive mask spans a 5,184-token prefix.

Serialized K/V storage is approximately 57.4-58.0 KiB per document token, independent of N.
The full retained grid occupies 1.97 GB. Warm lookup measurements include the host page-cache
state produced by repeated persisted reads; they are not remote-object-store estimates.

## Crossover interpretation

- The reported hit-rate crossover uses the frozen linear mixture of observed all-cold and
  all-warm TTFT. It is an analytical estimate, not a measured partial-hit curve.
- At 512 tokens, reuse becomes favorable around 62-73% hit rate and after roughly 2.4-3.6
  identical uses including one build.
- At 128 tokens, the required hit rate is 91-96% and amortization needs about 12-24 uses; the
  `N=2` cell is not robust because one seed has warm TTFT slower than fresh.
- At 32 tokens there is no feasible hit rate or reuse count because even all-hit warm reads are
  slower than fresh recomputation.

The secondary partial-hit sweep was not run: the primary grid answers the ticket's crossover
question, while V58 already rejects the quality-preserving deployment claim.

## Verification

- `scripts/audit_cache_ttft.py results/v59_ttft_grid_20260908T131148Z`: pass
- Scoped Ruff and `git diff --check`: pass
- Full suite: 587 passed, 28 skipped.

No production cache behavior, defaults, or paper evidence was changed.

## Addendum 2026-09-09: measured partial-hit, decode, and memory campaign

The secondary partial-hit sweep and the missing totals were subsequently measured
and audited; all artifacts pass `scripts/audit_cache_ttft.py` with zero token
mismatches. Tracker ticket VIBEQUEUE-59 documents each slice; summary:

| Artifact | Adds |
|---|---|
| `results/v59_partial_n2_l512_20260909T081500Z/` | Actual 0/1/2-hit TTFT: 222.8 / 151.7 / 74.1 ms. |
| `results/v59_partial_n5_l512_20260909T082000Z/` | Actual 0..5-hit TTFT: 536.9 down to 158.6 ms. |
| `results/v59_partial_decode16_n2_l512_20260909T083000Z/` | Fixed-16 decode: TPOT ~23.9 ms in all arms; request totals 486.7/618.5/440.4 ms; all-hit TTFT 1.813x but request total only ~1.105x. |
| `results/v59_partial_decode16_n5_l512_20260909T084000Z/` | Same at N=5: all-hit TTFT 2.167x, request total ~1.269x. |
| `results/v59_partial_natural16_n2_l512_20260909T085000Z/` | Natural decode diagnostic (non-arm-comparable by design). |
| `results/v59_memory_n2_l512_20260909T090500Z/` | Peak memory: ~15.4-15.7 GB allocated, 16.0 GB reserved across arms. |
| `results/v59_partial_n10_l512_20260909T091500Z/` | Actual 0..10-hit TTFT: 1233.5 down to 353.7 ms; all-hit 2.47x. |
| `results/v59_partial_n2510_l128_20260909T093000Z/` | L=128 with hit states: N=2 regresses (0.941x), N=5/10 need 92-96% hits. |
| `results/v59_partial_n2510_l32_20260909T094000Z/` | L=32: reuse never wins (0.603-0.728x); no feasible crossover in any cell. |

The runner gained measured partial-hit schedules, paired instrumented/control
accounting with residuals, fixed-work decode (`--decode-length`), a natural
decode diagnostic (`--natural-decode-max-length`), and per-repeat peak-memory
capture, all schema-compatible with the retained v1 artifact. The verdict above
is unchanged and now measurement-backed.
