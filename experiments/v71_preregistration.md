# V71 Preregistration — Trained-Contract Serving Boundary

**Frozen:** 2026-09-12, before any timed measurement and before the runner is
hash-pinned. **Ticket:** VIBEQUEUE-71. **Design:** `experiments/v71_trained_serving_design.md`.

## Question

Does the V70 trained k=5 link-token reuse path keep the audited warm-path
serving win against the original monolithic baseline, and where does its
crossover boundary sit relative to the untrained boundary measured in V66?

## Frozen grid and data

Identical to V66, unchanged: frozen V62 development split (split-manifest
SHA `5ad3761106ecaed967b28d21c3919fa1869ebf70bf0e178c1e773a14bf8ee098`;
V60-manifest exclusion SHA `2dffebf5d04065021dfb2b9e694651b34f0ebfae4f3d4b9aabd672908aa6cb07`),
hit fractions `(0.0,0.25,0.5,0.6,0.7,0.8,0.9,1.0)` realized as hit counts
(round-half-up, clamped), L ∈ {128, 512} plus conditional L=32, N ∈ {2,5,10},
decode budgets d=1/TTFT then 8,16,32,64, one query record drawn with
`random.Random(47)` from the dev IDs, paragraphs packed to exact lengths by
the frozen `pack_documents`. Development data only; confirmation IDs forbidden.

## Arms

| Arm | Weights | Execution |
|---|---|---|
| A | original untrained restart checkpoint (V66 pin `00df4c47…`) | monolithic full-causal prefill + fixed-work greedy decode, restart positions (V66 Amendment 1 semantics, `generate_ids`) |
| BP | V70 `checkpoint_k5` (aggregate SHA `04fe4dcb…`) | trained-contract fresh: per-pass independent canonical encode of all N documents (in-memory), k=5 fixed-marker layout prefill over links+query, fixed-work greedy decode (`generate_independent_ids`, `eos_token_id=None`) |
| CP | V70 `checkpoint_k5` | trained-contract cached: per-pass document entries from a per-unit `CanonicalCache`+`DocCache` store; mixed cells write through misses; k=5 layout prefill and decode identical to BP |
| DP | V70 `checkpoint_k5` | trained full-causal no-link monolithic prefill (`monolithic_forward(..., full_causal=True, use_cache=True)`), **TTFT only, D1 block only, diagnostic** |

BP and CP share the executed callable (`generate_independent_ids` /
`independent_forward_from_entries`); they differ only in entry provenance
(fresh encode vs store read/write-through). The store format, metadata and
keys are the V66/V70 `DocCache` format.

## D1–D4 resolutions

- **D1 (warm-up):** per unit, an untimed prebuild pass encodes all N unit
  documents through the trained model into the unit store; the store is then
  reset to exactly the `hit_count` pre-state (`reset_store`) and verified
  (`verify_store`) after every block.
- **D2 (mixed hits):** CP misses inside a timed pass are encoded on the
  trained model through the same `CanonicalCache` write-through path used by
  V66's B, and pruned untimed after every pass; per-pass miss-encode counts
  are recorded. All-hit cells fail closed on any encode attempt.
- **D3 (BP scope):** BP runs in every block of every cell.
- **D4 (repeats):** 3 shared warmups + 15 timed repeats per block; cap
  pressure may shrink later units to 10 repeats (frozen shrink rule, V66
  semantics); 10 is the floor.

## Timing semantics

- A wall: synchronized wall of `generate_ids(model_a, tokenizer_a, batch,
  "ddi_qf", d, stop_on_eos=False)`; A TTFT = its `prefill_ms` (mask build
  untimed). A model and tokenizer are the untrained checkpoint's.
- BP/CP wall: synchronized wall of entries-provenance + `generate_independent_ids(...)`;
  BP/CP TTFT (D1 block only): synchronized wall of entries-provenance +
  `independent_forward_from_entries` + last-logit argmax. Layout build and
  entry assembly are inside the timed callable for BP/CP (V66 `request()`
  parity). BP-TTFT prices the trained fresh prefill including its own
  per-pass encode; the A/BP TTFT ratio is descriptive.
- DP TTFT: synchronized wall of `monolithic_forward(..., full_causal=True,
  use_cache=True)` including its layout build (declared diagnostic; not
  semantically identical to A's `prefill_ms`).
- Repeat order (frozen), D1 block: A, CP-TTFT, [prune], CP-wall(1), [prune],
  BP-TTFT, BP-wall(1), DP-TTFT. Decode blocks: A, CP-wall(d), [prune],
  BP-wall(d). Warmups run the full pass set with prunes.

## Gates

- **Primary (per cell, per budget):** A/CP paired ratio of medians
  (`paired_ratio_bootstrap`, 10,000 resamples, seed 47, CI 0.95) with the
  frozen V66 materiality rule (win: CI low > 1.05; lose: CI high < 1/1.05;
  else tie). Deliverable: the crossover table (smallest nominal h with a
  CI-supported win) and its diff against the V66 table.
- **Secondary sanity (per repeat, every block):** CP and BP generated ids
  byte-identical. Any divergence pauses the sweep (abort) unless
  `--continue-on-identity-divergence` is passed after diagnosis; this gate
  has priority over primary verdicts.
- **Descriptive (no gates):** A/BP ratios (trained-fresh pricing), DP TTFT
  ratio vs A TTFT, per-pass miss-encode counts, first-token agreement between
  A and BP/CP (expected to diverge across weights; recorded, never gated).
  Cross-run comparisons against V66's stored walls are descriptive only.

## Caps and stop rules

Scaled from V66 for the added arms: hard cap 300 min, unit cutoff 270 min,
L=32 gate 170 min, infeasible-cell handling identical to V66 (OOM records
`infeasible` and continues). Stop on NaN/nonfinite outputs, checkpoint
identity mismatch, store pre/post-state mismatch, or an unexplained identity
divergence. No mid-run protocol edits; amendments only between stages.

## Checkpoint and code pins

- A checkpoint dir SHA (V66 pin): `00df4c47d67660cab8292da313ea133290b0bb320437290d725cd149c3d31de9`
- CP/BP/DP checkpoint: `results/v70_arms_valid_seed47/checkpoint_k5`,
  aggregate SHA `04fe4dcb3c70f0ef75f0cdfbf1f99164a90ab54df0833cb4b14cc56bd723ba87`
  (verified against the V70 Amendment 3 pin; re-verified at manifest time).
- The runner (`scripts/run_v71_trained_serving.py`) pins this prereg JSON's
  SHA in-code and refuses a mismatch; the manifest records the runner's own
  SHA and the `ddi/` source tree hashes before the model loads.
- Environment: repo `.venv`, torch 2.13.0+cu126, transformers 4.57.6, one
  A100 80GB, `CUDA_VISIBLE_DEVICES=0`, GPU idle check ≥70% free.

## Analysis

`scripts/analyze_v71_boundary.py` consumes `result.json` + V66's
`results/v66_boundary_20260910T132922Z/result.json` and emits: per-cell
verdict grid, crossover table (TTFT and each decode budget), and the V66→V71
boundary diff. No thresholds are computed at analysis time; every verdict is
runner-emitted under the frozen rule.

## Amendments

To be appended below, timestamped, only between stages.

## Amendment 1 — Serving Path Rebuilt After Harness-Overhead Finding (2026-09-12, pre-data)

The first sweep attempt (20 L=128 cells completed into
`results/v71_trained_serving/`) measured BP/CP ratios dominated by harness
overhead, not model work: `build_request_layout` costs ~51 ms of pure Python
at prefix 729 and `layout_tensors` ~43 ms, and the V70 harness rebuilds both
on every call **and every decode step**; measured CP TTFT at the all-hit
L128/N5 cell (122.9 ms vs arm-A 69.6 ms) is explained almost entirely by this
overhead (model forward ≈ 30 ms). Those cells are **discarded**; no verdicts
are carried. The sweep restarts into a fresh output directory.

The BP/CP/DP timed callables are replaced by precomputed-structure execution
(`scripts/v71_serving_paths.py`): the request layout, its mask/position
tensors, the per-decode-step mask rows and the DP full-causal structures are
built once per unit **untimed** — mirroring arm A's untimed batch construction
(V66 semantics: A's TTFT excludes mask build). The executed model calls
receive identical input_ids, attention-mask values, position_ids and
cache_position as the V70 harness path (prefix rows are budget-independent;
asserted in `tests/test_v71_serving_paths.py` and re-verified on the real
model by a fast-vs-V70-harness cross-check inside the smoke, which gates the
sweep). The CP≡BP byte-identity gate and all grid/gate/cap rules are
unchanged. This amendment changes timing harness only — no treatment, arm
definition, grid, threshold or analysis change.
