# VIBEQUEUE-66 — Optimized-path serving boundary: preregistration

Frozen **before any V66 timing run**. The machine-readable twin of this document is
`experiments/v66_boundary_preregistration.json`; the sweep runner
(`scripts/run_v66_boundary.py`) pins both files' sha256 in its run manifest, written
before the model loads. No threshold, grid, or rule below may change after results are
seen; any forced deviation is recorded in `results/v66_boundary_<UTC>/README.md` under
`deviations` at run time, never by editing this file.

## Motivation

V59 measured the reuse-vs-monolithic crossover on the **unoptimized** reuse path
(all-hit ~221 ms; hit rates 0.616–0.733 at L=512). V65 optimized the canonical reuse
path to ~62.3 ms all-hit TTFT — 2.05x **faster** than the monolithic baseline
(byte-identical outputs) — so the true crossover moved and is unknown. V66 measures
where optimized reuse wins as a function of hit fraction `h`, document length `L`, and
decode budget, on development data only.

## Arms (frozen)

- **A (monolithic)** — fresh encode+decode every request on the shared SDPA model,
  exactly the V65 arm-A execution: mask built untimed via `build_mask(...,"ddi_qf")`,
  TTFT = the synchronized prefill `model()` call wall (`generate_ids(...)["prefill_ms"]`
  semantics), wall = synchronized wall around the full fixed-work greedy
  `generate_ids(model, tokenizer, batch, "ddi_qf", d, stop_on_eos=False)`.
- **B (optimized V65 reuse)** — `CanonicalCache(model, contract, DocCache store)`
  (`ddi/canonical_cache.py` V65-optimized path: 2 GiB LRU, store-time validation
  markers, staged H2D), LRU-warm steady state. TTFT = synchronized wall of
  `cache.request(docs, query_ids)` with `max_new_tokens=None` (documents resolution +
  assembly + query forward; the V65 arm-C convention). Wall(d) = synchronized wall of
  `cache.request(docs, query_ids, max_new_tokens=d, eos_token_id=None)`.

Both arms run on the same loaded SDPA model in one process (`--canonical-on-sdpa`
execution, V64 audit qualifier (c)). Per-request wall includes each arm's own mask
build; the known asymmetries are inherited from the V65 harness and frozen: (i) arm-A
TTFT excludes mask build, arm-B TTFT includes it (this is the convention behind the
V65 2.05x anchor); (ii) at fixed budget `d` the canonical decode loop performs `d`
step forwards vs the monolithic loop's `d-1` (one step result is discarded by
`generate_tokens_with_doc_cache`). Both asymmetries are constant per cell and favor
neither arm's *comparison across h*; they are reported here so no post-hoc
reinterpretation is possible.

## Grid (frozen)

- Hit fraction `h ∈ {0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}`, realized as an explicit
  cache pre-state: `hit_count = min(N, max(0, floor(h·N + 0.5)))` (round-half-up) of
  the request's `N` documents are pre-stored in the cell's DocCache before the timed
  section; the hit documents are the **first `hit_count` documents in request order**.
  The remaining `N − hit_count` documents are fresh-encoded on every pass. Achieved
  fraction `h_eff = hit_count/N` is reported alongside nominal `h`. Where two nominal
  `h` values map to the same `hit_count` (possible at N=2 and N=5), the cell is
  **measured once and shared**; the crossover table lists every nominal `h` with its
  `hit_count` and `h_eff`.
- `L ∈ {128, 512}` tokens per document. `L=32` is **conditional**: it runs only if the
  elapsed wall time after both main L blocks is below 105 minutes (see Cost cap).
- `N ∈ {2, 5, 10}` documents per request.
- Decode budget `d ∈ {1 ("TTFT-only"), 8, 16, 32, 64}` new tokens, fixed-work greedy
  (no EOS stop on either arm: `stop_on_eos=False` / `eos_token_id=None`), so every
  repeat of a cell does identical work.

Cell = (L, N, hit_count, d). Measured units: 16 (L,N,h) units per L (3 unique
hit_counts at N=2, 5 at N=5, 8 at N=10) × 5 budgets. The `d=1` budget and the TTFT
metric are measured in the same per-unit TTFT pass structure (below).

## Data (frozen)

Development records only, from the frozen V62 split manifest
(`experiments/v62_split_manifest.json`, sha256
`5ad3761106ecaed967b28d21c3919fa1869ebf70bf0e178c1e773a14bf8ee098`): the 100
`development` record ids. The 100 confirmation ids and every other eval set are
**forbidden**; the runner asserts the content pool is disjoint from the confirmation
arm and the V60 24 before anything runs. Seed **47** for content selection.

Because real development paragraphs are short (median 116 tokens, max 570; only 3 of
998 reach 512 tokens), exact-L requests are built by a **preregistered packing rule**
over real dev content (no synthetic tokens):

- Paragraph pool: every `info_contexts` paragraph of the 100 dev records, records in
  ascending id order, paragraphs in record order; tokenized with
  `add_special_tokens=False` (the `encode_segments` convention).
- Document packing: for each (L, N), a cursor starts at pool position 0. Each of the
  N request documents accumulates whole consecutive paragraphs (token-level
  concatenation) until its length is `>= L`, then is truncated to **exactly L
  tokens**; the cursor advances past every consumed paragraph (the truncated
  paragraph's remainder is discarded, never reused). The cursor resets to 0 for each
  (L, N), so document sets are nested across N within an L.
- Query: one fixed dev record selected by `random.Random(47)` over the sorted dev ids;
  query text = `build_query_text(tokenizer, record)`, tokenized
  `add_special_tokens=False` (~64–85 tokens).
- The request (documents + query, in order) is **fixed per (L, N) and byte-identical
  across arms**; arm A's batch is assembled from the same token streams with
  `build_positions(doc_lengths, query_len, 0, "sequential")` — the exact V65 arm-A
  batch semantics (`inference.encode` default layout), verified by unit test against
  `inference.encode` on text inputs.

## Repeats and order (frozen)

Per (L, N): one prewarming pass encodes the missing hit documents into the cell store
(measured separately, reported separately — never inside any timed statistic). Per
(L, N, hit_count) unit: a fresh `CanonicalCache` (fresh 2 GiB LRU), then **3 warmup +
15 timed repeats**, **interleaved by repeat index**: warmups run (A,B) ×3, then timed
repeats run A_i, B_i for i = 1..15. Each B pass is followed by an **untimed prune** of
its fresh write-through store entries (`documents()` writes fresh docs through), so
every pass starts from exactly the frozen `hit_count` pre-state; the store file set is
verified against the expected hit keys before and after every timed section. Decode
budgets run in the fixed order: TTFT unit (d=1 + TTFT passes), then d = 8, 16, 32, 64.

Per repeat the runner records: arm, wall ms, TTFT ms (A: internal synchronized
`prefill_ms`; B: the separate `max_new_tokens=None` pass), generated token ids sha256,
byte-identity verdict, per-arm CUDA peak allocated/reserved bytes within the
repeat-block window, hit document indices/keys, and store file counts. TTFT repeats:
the same 3 + 15 structure (A's `generate_ids(d=1)` pass yields A-TTFT and wall(1); B's
`request(max_new_tokens=None)` pass yields B-TTFT; B's `request(max_new_tokens=1)`
pass yields B-wall(1)).

Reported per cell: **median** of each arm's 15 timed values and the **paired
percentile bootstrap 95% CI (10,000 resamples, seed 47)** of the ratio of medians
**A/B** (speedup; > 1 means reuse faster), resampling repeat-index pairs — the same
estimator as `scripts/run_v65_benchmark.paired_ratio_bootstrap`.

## Materiality rule (frozen, no post-hoc tuning)

Reuse **wins** a cell iff the A/B ratio 95% CI **lower bound > 1.05**; **loses** iff
the CI **upper bound < 1/1.05**; otherwise **tie**. Applied identically to wall and
TTFT ratios.

## Identity checks (frozen)

Per cell, per repeat: arm-A and arm-B generated token id sequences must be
**byte-identical** (this covers the documents served from cache; V65 verified A vs C
byte-identity on all 24 frozen records, so any V66 divergence is anomalous). Any
divergence **pauses the sweep**: the runner appends a diagnostic record (cell id,
repeat index, both id sequences, hit indices) to the progress log and exits non-zero
for diagnosis. Divergences are never averaged away and never silently retried.

## Metric definitions (frozen)

- **TTFT** — time to first token: arm A = synchronized prefill `model()` wall
  (`prefill_ms`, mask build untimed); arm B = synchronized wall of the full
  `CanonicalCache.request(..., max_new_tokens=None)` call (documents + assembly +
  query forward). Decode-budget-independent; measured once per (L, N, hit_count) and
  reported for every decode budget of that unit.
- **wall(d)** — full fixed-work greedy generation of exactly d new tokens,
  synchronized wall around the whole arm call (includes mask build / assembly).
- **prewarming / cold-store cost** — the untimed per-(L, N) hit-store encode wall
  (and the h=0 units' fresh-encode cost visible in their B walls), reported
  separately from all steady-state statistics.

## Cost estimate and hard cap (frozen)

Counted cells: 8 nominal h × 2 L (3 if L=32) × 3 N × 5 budgets × 2 arms = 480 nominal
arm-cells (720 with L=32), realized as 32 (L,N,h) units × 5 budgets × 2 arms ≈ 6,400
timed passes plus warmups (≈ 7,700 passes total; ~2,900 of them carry ≥8 decode
steps). Per-pass cost from prior artifacts: monolithic prefill 0.03–0.6 s
(V61 grid; scales with N·L), reuse TTFT 0.02–0.25 s (V65: 62 ms at ~1.2k tokens),
cold canonical pass 0.05–1.2 s (V60/V61), decode ≈ 24–26 ms/step (V59/V65). Estimated
total: **1.5–2.2 h** of GPU wall. **HARD CAP: 3 h.** Enforcement, frozen in advance:

- the runner refuses to start a new (L, N, hit_count) unit after 165 min elapsed;
- if measured pace projects the remaining grid past the cap, timed repeats for
  **remaining** units shrink 15 → 10 (floor 10) — recorded as a deviation; if still
  over, L=32 is dropped — recorded as a deviation; the **h grid is never shrunk**;
- `L=32` runs only if, after both main L blocks complete, elapsed < 105 min.

## Infeasible cells (frozen)

OOM or any infeasible cell is marked `status: "infeasible"` with the exception text in
the per-cell summary and the crossover table, and the sweep continues. OOM recovery:
`torch.cuda.empty_cache()` before the next unit.

## Smoke allowance (frozen)

Before the sweep, one mechanics-validation smoke cell (L=128, N=2, hit_count=2, d=8,
2 repeats) may run to validate harness plumbing. Its timings are **discarded** and are
not part of any reported statistic; only its pass/fail is recorded.

## Checkpointing (frozen)

All results land in `results/v66_boundary_<UTC>/`. The runner appends one JSON line
per completed unit-cell to `progress.jsonl` (**append-only**) and per-pass rows to
`raw_repeats.jsonl`; a restarted run skips unit-cells already marked `ok` in
`progress.jsonl` (unit granularity re-run otherwise).

## Analysis plan (frozen)

Per (L, decode budget): crossover table (`crossover_table.md`) over nominal h with
hit_count, h_eff, medians, CIs, verdicts; the **measured crossover h** = the smallest
nominal h in the frozen grid whose wall verdict is *win* (and the same for TTFT);
"none in grid" if no cell wins. Figures (`crossover_figure*.png|pdf`, matplotlib):
median wall speedup A/B vs h with bootstrap-CI bands and materiality markers (win/tie/
lose), horizontal lines at 1.0, 1.05 and 1/1.05; the same for TTFT. Deliverables in
the results dir: `crossover_table.md`, figures, `summary.json`, `README.md`
(provenance, prereg shas, deviations), `scoping_sentence.md` (the updated paper
scoping sentence replacing "high-hit-rate" with the measured boundary), and a
ready-to-paste `t9_followup_fragment.tex` + `t9_followup_evidence_fragment.json` for
the paper's evidence/render pipeline (the pipeline itself is pinned and is not
modified by this task).

## Provenance

`sha256` of this file and of `experiments/v66_boundary_preregistration.json` are
recorded below and pinned in the sweep manifest before the model loads.

- sha256(`experiments/v66_boundary_preregistration.json`) =
  `b6654e8e88b24b18d8076a388a9c72ed8a5ef3d1049cb382ed201fa5017afe3b` (revision 3,
  after Amendments 1–2; revision 1 was
  `674578fb7d93a24cc56644e1d027d975be2093f610cee4457660d494c02e0b1a`, revision 2
  `b9c978332769f9b05445058376997b399df86b867d9d033db3d12c5318ac761c`)
- sha256(this file) = recorded in the sweep manifest (a file cannot contain its own
  digest); revision 1 of this file was
  `566ca972b5cb1751702afbcb5b4964bf58c31cfcef57e68db152e8fd9e1796e8`

## Amendment 2 (2026-09-10, at the identity pause of cell L512_N10_H10_D8, before any further timed data)

The frozen identity gate paused the sweep at cell `L512_N10_H10_D8` repeat 0 (all-hit,
largest shape): arm A and arm B agreed for the first three generated tokens and then
diverged. Diagnosis (recorded in the progress log with both id sequences): the two
arms are **mathematically equivalent executions with identical restart positions and
masks** (post-Amendment 1) but different kernel structure — one joint bf16 SDPA
prefill over the 5280-token `[docs][query]` sequence versus per-document 512-token
prefills whose KV blocks are assembled and attended by the query forward. Different
kernel tiling gives different bf16 accumulation order; at 5200 document tokens the
logit differences are large enough that a near-tie greedy argmax flips
mid-trajectory. This is the same phenomenon class V65's byte-identity gate implicitly
tested for at ~1200 tokens (24/24 there); it is a genuine, shape-dependent property
of bf16 reuse, not a harness error. Both arms perform identical fixed work either
way, so wall/TTFT timing validity is unaffected by which of two near-tie tokens is
chosen.

Amendment: after this diagnosis, the sweep **continues in record mode** — per-repeat
identity verdicts are recorded per cell (with divergent sequences kept in the
progress log), divergent cells are flagged `identity_divergent`, and the final
report scopes the byte-identity claim to the cells where it holds (exact counts
reported). No grid, budget, repeat, threshold, metric or materiality rule changes;
the materiality analysis never depended on identity. Runner flag:
`--continue-on-identity-divergence`. The paused cell re-runs from scratch in record
mode on resume (bf16 kernels are deterministic for fixed shapes/inputs on this GPU,
so its divergence will reproduce and be recorded rather than silently skipped).

## Amendment 1 (2026-09-10, before any timed sweep data — smoke-triggered harness repair)

The preregistered smoke cell (declared discardable above) paused on a byte-identity
failure at repeat 0. Diagnosis: the preregistered arm-A batch used
`build_positions(..., "sequential")` (the V65 harness default, behind the V65
calibration anchor), while the canonical reuse path uses `build_positions(...,
"restart")`. Under `restart` the query block starts at `max(doc_lengths)`, not at
`doc_total`, so query→document RoPE relative distances differ between the two layouts
by each document's start offset: they are **numerically different executions of the
same mask family**, and greedy trajectories diverge on some content. The V65 24/24
A-vs-C byte-identity was an empirical robustness fact on those 24 records, not layout
invariance; the V60/V65 runners' `"layout": "sequential"` records confirm the
harness convention.

Amendment: arm A's batch positions switch to `build_positions(doc_lengths,
query_len, 0, "restart")` so both arms execute the same numerics and the per-repeat
byte-identity gate verifies what it was frozen to verify — that the reuse path
returns exactly the monolithic result. No grid, budget, repeat, threshold, metric or
materiality rule changes. Timing semantics are unchanged: position IDs do not alter
prefill/decode kernel cost, so continuity with the V65 timing anchor (A ≈ 127.5 ms,
all-hit ≈ 62.3 ms, 2.05x) is preserved. The frozen JSON's `data.request_fixity` and
`arms.A_monolithic.execution` texts are updated accordingly (`prereg_revision: 2`);
the revision-1 JSON digest is preserved above and in the sweep manifest. No timed
sweep data existed when this amendment was written.
