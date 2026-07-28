# Paper section audit — post-EXP-02.4 alignment

Audit of `paper/sections/*.tex` against current experimental state
(RESEARCH_PLAN.md, EXP-01 findings in issue #33 comments, EXP-02.1–02.6
findings in issue #58 comments, CHANGELOG.md, README.md). One section per
`.tex` file; three-column table `claim | status | source of truth`.
`current` = matches evidence; `stale` = needed rewriting;
`unsupported` = kept but marked `\todo{}` for future EXP.

**Δ note.** The only Δ established in the store is on `composition` for
Qwen2.5-7B-Instruct at k=2: Δ_f1 = 0.0147, Δ_em = 0, Δ_presence = 0,
Δ_loss = 1.914 (from EXP-01.1 / issue #34). Every "within Δ" claim
implicitly references this margin unless otherwise noted. Per-dataset Δ
for hotpot/musique/ddi_stress/kilt_nq is a small follow-up.

---

## 00_intro.tex

| claim | status | source of truth |
|---|---|---|
| RAG prompt structure and $O(T^2)$ prefill cost | current | §1 of RESEARCH_PLAN; unchanged |
| Cross-doc attention as unmodeled dependency | current | §1 of RESEARCH_PLAN |
| Cache invalidation as production cost, prefix caching only helps on exact prefix repeat | current | §1 of RESEARCH_PLAN, unchanged |
| Bridge-width $k$ parameterizes the mask family | current | §6 axis M, §2.3 |
| Δ is fixed from full baseline's seed variance | current | RESEARCH_PLAN §4 "non-inferiority margin" |
| "how much cross-document attention can be removed at each bridge width, on parity-detection benchmarks" | **stale** — did not acknowledge that parity is conditional (single-hop only) per EXP-02.1 | rewritten to note conditional parity |
| "co-headline: cache hit skips prefill entirely" | **stale** — caching is now load-bearing (sole end-to-end wall-clock win), not merely co-headline, because C3 is structural on sdpa per EXP-02.3 | rewritten to "load-bearing efficiency result" |
| "degradation on questions that require composing evidence across documents" | **stale** — needed the family-dependence finding (needle 78%, aggregation 12%) from EXP-02.4 | rewritten to include the family-dependence span |
| "If one variant — `ddi_strict` — turns out to be non-viable" | **stale** — hypothetical framing; EXP-01.2 / EXP-02.1 confirm `ddi_strict` collapses to blind on the primary instruct model on composition | rewritten to state the result, not hypothesise it |

---

## 01_related_work.tex

| claim | status | source of truth |
|---|---|---|
| FiD encodes passages independently, fuses in decoder | current | canonical citation |
| PCW = independent context windows in decoder-only LM, training-free | current | canonical citation |
| CEPE = trained adapter for parallel context encoding | current | canonical citation |
| Sparse Transformers / Longformer / BigBird use position-fixed sparsity, not document-boundary sparsity | current | canonical |
| StreamingLLM sinks; $N$ sinks compete under block-diagonal | current — diagnostic prediction, not measured yet | canonical; §2 |
| PagedAttention as practical caching alternative, exact-prefix-only | current | §1, §2 |
| LitM protocol for controllable-$N$ evaluation | current | canonical |
| Differentiators: 7–8B decoder-only, bridge-width sweep, analytical/realized gap reported | current — all three are still true of the current results | §12 verdicts |

No edits needed. Fully current.

---

## 02_method.tex

| claim | status | source of truth |
|---|---|---|
| Block-diagonal mask parameterized by which query rows bridge | current | §2 |
| Mask taxonomy T1 (full, ddi_qf, ddi_strict, ddi_bridge_k, blind) | current | §6 axis M; matches `ddi.masks.build_mask` |
| `ddi_bridge_k` as pivot with $k=1$ ≡ strict, $k=Q$ ≡ qf | current | RESEARCH_PLAN §6 |
| Cost equations for full / ddi_qf / ddi_strict | current | RESEARCH_PLAN §7; matches `phase3_timing` analytical column |
| "As $Q \to 0$ speedup approaches $N$"; $Q \cdot N\ell$ term dominates for short docs | current | RESEARCH_PLAN §7 |
| Analytical vs realized gap is itself a measurement target | current | RESEARCH_PLAN §7, §12 |
| Two position layouts (sequential, restart) definitions | current | §6 axis P; matches `ddi.positions.build_positions` |
| "The restart layout … is untested in the prior work" | current | still true |
| "unlocks the co-headline result of Section 6" | **stale** — reframed as load-bearing; also silent on the asymmetry that restart hurts `full` badly (from EXP-01/#37) | rewritten: added asymmetry paragraph + noted quality-free under ddi_qf, order-invariant to fp precision |
| Δ = $k \cdot \sigma_{full}$ computed via `ddi-delta`, committed alongside baseline | current | RESEARCH_PLAN §4; `ddi.delta.compute_delta`; #34 |

---

## 03_experimental_design.tex

| claim | status | source of truth |
|---|---|---|
| Model selection criteria (shared arch, two families, public reproducibility) | current | RESEARCH_PLAN §5 |
| Primary models = Qwen2.5-7B and Llama-3.1-8B, base+instruct | current as design, but not run: only Qwen has quality rows | RESEARCH_PLAN §5; EXP-02.2 base-model, EXP-01 Qwen-instruct |
| "Qwen2.5-1.5B as ablation … 14B for scale trend … YandexGPT third-family" | **stale** — none of these were run; kept in table for design continuity but added `\todo{}` noting current-results scope | rewritten with `\todo{}` scope note |
| bf16, greedy decoding, seed control, batch size 1 | current | matches CHANGELOG entries |
| Dataset table: "Parity detection: NQ, TriviaQA, HotpotQA … Cross-doc: MoNaCo, 2WikiMultiHop, MuSiQue … Comparability: PCW/FiD dataset … General: Open RAG Bench" | **stale** — actual loaded datasets differ (composition, ddi_stress with 5 families, hotpot_qa, musique, kilt_nq, synthetic_qa, nq); TriviaQA/MoNaCo/Open RAG Bench/LitM never landed | rewritten table + `\todo{}` for missing loaders |
| "NQ, TriviaQA, HotpotQA: low seed variance, well-known headroom" as parity-detection | **stale** — HotpotQA is a multi-hop benchmark, not a parity instrument. NQ-Open has no retrieved passages so every mask collapses to full (EXP-01 finding). | rewritten: split into single-hop parity (KILT-NQ, synthetic_qa), no-doc smoke (NQ-Open), multi-hop (HotpotQA, MuSiQue), synthetic stress (composition, ddi_stress) |
| Six design axes definitions (M, T, P, N, L, K) | current | §6 |
| "Four discipline rules" | **stale** — undercount; the referenced issue #1 has ≥5 (provenance, store-backed, one mask, ≥3 seeds, Δ); the seed rule (ground rule 5) was omitted | rewritten to Five, with ground rule 5 explicitly listed |
| "any DDI variant must be numerically identical to `full` at $N=1$" | current — encoded as Phase 0.5 gate #4, still holds | matches `tests/test_gates.py` and CHANGELOG entry |

---

## 04_results_quality.tex

| claim | status | source of truth |
|---|---|---|
| Train-free sweep as primary result, ≥3 seeds, mask swapped on stock weights | current | §8 Phase 2.1; EXP-02.1 achieved n=3 on 4 datasets |
| "If the median gap to `full` is smaller than Δ … train-free result alone supports the parity claim" | current — this is the design rule | RESEARCH_PLAN §4, §8 |
| "Any cell whose train-free row is inside the Δ band carries the parity claim" | **stale** — silent on the fact that multi-hop cells are NOT inside Δ, and that the fine-tuned sweep therefore becomes load-bearing rather than confirmatory | rewritten: added conditional-parity paragraph with numeric F1 recovery clusters (18–28% multi-hop) + `\todo{}` noting fine-tuned rows not yet run |
| Bridge-width curve F2 traces knee | current — F2 renderer exists, phase2_bridge runner exists; EXP-01.3 confirmed non-overlapping IQRs at endpoints (k=1 vs k=26) | RESEARCH_PLAN §8 Phase 2.3; #36 |
| F6 quality/speed Pareto frontier is the paper's central diagram | current as design; F6 renderer exists in `ddi.reports.phase2_pareto` | CHANGELOG 2026-07-27 realization ratio entry; RESEARCH_PLAN §10 |
| "Instruction tuning trains the model to route context through the query region … `ddi_qf` costs instruct models less than base" | **stale** — measured direction is the **opposite**: base Qwen retains more of `full`'s F1 under every mask than instruct does (65% vs 27% on musique under ddi_strict, EXP-02.2) | rewritten to state the observed sign flip, keep the hypothesis as ex ante, add `\todo{}` for Llama-side which was not run |

---

## 05_results_efficiency.tex

| claim | status | source of truth |
|---|---|---|
| FTT / TPOT / end-to-end latency, CUDA-event timed, ≥5 repeats, warmup discarded, variance reported | current | RESEARCH_PLAN §9; `ddi.timing.measure` |
| T3 tabulates analytical vs realized vs realization ratio | current — T3 renderer exists (`ddi.reports.phase3_realization`) | CHANGELOG 2026-07-27 realization ratio entry |
| Realized speedup = realized / theoretical, "1.0 = fully translated" | current | §9, §7 |
| F3 plots realization ratio vs $N$, one line per $\ell$ | current — F3 renderer exists | CHANGELOG |
| "A gap between analytical and realized is expected … first-class result" | current | §7, §12 |
| Candidate list of gap causes (launch overhead, memory bandwidth, $Q\cdot N\ell$ dominance, non-prefill bottleneck) | current — the list is still the paper's framing | §7 |
| "We report prefill-only AND end-to-end numbers" | current as design; T3 renderer supports both | CHANGELOG |
| No statement of what the gap actually measures at | **stale** — the section previously read as if the gap size were open, but EXP-02.3 measured it as structural on sdpa (realization drops from 0.98× at N=1 to 0.061× at N=25 while sparsity climbs to 93.6%); FA2 wiring is broken (#70), so kernel-fix half is deferred | added "Measurement to date: structural on sdpa" paragraph with concrete numbers + `\todo{}` for FA2 |

---

## 06_results_caching.tex

| claim | status | source of truth |
|---|---|---|
| Under full causal, no doc's KV block is reusable | current | §1, §2 |
| Under block-diagonal + restart, doc KV block is a pure function of the doc | current | §2, §6 axis P |
| Cache hit skips prefill entirely | current | §4 C4 |
| "co-headline with the compute result" / "survives even if realization gap is structural" | **stale** — no longer merely co-headline; it is the *sole* end-to-end wall-clock win, because the compute-side realization gap **is** structural on sdpa (EXP-02.3) | rewritten to "load-bearing" and "sole wall-clock efficiency win", removed hypothetical framing |
| "Quality under restart (Phase 4.1) confirms restart does not cost quality" | **stale** — this was framed as a future check; result exists (EXP-01/#37): restart is free under ddi_qf (F1 0.131 vs 0.122; presence 0.883 vs 0.917 on composition) and harmful under full (F1 −58%, presence −43pp) | rewritten with measured numbers + asymmetry |
| "Per-doc cache correctness (Phase 4.2) verifies logits identical within bf16 tolerance" | **stale** — result exists (EXP-01/#39): kv_deterministic = 1.0 and kv_cache_roundtrip = 1.0 across every doc×record | rewritten to state result: byte-stable at bf16 |
| "Order invariance (Phase 4.3) verifies permutation produces identical outputs" | **stale** — result exists (EXP-01 #39 + EXP-02.5 #63): bf16 median max-delta 0.312 collapses to fp32 0.017, matching 28-layer fp32 accumulation floor; binary `atol=1e-3` fails in both dtypes because that threshold is unreachable at any finite precision | rewritten with bf16→fp32 collapse + `\todo{}` noting atol=1e-3 should be dropped from any table |
| "At any non-trivial reuse rate the per-document line is expected to dominate the prefix-caching line" | **stale** — F5 was measured (EXP-02.6 #64) with three modes: `ddi_cache` (3.84 QPS, 256 ms, 99% hit), `full_causal` (2.18 QPS, 459 ms), `ddi_recompute` (2.00 QPS, 500 ms). Prefix-caching line was NOT run; the actual F5 has ddi_recompute in that slot | rewritten with measured numbers, replaced "prefix caching" with "ddi_recompute" per what was actually run, added nuance that ddi_recompute is not a clean compute-vs-caching control at these shapes |
| F5 caption says "prefix caching" as one of the modes | **stale** — the rendered F5 does not include a prefix-caching mode; runner had `ddi_cache`/`ddi_recompute`/`full_causal` | rewritten caption |
| "If bridge width requires wide bridge, caching still holds" | current — mask type doesn't affect doc block cacheability | still true |
| "If realization gap does not close, cache hits still skip prefill" | **stale** — reframed from conditional ("if") to statement of fact ("since the realization gap is structural on the tested backend") | rewritten |

---

## 07_failure_envelope.tex

| claim | status | source of truth |
|---|---|---|
| Testable prediction: cross-doc composition should degrade under block-diagonal | current | §4 C5 |
| Every parity dataset admits a single-hop / multi-hop split | current — `ddi.hop_labels` implements this; T5 renderer registered | CHANGELOG 2026-07-27 T5/F4 entry |
| "Single-hop is where parity holds; multi-hop is where DDI degrades" | current as directional prediction, but silent on **family dependence within multi-hop** (needle 78% vs aggregation 12% — 6.5× spread) | rewritten to add per-family recovery numbers + `\todo{}` for multi-seed per-family + base-model rerun |
| T5 table hop-split per mask | current — renderer exists, empty rows render as `n/a` | CHANGELOG |
| F4 grouped bar chart single/multi | current — renderer exists | CHANGELOG |
| PCW baseline "we evaluate PCW on at least one shared dataset with matched retrieval" | **stale** — `phase5_pcw` runner exists in the harness but the paired sweep at matched shapes has not been run against `ddi_qf`; also PCW-in-DDI-terms is exactly `ddi_qf + restart`, which is more concrete than "positional partitioning without an explicit mask family" | rewritten to identify PCW = ddi_qf + restart in DDI's family + `\todo{}` for missing paired sweep |
| Composition probe with distance/pair_position, T6 + F7 | current — dataset + runner + renderers exist | CHANGELOG 2026-07-27 composition probe entry; #28 |
| "Monotone degradation in the compositional axis is the signature we look for" | current as framing; EXP-01.4 found no clear monotonic decline over distance {0..3} but bucket sizes 10–21 so not a strong null | matches; kept as prediction |

---

## 08_limitations_conclusion.tex

| claim | status | source of truth |
|---|---|---|
| Batch size 1 limitation, FA2 varlen would lift it | current | RESEARCH_PLAN §11; CHANGELOG |
| "Primary claims rest on 7–8B parameter models" | **stale** — primary quality claims land on **Qwen2.5-7B only**; Llama-3.1-8B rows not produced | rewritten paragraph: model-scale AND family coverage; added base/instruct sign flip observation |
| "Ablation model at 1.5B and scale model at 14B give one dimension of scale evidence" | **stale** — 1.5B / 14B not run | rewritten to state scope honestly |
| "If the analytical/realized gap does not close … co-headline caching carries proportionally more" | **stale** — the "if" is answered: gap is structural on sdpa; caching carries the full wall-clock win, not just "proportionally more" | rewritten as measured, not conditional |
| No task-family limitation | **stale** (missing) — the parity claim scopes to single-hop; multi-hop misses Δ and family-dependence is 6.5× within multi-hop | added new paragraph "Task-family coverage of the parity claim" |
| Conclusion: characterization framing, bridge-width sweep is central artifact | current | §11, §12 |
| Conclusion: "If `ddi_strict` turns out to be non-viable, that is the ablation" | **stale** — is → was; the result is in | rewritten from conditional to statement of measured result |
| Conclusion: "value of the paper does not depend on any variant beating a baseline" | current | §12 |
| Conclusion silent on multi-hop split verdict | **stale** — added acknowledgment that the honest reading is split-verdict, not blanket parity | rewritten |

---

## Preprint scaffold (preprint.tex)

Added `\todo{...}` command macro so audit-introduced inline stubs compile
without adding a new package. Distinct from `\todoreport{...}` (which
stands in for missing tables/figures).
