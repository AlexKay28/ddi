# DDI — research plan toward an arXiv preprint

**Status:** design document, not a paper. This is the pre-registration: what we claim, what we
must run to earn the claim, and which models / methods / setups go into the comparison grid.
Everything here is either (a) a measured number with a source, or (b) explicitly marked
`PENDING`. Nothing in between.

Companion to the repo write-up in [`../README.md`](../README.md).
Working tree: `/home/alexkay28/alexkay28/tickets/ALIGNMENT-2445`. Ticket: ALIGNMENT-2445.

---

## 1. Research problem

A RAG prompt is a concatenation of retrieved documents followed by the dialog:

```
[doc_1][doc_2] ... [doc_N][dialog / query][answer]
```

Standard causal attention makes prefill quadratic in the length of the *whole* concatenation.
But the retrieved documents are mutually independent artifacts — they were fetched by a
retriever, they have no discourse relationship, and `doc_7` attending into `doc_2` encodes a
dependency that does not exist in the data. We are paying `O(T²)` to model interactions that are
mostly noise.

Two costs follow from this, and the second is the expensive one in production:

1. **Compute.** Cross-document attention is the dominant term once `N` is large.
2. **Cache invalidation.** Because `doc_2`'s keys and values depend on `doc_1`, no document's KV
   block is reusable. Change one document in the retrieved set — or even its position — and the
   entire prefix must be recomputed. A document that appears in a thousand requests is prefilled a
   thousand times.

**The question this paper answers:** how much cross-document attention can we delete before
answer quality degrades, and what do we actually get back when we delete it?

---

## 2. Resolution idea

Replace the lower-triangular mask with a **block-diagonal** one. Each document is causal within
itself and blind to every other document. The dialog region is the only place where information
is allowed to mix.

```
full causal                 DDI-QF                      DDI-strict
(baseline)                  docs isolated,              docs isolated,
                            query sees all              query isolated
d1 ██                       d1 ██                       d1 ██
d2 ████                     d2   ██                      d2   ██
d3 ██████                   d3     ██                    d3     ██
 q ████████                  q ████████                   q       ██
 a ██████████                 a ██████████                 a ██████████
                                                         ^ single SEP row bridges
```

Two consequences, and we should be honest that only the first is a contribution of *this* paper:

- **Compute.** Cost drops from `O(T²)` to `O(Σ Lᵢ² + Q·T)`. Under equal document lengths this is
  a factor of `N` — see §7.
- **Cacheability.** Each document's KV block is now a pure function of that document alone. It
  becomes cacheable and reusable across requests. **This is the real prize**, and it is the part
  we have not yet measured at all. It also opens a design axis that full causal attention
  forecloses: if each document's positions restart at 0, blocks become position-independent and
  reusable regardless of retrieval order (§6, axis P).

### Prior art we must position against, not rediscover

This idea is not new in encoder-decoder land, and a reviewer will say so immediately. We cite and
differentiate:

| work | relation to DDI |
|---|---|
| **Fusion-in-Decoder** (Izacard & Grave, EACL 2021) | encodes passages independently, fuses in decoder. DDI is the decoder-only analogue. |
| **Parallel Context Windows** (Ratner et al., ACL 2023) | independent context windows in a decoder-only LM, training-free. **Closest prior work — must be a baseline, not just a citation.** |
| **CEPE** (Yen et al., ACL 2024) | parallel context encoding with a trained adapter. |
| Longformer / BigBird / Sparse Transformers | sparsity patterns fixed by position, not by document boundary. |
| StreamingLLM (Xiao et al., 2023) | attention sinks; explains why the SEP-row variant might have been expected to work. |
| PagedAttention / prefix caching | the *practical* alternative to DDI for the caching claim. Must be discussed or the caching contribution is unmotivated. |

Our differentiators, stated plainly: (i) decoder-only, at 7–8B, with and without fine-tuning;
(ii) a measured characterization of *how much* isolation is affordable, not a binary works/doesn't;
(iii) the theory-vs-realized speedup gap, which the prior work does not report.

---

## 3. What we already know

**Trustworthy.** One epoch of Qwen2.5-7B fine-tuning per mask, best validation loss, from
`v2/logs/`. Masks built by `v2/fine_tune.py:create_learning_sample()` using real token spans:

| mask | best val loss | Δ vs baseline | final train loss |
|---|---|---|---|
| `full` (baseline) | **0.2355** | — | 0.0170 |
| `block-sparsed-query-full` (DDI-QF) | 0.2773 | **+0.042** | 0.0178 |
| `block-sparsed` (DDI-strict) | 1.4801 | **+1.245** | 0.0543 |

**The finding that shapes the whole paper:** document↔document isolation is nearly free. Query
isolation is catastrophic. Whatever a single SEP bridge row can carry, it is not enough — a
6× loss increase. DDI-strict is a *negative result* and we report it as one; it is the ablation
that makes DDI-QF meaningful.

**Weakly supported.** `v2/README.md` reports 1.20× prefill (720 ms → 600 ms) at 13 contexts,
91.6% sparsity. Single configuration, single run, no variance, unverified by re-run. Analytically
that configuration should give ~10.6× (§7). **A 9× gap between theory and measurement is the most
scientifically interesting thing we have** and the microbenchmarks in
`notebooks/flasattention_tests*.py` (per-block launch overhead, "actual vs. theoretical speedup")
already point at why.

**Not evidence at all.** No generation-quality numbers (EM / F1 / answer-presence) survive from
any run. And note: `v3/v4/infer_model.py` computes `documents_sizes_in_tokens` with `len()` on a
`[1, L]` tensor, so every document is treated as one token long — the v3/v4 DDI inference path has
been inert. **Any quality number collected through v3/v4 serving is void.** The v2 fine-tuning
numbers above are unaffected (different code path, correct spans).

---

## 4. Claims, and what would falsify each

| # | claim | evidence needed | status |
|---|---|---|---|
| C1 | Document↔document isolation costs little quality | val loss + EM/F1 across ≥2 model families, ≥2 datasets | partial (1 model, 1 dataset, loss only) |
| C2 | Query isolation is not viable | same grid, DDI-strict arm | partial (1 model, loss only) |
| C3 | DDI reduces prefill cost, scaling with `N` | latency sweep over `N`, with variance | 1 point, 9× below theory |
| C4 | DDI makes per-document KV caching possible | cache-reuse throughput experiment | **not started** |
| C5 | DDI degrades specifically on cross-document reasoning | multi-hop vs single-hop split | **not started** |

C5 is a prediction of *failure*, and we should chase it deliberately. If documents cannot see each
other, a question whose answer requires composing `doc_2` with `doc_7` must be answered in the
dialog region alone. The honest version of this paper measures that boundary and reports it as the
method's operating envelope. If we do not, a reviewer will find it for us.

---

## 5. Models to compare

Selection criteria: shared architectural family (decoder-only, RoPE, GQA) so the mask logic
transfers unchanged; two independently-trained families so C1 is not a Qwen artifact; a scale
axis; and — important for arXiv — **primary claims must rest on publicly reproducible weights.**

| model | role | why | in repo? |
|---|---|---|---|
| `Qwen/Qwen2.5-7B` + `-Instruct` | **primary** | already the workhorse; all v2 results are here | yes |
| `meta-llama/Llama-3.1-8B` + `-Instruct` | **primary** | independent family; rules out Qwen-specific effects | yes |
| `Qwen/Qwen2.5-1.5B` | ablation sweeps | cheap enough to run the full mask × N × ℓ grid | no — add |
| `Qwen/Qwen2.5-14B` (or 32B) | scale trend | does the affordable-isolation gap shrink or grow with scale? | no — add |
| `yandex/YandexGPT-5-Lite-8B` (+ `-pretrain`, `-instruct`) | secondary / appendix | third family, but **not externally reproducible** | yes |

Two deliberate decisions:

- **Base *and* instruct for both primaries.** v1 already split these. The interesting question is
  whether instruction tuning has already taught the model to route information through the query
  region — if so, DDI-QF should cost instruct models less than base models. Cheap to test, and it
  is a mechanism claim rather than a benchmark number.
- **YandexGPT moves to the appendix.** It cannot be downloaded by a reviewer. Keeping it in the
  main tables makes the central claims unverifiable; keeping it in an appendix as a third-family
  confirmation is strictly better for the paper.

Fixed across all runs: bf16, `batch_size=1` with gradient accumulation (v3/v4 `generate` hard-fails
above 1), greedy decoding for quality metrics, temperature as-configured for any sampling runs.

---

## 6. Methods and setups to implement

Six axes. We are **not** running the full cross-product — §8 prioritizes.

**Axis M — mask** (the independent variable)
1. `full` — causal baseline
2. `DDI-QF` — docs isolated, whole query block sees all docs ← *the proposal*
3. `DDI-strict` — docs isolated, query isolated, single SEP bridge row
4. `DDI-bridge-k` — generalizes 2↔3: last `k` prefix rows see everything. `k=1` is strict,
   `k=Q` is QF. **New, and the most informative single sweep we can run** — it turns the
   0.2355 / 0.2773 / 1.4801 step function into a curve and locates the knee.
5. `blind` — all-zero mask control, from `v1/evaluate_models__blind.py`. Non-negotiable: if a
   model scores well seeing nothing, the metric is not measuring retrieval.

**Axis T — how the mask is introduced**
- `train-free` — swap the mask at inference on a stock checkpoint (v2 patched path)
- `finetuned` — mask present during SFT, so the model adapts to it (v2 `fine_tune.py`, v4 `train.py`)

This 2× axis is the paper's second-most-important comparison. Train-free is what practitioners can
deploy today; fine-tuned is the achievable ceiling. The gap between them is a result.

**Axis P — position encoding** *(new, and the one that unlocks C4)*
- `sequential` — documents occupy consecutive absolute positions (current behavior)
- `restart` — every document's RoPE positions start at 0

`restart` is what makes a KV block genuinely reusable: order-independent and cache-keyable by
document hash alone. `notebooks/KV-cache-and-position-analysis.ipynb` already built the probe for
this. Untested and high-value.

**Axis N — document count:** `{1, 3, 5, 9, 13, 25}`. `N=1` is a correctness check — DDI and full
causal must be *numerically identical* there. Any divergence at `N=1` is a bug, and it is the
cheapest regression test we have.

**Axis L — document length:** `{128, 512, 2048}` tokens. Drives the launch-overhead story: small
blocks are where FlashAttention varlen should lose to a dense kernel.

**Axis K — kernel:** dense SDPA + explicit `-inf` mask (reference semantics) · FlashAttention-2
varlen (`v2/utils/flash_attention_sparse.py`) · Triton block-diagonal
(`notebooks/triton_block_attention.py`). Three implementations of *one* mathematical object, so
they double as each other's correctness tests.

### Baselines that are not DDI variants

- **Parallel Context Windows** — closest prior work, decoder-only, training-free. Must be
  implemented and run, or C1's novelty is not defensible.
- **Prefix caching** — the practical alternative for the caching claim. Can be cited-and-discussed
  rather than run, but must be addressed.
- **Retrieval-order shuffle** — same documents, permuted. Full causal is order-sensitive;
  `DDI-QF + restart` should be provably order-*invariant*. A clean, cheap, striking result.

---

## 7. Cost model — drives the sweep design

With `N` documents of length `ℓ`, query block `Q`, total `T = Nℓ + Q`, counting attended pairs:

```
full      ≈ T² / 2
DDI-QF    ≈ N·ℓ²/2  +  Q·Nℓ  +  Q²/2
DDI-strict≈ N·ℓ²/2  +  Q²/2  +  T
```

As `Q → 0` the speedup is exactly `N`. At `ℓ = 512, Q = 64`:

| N | T | theoretical speedup | sparsity | measured |
|---|---|---|---|---|
| 1 | 576 | 1.00× | 0% | — (must be 1.00, correctness check) |
| 3 | 1600 | 2.59× | 61.4% | `PENDING` |
| 5 | 2624 | 4.19× | 76.1% | `PENDING` |
| 9 | 4672 | 7.39× | 86.5% | `PENDING` |
| 13 | 6720 | **10.59×** | 90.6% | **1.20×** (v2/README.md, n=1) |
| 25 | 12864 | 20.19× | 95.0% | `PENDING` |

The 90.6% at `N=13` is consistent with the 91.6% reported in `v2/README.md` (differences from
variable real document lengths). So the sparsity accounting is sound and the **gap is entirely in
realization, not in the arithmetic**: 10.59× available, 1.20× delivered.

That gap is a section of the paper, not an embarrassment. Hypotheses to test: per-block kernel
launch overhead at small `ℓ`; the varlen path being memory-bound rather than FLOP-bound; the query
block's `Q·T` term dominating once `N` is small; prefill not being the bottleneck at these lengths
at all. Axis L × axis K is designed to separate these.

---

## 8. Experiment grid and phasing

Phase 1 is the minimum viable paper. Later phases strengthen it.

**Phase 0 — unblock (must precede everything)**
- Fix `documents_sizes_in_tokens` in `v3/v4/infer_model.py` (`len()` → `.shape[-1]`). Until this
  lands, the serving path produces no valid DDI measurement.
- Add the `N=1` DDI ≡ full-causal numerical equivalence test.
- Cross-check the three kernels (axis K) agree on logits for one fixed input.

**Phase 1 — core quality claim (C1, C2)**
Grid: `{Qwen2.5-7B, Llama-3.1-8B} × {base, instruct} × {full, DDI-QF, DDI-strict, blind} ×
{train-free, finetuned}` on 2 datasets. 64 quality runs.
Metrics: val loss, EM, token-F1, answer-presence.

**Phase 2 — the bridge curve (C1, sharpened)**
`DDI-bridge-k` for `k ∈ {1, 2, 4, 8, 16, 32, Q}` on Qwen2.5-7B. Locates the knee between 0.2773
and 1.4801. This is the paper's most quotable figure and it is one model, one sweep.

**Phase 3 — efficiency (C3)**
Axis N × L × K, ≥5 repeats with variance, warmup discarded. Report realized *and* theoretical on
the same axes. This is where the 9× gap gets explained or bounded.

**Phase 4 — caching and order-invariance (C4)**
`restart` positions + per-document KV cache. Metrics: cache hit rate, end-to-end throughput under
a realistic document-reuse distribution, and quality-under-`restart` (does it cost anything?).
Plus the retrieval-order shuffle invariance check.

**Phase 5 — the failure envelope (C5)**
Split every dataset into single-document-answerable vs. requires-cross-document-composition.
Report DDI-QF degradation separately per split. MoNaCo is multi-hop and is the natural probe.

### Datasets

| dataset | role | status |
|---|---|---|
| Open RAG Bench (`vectara/open_ragbench`) | primary, RAG-native | sourced, `datasets/revision.ipynb` |
| MoNaCo (`allenai/MoNaCo_Benchmark`) | multi-hop → C5 probe | sourced |
| Natural Questions / HotpotQA / 2WikiMultiHop | comparability with FiD & PCW literature | `PENDING` — add |

The third row matters more than it looks: without at least one dataset the prior work also reports
on, we cannot place DDI-QF against PCW's published numbers.

---

## 9. Metrics

**Quality:** validation loss (answer tokens only, `-100` elsewhere) · exact match · token-F1 ·
answer-presence (from `notebooks/results_analysis*.ipynb`) · degradation vs. `full` as the
headline, since absolute scores are dataset-specific.

**Efficiency:** FTT (prefill, CUDA-event timed) · TPOT · realized speedup vs. `full` ·
theoretical speedup from §7 · **realization ratio** = realized / theoretical, the number that
carries the gap story · mask sparsity · peak GPU memory.

**Caching (phase 4):** KV cache hit rate · prefill FLOPs saved · end-to-end QPS under document reuse.

Every efficiency number needs ≥5 repeats, discarded warmup, and reported variance. The existing
1.20× has none of these and cannot go in a paper as-is.

---

## 10. Figures and tables to produce

| # | artifact | source | status |
|---|---|---|---|
| T1 | mask taxonomy (mask × what-sees-what) | §2 | ready |
| T2 | quality: model × mask × train-free/finetuned | phase 1 | `PENDING` |
| T3 | theoretical vs. realized speedup vs. `N` | §7 + phase 3 | 1 of 6 rows |
| F1 | mask diagrams (full / QF / strict / bridge-k) | hand-drawn, TikZ | ready |
| F2 | **bridge curve**: val loss vs. bridge width `k` | phase 2 | `PENDING` |
| F3 | realization ratio vs. `N`, one line per `ℓ` | phase 3 | `PENDING` |
| F4 | quality degradation: single-hop vs. multi-hop | phase 5 | `PENDING` |
| F5 | throughput vs. document reuse rate | phase 4 | `PENDING` |

F2 is the paper's centerpiece: one model, one sweep, and it converts the three-point step function
we already have into a mechanism.

---

## 11. Paper outline, mapped to phases

1. **Introduction** — cross-document attention as unmodeled dependency; cache invalidation as the
   production cost. *(written)*
2. **Related work** — FiD, PCW, CEPE, fixed-pattern sparsity, prefix caching. *(§2 table)*
3. **Method** — mask family, bridge width `k` as the unifying parameter, cost model. *(§2, §7)*
4. **Experimental design** — models, datasets, axes, protocol. *(§5, §6, §8)*
5. **Results** — quality (phase 1–2), efficiency (phase 3), caching (phase 4).
6. **When DDI fails** — the cross-document reasoning envelope (phase 5). *Own section, not a
   limitations footnote.*
7. **Limitations** — batch size 1; 7–8B only; realization gap unresolved if phase 3 does not close it.
8. **Conclusion**

---

## 12. Honest assessment of where we stand

One model, one dataset, one epoch, loss-only, on two of five claims. One efficiency datapoint,
9× below its own theory, `n=1`. The headline caching benefit is unmeasured. The serving path had a
bug that voided its measurements.

What we *do* have is unusually clean for this stage: a sharp positive (isolation is affordable at
+0.042 loss), a sharp negative (query isolation is not, +1.245), and a quantified mystery (10.59×
available, 1.20× realized). Those three facts are a paper's spine. Phases 0–2 are perhaps a week of
GPU time and would make C1 and C2 defensible; phase 3 is what makes the efficiency claim honest
rather than aspirational.

Recommended framing for the preprint: **a characterization study**, not a method-wins paper. "How
much cross-document attention can be removed, and what does removing it actually buy" is a claim we
can fully support and is more useful to the field than a benchmark table. It also means the
negative results are contributions rather than things to explain away.
