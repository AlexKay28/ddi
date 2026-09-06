# DDI — research plan toward an arXiv preprint

**Status:** historical design document, not a paper or a statement of current
evidence. It records the pre-registration: what the project intended to claim,
what it would need to run, and the planned comparison grid. Current claim
boundaries are in `AUDIT_NOTES.md`, `preprint.tex`, and `../README.md`; this
plan contains superseded targets and must not be cited as empirical support.
In particular, its powered quality-expansion and larger-model cells are retained
as a reopenable VIBEQUEUE-48 plan, but are deferred optional future work rather
than prerequisites for the bounded characterization paper.

Companion to the repo write-up in [`../README.md`](../README.md) and the operational roadmap in
[issue #1](https://github.com/AlexKay28/ddi/issues/1) (source of truth for phase structure).

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
full causal                 ddi_qf                      ddi_strict
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

**Nothing, as evidence.** See [issue #1](https://github.com/AlexKay28/ddi/issues/1): the `v1`–`v4`
exploratory code is being replaced by one clean implementation, and every number below is demoted
from *result* to *hypothesis to re-test*. They stay listed because they are the right things to
test and useful as **sanity targets** for the new harness — not because they are findings.

**Hypotheses H1, H2.** One epoch of Qwen2.5-7B fine-tuning per mask, best validation loss, from
`v2/logs/`. `n=1`, one model, one dataset, no seed control, no provenance:

| mask | best val loss | Δ vs baseline | final train loss |
|---|---|---|---|
| `full` (baseline) | **0.2355** | — | 0.0170 |
| `block-sparsed-query-full` (ddi_qf) | 0.2773 | **+0.042** | 0.0178 |
| `block-sparsed` (ddi_strict) | 1.4801 | **+1.245** | 0.0543 |

**The hypothesis that shapes the whole paper:** document↔document isolation is nearly free. Query
isolation is catastrophic. Whatever a single SEP bridge row can carry, it is not enough — a
6× loss increase. If it survives re-evaluation, ddi_strict is a *negative result* and we report it
as one; it is the ablation that makes ddi_qf meaningful.

**Hypothesis H3.** `v2/README.md` reports 1.20× prefill (720 ms → 600 ms) at 13 contexts,
91.6% sparsity. Single configuration, single run, no variance, unverified by re-run. Analytically
that configuration should give ~10.6× (§7). **A 9× gap between theory and measurement is the most
scientifically interesting thing we have** and the microbenchmarks in
`notebooks/flasattention_tests*.py` (per-block launch overhead, "actual vs. theoretical speedup")
already point at why.

**Never existed.** No generation-quality numbers (EM / F1 / answer-presence) survive from any run.

**Why the rebuild, specifically.** Beyond `n=1` and missing provenance: the same mask semantics were
reimplemented six times across `v1`–`v4` and disagree in detail, so there is no single definition of
"the DDI mask" to point at. `v3`/`v4` additionally hand-roll the decoder loop — reimplementing RoPE,
GQA repeat and KV layout, which must exactly match HF's forward or every number is silently wrong —
and computed `documents_sizes_in_tokens` with `len()` on a `[1, L]` tensor, treating every document
as one token long. Any metric from that path is void. The clean implementation injects masks into
HF's own forward instead (issue #1, step 0.4), which makes the `N=1` equivalence test a test of the
mask rather than of a reimplementation of attention.

---

## 4. Claims, and what would falsify each

| # | claim | evidence needed | status |
|---|---|---|---|
| C1 | Document↔document isolation costs little quality | val loss + EM/F1 across ≥2 model families, ≥2 datasets, gap to `full` within `Δ` | **unverified** (H1 only) |
| C2 | Query isolation is not viable | same grid, `ddi_strict` arm | **unverified** (H2 only) |
| C3 | DDI reduces prefill cost, scaling with `N` | latency sweep over `N`, with variance; end-to-end and prefill-only both reported | **unverified** (H3, 9× below theory) |
| C4 | DDI makes per-document KV caching possible | cache-reuse throughput experiment; identical logits on cache hit | **not started** — **co-headline** |
| C5 | DDI degrades specifically on cross-document reasoning | multi-hop vs single-hop split | **not started** |
| C6 | `ddi_qf` is robust to distractor documents | quality vs. distractor count on the Lost-in-the-Middle-style set | **not started** — obtained "for free" alongside Phase 2.1 |

C5 is a prediction of *failure*, and we should chase it deliberately. If documents cannot see each
other, a question whose answer requires composing `doc_2` with `doc_7` must be answered in the
dialog region alone. The honest version of this paper measures that boundary and reports it as the
method's operating envelope. If we do not, a reviewer will find it for us.

### The non-inferiority margin `Δ`

C1 says "costs little quality" — that has no falsifier without a threshold. Enough seed noise hides
any real gap inside "couldn't detect a difference," which is not a claim. Before any DDI quality
number is interpreted, we measure `full`'s seed-to-seed variance (Phase 1) and fix `Δ` from that
alone. A DDI candidate satisfies C1 iff its gap to `full` is `< Δ` at the seed count used. Any
benchmark whose noise floor exceeds `Δ` cannot support C1 and moves to the failure-envelope side
(C5). This is [issue #1](https://github.com/AlexKay28/ddi/issues/1) ground rule 9, and it is what
turns C1 from an eyeballed judgment into a falsifiable test.

### C4 is a co-headline, not a nice-to-have

A per-document cache hit skips prefill entirely, which no prefill-kernel improvement under `full`
causal attention can match at any quality level. Under a parity-then-speedup framing, C4 is the
speedup that survives even if Phase 3's realization gap (§7) turns out to be structural rather
than fixable. Phase 4.1/4.2 (position restart + per-document cache) is part of the minimum viable
paper, not an appendix.

---

## 5. Models to compare

Selection criteria: shared architectural family (decoder-only, RoPE, GQA) so the mask logic
transfers unchanged; two independently-trained families so C1 is not a Qwen artifact; a scale
axis; and — important for arXiv — **primary claims must rest on publicly reproducible weights.**

| model | role | why | in repo? |
|---|---|---|---|
| `Qwen/Qwen2.5-7B` + `-Instruct` | **primary** | the model H1/H2 were measured on; keeps the sanity target comparable | yes |
| `meta-llama/Llama-3.1-8B` + `-Instruct` | **primary** | independent family; rules out Qwen-specific effects | yes |
| `Qwen/Qwen2.5-1.5B` | ablation sweeps | cheap enough to run the full mask × N × ℓ grid | no — add |
| `Qwen/Qwen2.5-14B` (or 32B) | scale trend | does the affordable-isolation gap shrink or grow with scale? | no — add |
| `yandex/YandexGPT-5-Lite-8B` (+ `-pretrain`, `-instruct`) | secondary / appendix | third family, but **not externally reproducible** | yes |

Two deliberate decisions:

- **Base *and* instruct for both primaries.** v1 already split these. The interesting question is
  whether instruction tuning has already taught the model to route information through the query
  region — if so, ddi_qf should cost instruct models less than base models. Cheap to test, and it
  is a mechanism claim rather than a benchmark number.
- **YandexGPT moves to the appendix.** It cannot be downloaded by a reviewer. Keeping it in the
  main tables makes the central claims unverifiable; keeping it in an appendix as a third-family
  confirmation is strictly better for the paper.

Fixed across all runs: bf16, greedy decoding for quality metrics, temperature as-configured for any
sampling runs.

Note on batching: the old `v3`/`v4` code hard-failed above `batch_size=1`, and the exploratory
results inherited that limit. The clean implementation should **not** — batched prefill with
per-sample block-diagonal masks is exactly what FlashAttention varlen is for, and lifting the limit
materially cuts the cost of the Phase 2–3 grids. If it stays at 1, that is a limitation to state,
not a constraint to inherit.

---

## 6. Methods and setups to implement

Six axes. We are **not** running the full cross-product — §8 prioritizes.

**Axis M — mask** (the independent variable)
1. `full` — causal baseline
2. `ddi_qf` — docs isolated, whole query block sees all docs ← *the proposal*
3. `ddi_strict` — docs isolated, query isolated, single SEP bridge row
4. `ddi_bridge_k` — generalizes 2↔3: last `k` prefix rows see everything. `k=1` is strict,
   `k=Q` is QF. **New, and the most informative single sweep we can run** — it turns the
   hypothesised 0.2355 / 0.2773 / 1.4801 step function into a curve and locates the knee.
5. `blind` — all-zero mask control. Non-negotiable: if a model scores well seeing nothing, the
   metric is not measuring retrieval.

**Axis T — how the mask is introduced**
- `train-free` — swap the mask at inference on a stock checkpoint
- `finetuned` — mask present during SFT, so the model adapts to it

This 2× axis is the paper's second-most-important comparison. Train-free is what practitioners can
deploy today; fine-tuned is the achievable ceiling. The gap between them is a result.

**Axis P — position encoding** *(new, and the one that unlocks C4)*
- `sequential` — documents occupy consecutive absolute positions (current behavior)
- `restart` — every document's RoPE positions start at 0

`restart` is what makes a KV block genuinely reusable: order-independent and cache-keyable by
document hash alone. Untested and high-value.

**Axis N — document count:** `{1, 3, 5, 9, 13, 25}`. `N=1` is a correctness check — DDI and full
causal must be *numerically identical* there. Any divergence at `N=1` is a bug, and it is the
cheapest regression test we have.

**Axis L — document length:** `{128, 512, 2048}` tokens. Drives the launch-overhead story: small
blocks are where FlashAttention varlen should lose to a dense kernel.

**Axis K — kernel backend:** dense SDPA + explicit `-inf` mask (reference semantics) ·
FlashAttention-2 varlen · Triton block-diagonal. Three implementations of *one* mathematical
object, so they double as each other's correctness tests.

### Baselines that are not DDI variants

- **Parallel Context Windows** — closest prior work, decoder-only, training-free. Must be
  implemented and run, or C1's novelty is not defensible.
- **Prefix caching** — the practical alternative for the caching claim. Can be cited-and-discussed
  rather than run, but must be addressed.
- **Retrieval-order shuffle** — same documents, permuted. Full causal is order-sensitive;
  `ddi_qf + restart` should be provably order-*invariant*. A clean, cheap, striking result.

---

## 7. Cost model — drives the sweep design

With `N` documents of length `ℓ`, query block `Q`, total `T = Nℓ + Q`, counting attended pairs:

```
full      ≈ T² / 2
ddi_qf    ≈ N·ℓ²/2  +  Q·Nℓ  +  Q²/2
ddi_strict≈ N·ℓ²/2  +  Q²/2  +  T
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

Phases 0 + 1 + 2 (especially 2.3) + 3.5 + **4.1/4.2** is the minimum viable paper — 4.1/4.2 is
part of the MVP, not an appendix (see C4 above). Later phases strengthen it. Phases 2, 3, 4 are
independent once 0 and 1 land. Full sub-issue breakdown in
[issue #1](https://github.com/AlexKay28/ddi/issues/1).

**Phase 0 — clean implementation and correctness gates (must precede everything)**
One `ddi/` package written from zero: a single mask builder, kernel backends behind one interface,
masks injected into HF's own forward (not a hand-rolled decoder loop), provenance capture, and a
results store that tables generate from. Three required gates: `N=1` DDI ≡ full-causal equivalence,
backend cross-check on fixed input, and a document-length regression test.

**Phase 1 — datasets, metrics, `full` baseline, fix `Δ`**
Load and freeze the datasets in the table below with committed split hashes. Implement metrics
(§9) with unit tests. Run `full` on each primary model with ≥3 seeds — **this is where `Δ` is
fixed**, from `full`'s seed-to-seed variance alone, before any DDI number is looked at. `Δ` is
committed alongside the baseline it was derived from and becomes the parity threshold for C1.

**Phase 2 — mask family quality (C1, C2)**
Four sub-phases, ordered by cost:

- **2.1 Train-free sweep — the primary result.**
  `{Qwen2.5-7B, Llama-3.1-8B} × {base, instruct} × {full, ddi_qf, ddi_strict, blind}`, mask swapped
  at inference on stock weights, ≥3 seeds, on the parity-detection datasets. If the gap to `full`
  is within `Δ`, this alone supports C1 and 2.2 becomes confirmatory. Diagnostic if it doesn't:
  every document gets its own first-token attention sink under block-diagonal masking, so `N`
  sinks compete instead of one — cheap check is `ddi_qf` with one shared global sink token, still
  train-free.
- **2.2 Light-adaptation and fine-tuned sweep — conditional.**
  Runs only if 2.1 leaves a gap exceeding `Δ`. Two rungs, cheapest first: a few hundred adaptation
  steps or LoRA (separates "needs a nudge" from "information destroyed"); then full SFT with the
  mask present throughout. The train-free vs. light vs. fine-tuned gap is itself a result.
- **2.3 Bridge-width curve — the centerpiece.**
  `ddi_bridge_k` for `k ∈ {1, 2, 4, 8, 16, 32, 64, Q}` on Qwen2.5-7B, ≥3 seeds. Pair each point
  with its realized speedup from Phase 3 to plot the **quality/speed Pareto frontier** (F6) with
  the `Δ` margin and the parity point marked. This is the POC's actual deliverable.
- **2.4 Base vs instruct mechanism probe.**
  Hypothesis: instruction tuning already routes context through the query region, so `ddi_qf`
  costs instruct models less than base models. Effect size with variance, or an explicit null.

**Phase 3 — efficiency (C3)**
Axis N × L × K, ≥5 repeats with variance, warmup discarded. Report realized *and* theoretical on
the same axes. **Phase 3.5** is where the 9× gap gets explained or bounded, with end-to-end
latency reported alongside prefill-only speedup for every cell — a prefill-only win can dilute to
near-nothing end-to-end.

**Phase 4 — caching and order-invariance (C4) — co-headline**
- **4.1 Position-restart variant.** Quality under `restart` vs. `sequential` — does restarting
  cost anything?
- **4.2 Per-document KV cache.** Keyed by document content hash. A cache hit must yield logits
  identical to cold compute within tolerance.
- **4.3 Retrieval-order shuffle.** `ddi_qf + restart` should be provably order-invariant.
- **4.4 Throughput under document reuse.** End-to-end QPS vs. full-causal recompute and vs.
  prefix caching as the practical alternative.

**Phase 5 — the failure envelope (C5)**
Split every dataset into single-document-answerable vs. requires-cross-document-composition and
report every Phase 2 cell separately per split. Needs the labels from issue #1 §1.2. Also runs the
Parallel Context Windows baseline on ≥1 shared dataset and a synthetic composition probe with
distance/position varied.

### Datasets

Split by function — a parity-detection benchmark and a failure-envelope benchmark are different
instruments (issue #1 §1.1):

| purpose | dataset | status |
|---|---|---|
| Parity detection (headroom, low seed variance; C1) | NQ, TriviaQA, HotpotQA | `PENDING` — load and freeze |
| Controllable `N` + gold-document position | Lost-in-the-Middle-style multi-doc QA (Liu et al. 2023) | `PENDING` — add |
| Cross-document stress (C5) | MoNaCo (`allenai/MoNaCo_Benchmark`); optionally 2WikiMultiHop, MuSiQue | MoNaCo sourced; the rest `PENDING` |
| Comparability with prior work | ≥1 dataset that PCW / FiD report on (likely NQ or TriviaQA) | overlap with row 1 |
| General RAG-native | Open RAG Bench (`vectara/open_ragbench`) | sourced |

Every dataset used as a parity instrument needs `full`'s seed variance measured and confirmed
smaller than `Δ` in Phase 1 before it is trusted for C1. Without the comparability row, we cannot
place `ddi_qf` against PCW's published numbers.

---

## 9. Metrics

**Quality:** validation loss (answer tokens only, `-100` elsewhere) · exact match · token-F1 ·
answer-presence · degradation vs. `full` as the headline, since absolute scores are
dataset-specific. All implemented fresh with unit tests; the old notebook definitions are worth
consulting but not lifting.

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
| T2 | quality: model × mask × train-free (+ finetuned if 2.2 triggers) | phase 2.1 (+ 2.2) | `PENDING` |
| T3 | theoretical vs. realized speedup vs. `N` | §7 + phase 3 | `PENDING` (H3 is not a filled cell — every measurement predates the correctness gates) |
| F1 | mask diagrams (full / qf / strict / bridge-k) | hand-drawn, TikZ | ready |
| F2 | **bridge curve**: val loss vs. bridge width `k` | phase 2.3 | `PENDING` |
| F3 | realization ratio vs. `N`, one line per `ℓ` | phase 3 | `PENDING` |
| F4 | quality degradation: single-hop vs. multi-hop | phase 5 | `PENDING` |
| F5 | throughput vs. document reuse rate | phase 4.4 | `PENDING` |
| F6 | **quality/speed Pareto frontier** with `Δ` and parity point marked | phase 2.3 × phase 3 | `PENDING` |

F6 is the POC's actual deliverable — the bridge sweep paired with realized speedups, not a single
"same quality" number. F2 is F6 collapsed to quality alone; both come from the same phase 2.3 run.

---

## 11. Paper outline, mapped to phases

1. **Introduction** — cross-document attention as unmodeled dependency; cache invalidation as the
   production cost. *(written)*
2. **Related work** — FiD, PCW, CEPE, fixed-pattern sparsity, prefix caching. *(§2 table)*
3. **Method** — mask family, bridge width `k` as the unifying parameter, cost model. *(§2, §7)*
4. **Experimental design** — models, datasets, axes, protocol. *(§5, §6, §8)*
5. **Results** — quality (phase 2, esp. 2.3 bridge sweep), efficiency (phase 3), caching (phase 4).
6. **When DDI fails** — the cross-document reasoning envelope (phase 5). *Own section, not a
   limitations footnote.*
7. **Limitations** — batch size 1; 7–8B only; realization gap unresolved if phase 3 does not close it.
8. **Conclusion**

---

## 12. Honest assessment of where we stand

**Zero verified results.** Every claim C1–C5 is unverified. What exists is exploratory code across
six divergent mask implementations, three loss numbers at `n=1` on one model, and one timing
datapoint 9× below its own theory. The headline caching benefit has never been measured. We are
starting the measurement work, not finishing it.

What we *do* have is a good set of questions, and they are unusually well-shaped: a hypothesised
sharp positive (isolation affordable at +0.042 loss), a hypothesised sharp negative (query isolation
not, +1.245), and a quantified mystery (10.59× available, 1.20× realized). If those three survive
re-measurement they are a paper's spine — and the mystery is interesting whichever way it resolves.

Phase 0 is the real cost here: building the clean implementation and the provenance discipline that
makes any subsequent number defensible. It is also the thing that was missing the first time.
Phase 1 fixes `Δ` from `full`'s seed variance so C1 has a threshold to be tested against; Phase 2
(train-free first, fine-tuning only if 2.1 misses `Δ`, then the bridge sweep) makes C1 and C2
defensible; Phase 3 makes the efficiency claim honest rather than aspirational; Phase 4.1/4.2 makes
the caching co-headline stand up.

Recommended framing for the preprint: **a characterization study**, not a method-wins paper. "How
much cross-document attention can be removed, and what does removing it actually buy" is a claim we
can fully support and is more useful to the field than a benchmark table. It also means the
negative results are contributions rather than things to explain away.
