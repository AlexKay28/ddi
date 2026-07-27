# ddi — Decoupled Document Inference

Research code for **DDI (Decoupled Document Inference)**: making RAG-style prefill cheaper by
*not* letting retrieved documents attend to each other.

> **Status: clean-room rebuild in progress.** A single clean implementation is being written
> from zero. See [issue #1](https://github.com/AlexKay28/ddi/issues/1) for the roadmap and
> [`paper/RESEARCH_PLAN.md`](paper/RESEARCH_PLAN.md) for the research design.

---

## 1. The idea

In a RAG prompt the sequence is

```
[doc_1][doc_2]...[doc_N][dialog / query][answer]
```

Standard causal attention makes the prefill cost quadratic in the *whole* concatenation, even
though the retrieved documents are independent of one another — `doc_2` gaining the ability to
attend into `doc_1` is mostly wasted compute.

DDI replaces the full lower-triangular mask with a **block-diagonal** one: each document is
causal *within itself* and blind to every other document. Only the dialog/query/answer region is
allowed to read across blocks.

```
full causal                ddi_strict                 ddi_qf
(baseline)                 (docs + query isolated)    (docs isolated, query sees all)

d1 ██                      d1 ██                      d1 ██
d2 ████                    d2   ██                    d2   ██
d3 ██████                  d3     ██                  d3     ██
 q ████████                 q       ██                 q ████████
 a ██████████                a ██████████               a ██████████
                              ^ last row / answer
                                always sees all
```

Two properties make this attractive:

- **Sparsity → speed.** The block-diagonal mask over documents removes most of the prefill's
  quadratic term; the realized speedup vs. the analytical arithmetic is what Phase 3 measures.
- **Cacheability.** Because a document's KV block does not depend on its neighbours, each
  document's KV cache is position-shiftable and reusable across requests — a per-document cache
  hit skips prefill entirely, which no prefill-kernel improvement under `full` causal attention
  can match.

### Mask variants

| name | documents | query / dialog | answer |
|---|---|---|---|
| `full` | causal over everything (baseline) | causal | causal |
| `ddi_strict` | block-diagonal, mutually blind | attends only to itself; **last token** ("SEP trick") sees all | sees all |
| `ddi_qf` | block-diagonal, mutually blind | whole query block sees all documents | sees all |
| `ddi_bridge_k` | block-diagonal, mutually blind | last **k** query rows see all documents (interpolates strict ↔ qf) | sees all |
| `blind` | all-zero mask over documents (control) | sees dialog only | sees dialog only |

The clean rebuild ([issue #1](https://github.com/AlexKay28/ddi/issues/1) §0.2) treats these as
one parameterised family with a single boolean allow-matrix builder. `ddi_bridge_k` sweeps the
bridge width between the two endpoints; `blind` is the retrieval-measurement control (if a model
scores well while seeing no documents, the metric is not measuring retrieval).

---

## 2. Scope — models, benchmarks, roadmap

### Models

Two primary models, base and instruct variants (per
[issue #1](https://github.com/AlexKay28/ddi/issues/1) §2.1):

- **Qwen2.5-7B** — `Qwen/Qwen2.5-7B` and `Qwen/Qwen2.5-7B-Instruct`
- **Llama-3.1-8B** — `meta-llama/Llama-3.1-8B` and `meta-llama/Llama-3.1-8B-Instruct`

### Benchmarks

Split by function rather than a single list — a parity-detection benchmark and a
failure-envelope benchmark are different instruments
([issue #1](https://github.com/AlexKay28/ddi/issues/1) §1.1):

| purpose | dataset |
|---|---|
| Parity detection (headroom, low seed variance) | NQ, TriviaQA, HotpotQA |
| Controllable `N` + gold-document position | Lost-in-the-Middle-style multi-doc QA (Liu et al. 2023) |
| Cross-document stress | MoNaCo (`allenai/MoNaCo_Benchmark`); optionally 2WikiMultiHop, MuSiQue |
| Comparability with prior work | ≥1 dataset that Parallel Context Windows (PCW) or Fusion-in-Decoder (FiD) reports on (likely NQ or TriviaQA) |
| General RAG-native | Open RAG Bench (`vectara/open_ragbench`) |

### Roadmap

Six phases in [issue #1](https://github.com/AlexKay28/ddi/issues/1); Phase 0 is blocking. Phases
2, 3, 4 are independent once 0 and 1 land.

| phase | goal |
|---|---|
| 0 | Clean implementation + correctness gates (`N=1` equivalence, backend cross-check, doc-length regression) |
| 1 | Datasets, metrics, `full` baseline, non-inferiority margin `Δ` fixed from seed variance |
| 2 | Mask family quality — train-free primary, fine-tuned conditional, bridge-width sweep (paper centerpiece) |
| 3 | Efficiency + realization-ratio analysis |
| 4 | Per-document caching + retrieval-order invariance — **co-headline** with the prefill speedup |
| 5 | Failure envelope: single-hop vs multi-hop, PCW baseline, cross-document composition probe |
| 6 | Writeup |

---

## 3. Glossary

| term | meaning |
|---|---|
| **DDI** | Decoupled Document Inference — the block-diagonal document mask scheme |
| **IC** | information context, i.e. one retrieved document |
| **FTT** | first-token time; prefill latency |
| **TPOT** | time per output token; decode latency |
| **SEP trick** | forcing the last prefix row of an otherwise-isolated mask to attend to everything, so a single token bridges the blocks |
| **`Δ`** | non-inferiority margin — the maximum quality loss vs `full` still called parity. Fixed from `full`'s seed-to-seed variance in Phase 1.4 before any DDI number is interpreted |
| **PCW** | Parallel Context Windows (Ratner et al., ACL 2023) — training-free independent context windows; closest prior work |
| **FiD** | Fusion-in-Decoder (Izacard & Grave, 2021) — encode passages independently, fuse in the decoder |
