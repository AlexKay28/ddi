# ddi — Decoupled Document Inference

Research code for **DDI (Decoupled Document Inference)**: making RAG-style prefill cheaper by
*not* letting retrieved documents attend to each other.

> **Status: preprint in progress.** Core code + experiment pipeline in place after the
> [#81 refactor](https://github.com/AlexKay28/ddi/issues/81). The preprint focuses on
> three claims — **C1** (quality parity), **C3** (speed / realized-vs-analytical),
> **C4** (KV-cache reuse under restart layout). See
> [`paper/RESEARCH_PLAN.md`](paper/RESEARCH_PLAN.md) for the research design and
> [`experiments/README.md`](experiments/README.md) for what runs.

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

All five variants share one parameterised boolean allow-matrix builder in `ddi/masks.py`.
`ddi_bridge_k` sweeps the bridge width between the strict/qf endpoints (not exercised by the
preprint's C1/C3/C4 experiments — see the scope note below). `blind` is the
retrieval-measurement control: if a model scores well while seeing no documents, the metric is
not measuring retrieval. In practice this check flagged `kilt_nq` on both instruct models
(blind stays within ~10-25 % of `full` across F1/presence/loss) — parametric-memory answers
dominate on NQ, so `kilt_nq` is treated as informational rather than evidential. See
[#72's DDI conclusions](https://github.com/AlexKay28/ddi/issues/72#issuecomment-5115470772).

---

## 2. Scope — models, datasets, paper claims

### Models

Preprint measurements run on the instruct variants of two 7-8B models:

- `Qwen/Qwen2.5-7B-Instruct` — see [#74](https://github.com/AlexKay28/ddi/issues/74)
- `meta-llama/Llama-3.1-8B-Instruct` — see [#75](https://github.com/AlexKay28/ddi/issues/75)

Base variants (`Qwen/Qwen2.5-7B`, `meta-llama/Llama-3.1-8B`) are supported by the same
inference / mask code but are not part of the preprint's measurement grid.

### Datasets

The C1 quality grid runs on four productive datasets plus a null column:

| purpose | dataset |
|---|---|
| Parity detection (single-hop retrieval Q&A) | `hotpot_qa`, `musique`, `kilt_nq` |
| Synthetic single-hop signal (headroom, low variance) | `synthetic_qa` (in-house generator) |
| Null column (no docs — all masks collapse to full) | `nq` |

C3 (timing) and C4 (cache) grids use synthetic tokens — no dataset axis.

Multi-hop failure-envelope datasets (MoNaCo / 2WikiMultiHop / `ddi_stress` families) and PCW
baseline comparisons are **future work** — dropped from preprint scope in
[#82](https://github.com/AlexKay28/ddi/issues/82).

### Paper claims

The preprint asserts three claims. C5 (failure envelope), C6 (distractor robustness), and
the fine-tuning path are out of scope per [#82](https://github.com/AlexKay28/ddi/issues/82).

| claim | what | evidence |
|---|---|---|
| **C1** | `ddi_qf` matches `full` within `Δ` on retrieval Q&A | measured on Qwen + Llama instruct grids ([#74](https://github.com/AlexKay28/ddi/issues/74), [#75](https://github.com/AlexKay28/ddi/issues/75)) |
| **C3** | Speed: analytical ≠ realized — the gap is the story | measured on both instruct models; realization ratio 0.05–0.4 on `sdpa` |
| **C4** | Per-document KV cache reuse under restart layout preserves quality and improves throughput | measured on Qwen; restart-vs-sequential parity within `Δ` |

---

## 3. Reproducing the numbers

Every experiment on `main` is one YAML file in [`experiments/`](experiments/) plus one
`Makefile` line. From a fresh clone:

```bash
# 1. Install (deterministic — resolved from uv.lock).
git clone https://github.com/AlexKay28/ddi.git
cd ddi
uv sync

# 2. Fetch model weights once, cached to a persistent HF_HOME.
#    No direct HF Hub access? Use `make yt-fetch` instead — same MODEL/HF_HOME contract.
HF_HOME=/mnt/models make hf-fetch MODEL=Qwen/Qwen2.5-7B-Instruct

# 3. Run an experiment offline; the runner writes a Markdown summary
#    with a provenance footer. No YAML names a model — pass it via MODEL=.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/mnt/models \
  make exp-c1 MODEL=Qwen/Qwen2.5-7B-Instruct

# 4. Publish the summary as a comment on its tracking issue.
make publish-summary SUMMARY=logs/c1_summary.md ISSUE=74
```

New experiment = new YAML in `experiments/`. `make exp-<name>` picks it up via a
pattern rule — no `Makefile` edit needed. See:

- [`experiments/README.md`](experiments/README.md) — index of every YAML with a
  one-line description and the GitHub issue tracking it.
- [`experiments/SERVERLESS.md`](experiments/SERVERLESS.md) — one-page operator
  recipe (container base, HF prefetch, offline mode, one-experiment-per-GPU rule).

---

## 4. Glossary

| term | meaning |
|---|---|
| **DDI** | Decoupled Document Inference — the block-diagonal document mask scheme |
| **IC** | information context, i.e. one retrieved document |
| **TTFT** | time to first token; prefill latency (also called `FTT` in some older code paths) |
| **TPOT** | time per output token; decode latency |
| **SEP trick** | forcing the last prefix row of an otherwise-isolated mask to attend to everything, so a single token bridges the blocks |
| **`Δ`** | non-inferiority margin — the maximum quality loss vs `full` still called parity. Fixed from `full`'s seed-to-seed variance in Phase 1.4 before any DDI number is interpreted |
| **PCW** | Parallel Context Windows (Ratner et al., ACL 2023) — training-free independent context windows; closest prior work |
| **FiD** | Fusion-in-Decoder (Izacard & Grave, 2021) — encode passages independently, fuse in the decoder |
