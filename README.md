# ddi — Decoupled Document Inference

Research code for **DDI (Decoupled Document Inference)**: making RAG-style prefill cheaper by
*not* letting retrieved documents attend to each other.

> **Status: clean-room rebuild in progress.** The prior working tree (`v1`–`v4`, kept outside
> this repo) was a sequence of debugging iterations, not a codebase to build on. It is
> **read-only archaeology** — consulted for what was attempted and what it suggested, never
> imported from. A single clean implementation is being written from zero, and every prior number
> is demoted to a **hypothesis to re-test**. See
> [issue #1](https://github.com/AlexKay28/ddi/issues/1) for the roadmap and
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
full causal                block-sparsed              block-sparsed-query-full
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

- **Sparsity → speed.** With 13 contexts, `v2/README.md` reports 91.6% mask sparsity and a
  **1.20× faster prefill** (720 ms → 600 ms) using FlashAttention-2 varlen. Analytical arithmetic
  predicts ~10.6× at the same shape; that ~9× gap is hypothesis **H3** below and is what Phase 3.5
  is designed to explain or bound.
- **Cacheability.** Because a document's KV block does not depend on its neighbours, each
  document's KV cache is position-shiftable and reusable across requests. The
  `notebooks/KV-cache-and-position-analysis.ipynb` experiment probes exactly this: it perturbs
  a prefix and checks whether the suffix KV cache changes under a causal vs. block-diagonal mask.

### Mask variants used throughout

| name | documents | query / dialog | answer |
|---|---|---|---|
| `full` | causal over everything (baseline) | causal | causal |
| `ddi_strict` (a.k.a. `block-sparsed`) | block-diagonal, mutually blind | attends only to itself; **last token** ("SEP trick") sees all | sees all |
| `ddi_qf` (a.k.a. `block-sparsed-query-full`) | block-diagonal, mutually blind | whole query block sees all documents | sees all |
| `ddi_bridge_k` | block-diagonal, mutually blind | last **k** query rows see all documents (interpolates strict ↔ qf) | sees all |
| `blind` | all-zero mask over documents (control) | sees dialog only | sees dialog only |

The clean rebuild ([issue #1](https://github.com/AlexKay28/ddi/issues/1) §0.2) treats these as
one parameterised family with a single boolean allow-matrix builder. `ddi_bridge_k` sweeps the
bridge width between the two prior endpoints; `blind` is the retrieval-measurement control from
`v1/evaluate_models__blind.py`.

`v1` additionally explored `DDI_3` / `DDI_3_blind` — appending a fixed assistant "intro" span
(*"Well, well well. Its a very good question. Answer according to what I see in documents is: "*)
that either can or cannot see the documents, to isolate how much the model actually reads from
context (see `v1/main_2.py`, `v1/evaluate_models__blind.py`).

---

## 2. Scope — models, benchmarks, roadmap

### Models

Two primary models, base and instruct variants (per
[issue #1](https://github.com/AlexKay28/ddi/issues/1) §2.1):

- **Qwen2.5-7B** — `Qwen/Qwen2.5-7B` and `Qwen/Qwen2.5-7B-Instruct`
- **Llama-3.1-8B** — `meta-llama/Llama-3.1-8B` and `meta-llama/Llama-3.1-8B-Instruct`

The archaeology (`v1`, `v2`) additionally exercised `YandexGPT-5-Lite-8B` in both flavours; not in
scope for the clean rebuild.

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

Prior work (`v1`, `v2`) sourced only Open RAG Bench and MoNaCo, via `datasets/revision.ipynb`.

### Roadmap

Six phases in [issue #1](https://github.com/AlexKay28/ddi/issues/1); Phase 0 is blocking. Phases
2, 3, 4 are independent once 0 and 1 land.

| phase | goal |
|---|---|
| 0 | Clean implementation + correctness gates (`N=1` equivalence, backend cross-check, doc-length regression) |
| 1 | Datasets, metrics, `full` baseline, non-inferiority margin `Δ` fixed from seed variance |
| 2 | Mask family quality — train-free primary, fine-tuned conditional, bridge-width sweep (paper centerpiece) |
| 3 | Efficiency + realization-ratio analysis (the ~9× gap between analytical and measured speedup) |
| 4 | Per-document caching + retrieval-order invariance — **co-headline** with the prefill speedup |
| 5 | Failure envelope: single-hop vs multi-hop, PCW baseline, cross-document composition probe |
| 6 | Writeup |

---

## 3. Layout

```
datasets/       revision.ipynb — benchmark sourcing (Open RAG Bench, MoNaCo) → upload to YT
notebooks/      exploratory analysis: FlashAttention microbenchmarks, Triton kernel,
                KV-cache/RoPE invariance study, results dashboards (*.html are plotly exports)
v1/             first pass: hand-built 4D additive masks + HF forward, offline eval on YT tables
v2/             production-ish pass: FlashAttention-2 varlen, SDPA monkey-patch, fine-tuning,
                multi-model inference server, benchmark harness
v3/             rewrite: hand-rolled prefill/decode loop, ClassicModel vs DDIModel, FastAPI
v4/             current: v3 + fixed decode KV bookkeeping + SFT training loop
```

The four `vN` directories are **successive rewrites, not layers** — later versions supersede
earlier ones. `v4` is the live one; `v1`/`v2` are kept for their evaluation and benchmarking
harnesses, which `v3`/`v4` never reimplemented.

---

## 4. Version by version

### v1 — masks by hand, offline evaluation

Builds `[P, P]` 0/1 masks in Python, converts them to HF additive form
(`0 = keep`, `-inf = block`) and feeds them straight into a Llama/Qwen forward.

- `utils/experiment_core.py`, `utils/experiment_core_v2.py` — greedy generate loop that measures
  **FTT** (first-token time, i.e. prefill) and **TPOT** (time per output token) with CUDA events.
  `_v2` also breaks out of degenerate repetition loops.
- `patch.py` — monkey-patches `LlamaModel._update_causal_mask` (so a supplied 4D mask survives
  instead of being overwritten) and `LlamaAttention.forward` (eager attention that zeroes
  post-softmax weights at `-inf` positions, so a fully-masked row yields 0 rather than a uniform
  distribution). The same patch is inlined into each `evaluate_models__*.py`.
- `evaluate_models__{base,instruct,blind}.py` + `Makefile` — fan out one model per GPU (0–5) over
  Llama-3.1-8B, Qwen2.5-7B and YandexGPT-5-Lite-8B, in both base and instruct flavours. `blind`
  adds all-zero masks as a control: if a model answers just as well while seeing *nothing*, the
  metric isn't measuring retrieval.
- I/O is YT (`YT_PROXY=hahn`); each row carries `instruct_id`, `query`, `query_rag`,
  `info_contexts`, `possible_answers`, and results are appended as `answer_source` / `answer` /
  `statistics`.

`main.py` / `main_2.py` are the scratch drivers this grew out of. Note `main.py:97` still has a
debug `assert False, "kek"` — use `main_2.py` or the `evaluate_models__*` scripts.

### v2 — FlashAttention-2, fine-tuning, serving

The most complete version. `v2/README.md` is its own quickstart.

- `utils/flash_attention_sparse.py` — the real optimization. Documents go through
  `flash_attn_varlen_func` with `cu_seqlens` at document boundaries (block-diagonal for free);
  the query region goes through `flash_attn_func` separately, split into "prefix" and "last token"
  when emulating `block-sparsed`.
- `utils/patch_attention_minimal.py` — swaps `F.scaled_dot_product_attention` globally. A module
  global holds the current document lengths; set it and attention goes block-sparse, clear it and
  it falls back to stock SDPA. That's the whole API:
  ```python
  model = patch_model_minimal(model)
  set_block_sparse_context([500, 600, 550])   # prefill
  outputs = model(input_ids)
  clear_block_sparse_context()                # decode
  ```
- `utils/inference.py` — two paths for the same mask semantics: an explicit `-inf` 4D mask for
  unpatched ("classic") models, and the patched FlashAttention path. Also `compute_sparsity()`,
  per-request CUDA streams, and input truncation that rewrites `context_lengths` accordingly.
- `benchmark.py` + `Makefile` — sweeps `{full, block-sparsed, block-sparsed-query-full}` per model,
  scores EM / F1 / answer-presence against `possible_answers`, and prints a transposed
  FTT/TPOT/sparsity table plus a speedup analysis.
- `fine_tune.py` + `Makefile.tune` — trains the *same* model under each of the three masks so the
  mask becomes part of the model rather than an inference-time hack. `create_learning_sample()` is
  where each mask is constructed; labels are `-100` over documents+query so loss lands only on the
  answer. `parallel-monitor` launches all three on separate GPUs and tails them via `multitail`.
- `server.py` — FastAPI inference service (~780 lines): auto-discovers fine-tuned checkpoints under
  `models/`, pins each base model to a GPU, keeps classic and patched copies separately, and
  unloads on a TTL. `mask_type` is per-request. `simple_client.py` / `test_curl.sh` exercise it.
- `upload_models.sh` — pushes `models/*/Qwen2.5-7B/best_model` to YT via `yt dirtable upload`.

### v3 — explicit prefill/decode, no HF generate

Drops HF's `generate` and mask plumbing entirely and walks the decoder layers by hand
(`embed → per-layer q/k/v → RoPE → attention → MLP → norm → lm_head`), keeping per-layer KV in
plain dicts. Three model classes share that skeleton:

- `source/model_classic.py` — baseline; `is_causal=True` SDPA, no explicit mask.
- `source/model_ddi.py` — same loop, but `create_prefill_attn_mask()` builds the block-diagonal
  document mask and passes it as an explicit `attn_mask`. Decode uses an all-zeros mask (attend
  to everything cached).
- `source/model_shadow.py` — a *different* line of work carried in for comparison: chunked
  landmark-based sparse KV selection with fp8 quantization emulation (sink / sparse pool / local
  window). It imports a `reference/` package that is **not present** in the tree, so it does not
  currently run.

`server.py` here is the richer of the two servers — it exposes `/set_model_class` to swap
model + inference class at runtime and validates against an allow-list of model names.

### v4 — current

Same three model classes, plus:

- **Decode KV fix.** v3's `model_ddi.decode_step` appended new K/V into `_layer_prefix_k`, growing
  the same buffer it treated as the frozen document prefix. v4 keeps `_layer_decode_k/v` separate
  and concatenates `[prefix, decode]` at attention time — so the prefix stays reusable, which is
  the point of DDI.
- **`train.py` + `dataset.py`** — a from-scratch SFT loop (no `Trainer`/`trl`/`peft`) for
  `Qwen/Qwen2.5-7B-Instruct`. `--attention_mode causal|ddi` selects stock SDPA vs. eager attention
  with a `build_ddi_mask()` 4D mask. Gradient accumulation at `batch_size=1`, gradient
  checkpointing, linear warmup then constant LR, non-finite-loss skipping, periodic
  validation + checkpointing.
- `dataset.py` tokenizes exactly like `infer_model.py`: each document raw (no chat template), the
  dialog via `apply_chat_template`, documents prepended; labels masked with `-100` up to the last
  `<|im_start|>assistant\n` header.
- `server.py` is simplified relative to v3 — one model chosen by the `MODEL_CLASS` env var
  (`ClassicModel` | `DDIModel`), no runtime swap.

---

## 5. Data format

Records are uniform across versions (`v4/dummy_data.jsonl` is a 50-line smoke set):

```json
{
  "info_contexts": ["doc 1 text", "doc 2 text"],
  "dialog": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

v1/v2 read the flat YT variant instead — `query`, `query_rag` (query with documents inlined),
`info_contexts`, `possible_answers`, `instruct_id`. Benchmarks were sourced from
`vectara/open_ragbench` and `allenai/MoNaCo_Benchmark` (`datasets/revision.ipynb`), staged as YT
tables under `//home/gena/alexkay28/tickets/ALIGNMENT-2445/` and
`//home/gena/alexkay28/nirvana/output1_*`.

The serving API is:

```
POST /infer   {"dialog": [...], "meta_info": {"info_contexts": [...]}}
           →  {"model_name": "...", "answer": "..."}
GET  /health
```

---

## 6. Reference hypotheses (from the archaeology, **not** results)

> These numbers came from the prior working tree. They are not results — the harness that produced
> them predates the correctness gates in [issue #1](https://github.com/AlexKay28/ddi/issues/1)
> §0.5, and every implementation detail is being rebuilt. They are kept as **sanity targets**:
> if the clean harness reproduces them, that raises confidence in the rebuild. Nothing here is
> claimed as a finding.

**H1 / H2 — quality cost of the mask** — one epoch of Qwen2.5-7B fine-tuning, best validation loss
(`v2/logs/`):

| id | mask | prior best val loss | prior final train loss |
|---|---|---|---|
| H1 | `full` (baseline) | 0.2355 | 0.0170 |
| H1 | `ddi_qf` (`block-sparsed-query-full`) | 0.2773 | 0.0178 |
| H2 | `ddi_strict` (`block-sparsed`) | 1.4801 | 0.0543 |

The story the numbers *suggested*: isolating documents from each other is nearly free (+0.04) as
long as the query still sees all of them; isolating the query too — leaving only the SEP-token
row as a bridge — is not viable (0.24 → 1.48). This is what motivated `ddi_qf` being the default
mask in `v2/fine_tune.py`, and what the bridge-width sweep in Phase 2.3 is designed to turn from
a three-point step function into a mechanism with a located knee.

**H3 — prefill speed** — `v2/README.md`: 1.20× (720 ms → 600 ms) at 13 contexts / 91.6% mask
sparsity. Analytical arithmetic at `ℓ=512, Q=64`
([`paper/RESEARCH_PLAN.md`](paper/RESEARCH_PLAN.md) §7) predicts 10.59×; the ~9× gap between
analytical and measured is Phase 3.5's job to explain or bound. Microbenchmarks in
`notebooks/flasattention_tests*.py`, `notebooks/flash_attention_blocks_eval.ipynb`, and the
Triton kernel in `notebooks/triton_block_attention.py` are where the launch-overhead vs.
memory-boundedness questions were first raised.

Generation-side scoring (EM / token-F1 / answer-presence, against FTT and TPOT) lives in
`notebooks/results_analysis.ipynb` and `results_analysis_v2.ipynb`; the `*.html` files are the
exported plotly dashboards from those runs. No EM / F1 / presence number has ever been produced by
a validated harness — this is Phase 2's job.

---

## 7. Running things (archaeology)

The commands below run the prior working tree, not the clean rebuild. Kept for reference until the
Phase 0 CLI (`run.py --config <file> --seed <n>`) lands.

Everything assumes CUDA, `transformers`, `torch`, and — for v2 — `flash-attn`.
v1/v2 additionally need `yt.wrapper` with `YT_PROXY`/`YT_TOKEN` set, and an `HF_TOKEN` in `.env`
for gated checkpoints. v3/v4 shell scripts call `python3.11` explicitly.

```bash
# v4 — serve
MODEL_CLASS=DDIModel PORT=6890 ./v4/run_server.sh
./v4/test_query.sh

# v4 — compare classic vs DDI on one hardcoded request
python v4/infer_model.py

# v4 — SFT smoke test
./v4/run_train.sh                      # causal mode; DDI variant is commented out in the script

# v2 — fine-tune all three masks in parallel on 3 GPUs
make -f v2/Makefile.tune parallel-monitor GPU1=5 GPU2=6 GPU3=7

# v2 — benchmark masks × models
make -f v2/Makefile parallel-all

# v1 — offline eval fan-out across 6 GPUs
make -C v1 parallel-all
```

---

## 8. Known gaps (in the archaeology)

Carried forward as-is; worth knowing before extending this code.

- **`documents_sizes_in_tokens` is wrong in `v3/v4/infer_model.py`.** `tokenizer.encode(...,
  return_tensors="pt")` returns a `[1, L]` tensor, so `len(ctx_indices)` evaluates to `1` for every
  document. `DDIModel` therefore builds a mask of N one-token documents rather than the real spans
  — the DDI path is effectively inert at inference. Should be `ctx.shape[-1]`. Encoded as a
  permanent regression test in [issue #1](https://github.com/AlexKay28/ddi/issues/1) §0.5.
- **`v4/run_train.sh` runs `causal` only.** The DDI branch is commented out in the script, so no
  clean run of the v4 SFT loop under the DDI mask exists. Combined with the bullet above, the DDI
  path in `v4` is inert at both training and inference.
- **`v3/v4/source/model_shadow.py` doesn't import** — needs a `reference/` package
  (`compute_representatives`, `select_representatives`, `compute_dist`) that isn't in the tree.
- **`v2/validate_triton.py` doesn't import** — wants `utils/triton_block_sparse_attention.py`;
  only the standalone `notebooks/triton_block_attention.py` exists.
- **`v3/run_train.sh` is empty**; the training loop only appeared in v4.
- **`v1/main.py:97`** has a leftover `assert False, "kek"`.
- Batch size is hard-pinned to 1 in v3/v4 (`generate` raises otherwise) and in v2's fine-tuning
  collate; scale comes from gradient accumulation only.
- YT table paths and GPU indices are hardcoded throughout the Makefiles and `main*.py`.

---

## 9. Glossary

| term | meaning |
|---|---|
| **DDI** | Decoupled Document Inference — the block-diagonal document mask scheme |
| **IC** | information context, i.e. one retrieved document (`info_contexts`) |
| **FTT** | first-token time; prefill latency |
| **TPOT** | time per output token; decode latency |
| **SEP trick** | forcing the last prefix row of an otherwise-isolated mask to attend to everything, so a single token bridges the blocks |
| **classic / patched** | unmodified model vs. one with `F.scaled_dot_product_attention` swapped for the block-sparse implementation |
| **shadow** | the separate landmark-based sparse-KV line of work vendored into `v3`/`v4` for comparison |
| **`Δ`** | non-inferiority margin — the maximum quality loss vs `full` still called parity. Fixed from `full`'s seed-to-seed variance in Phase 1.4 before any DDI number is interpreted |
| **PCW** | Parallel Context Windows (Ratner et al., ACL 2023) — training-free independent context windows; closest prior work |
| **FiD** | Fusion-in-Decoder (Izacard & Grave, 2021) — encode passages independently, fuse in the decoder |
| **H1 / H2 / H3** | prior-work numbers demoted to hypotheses: quality of `full` / `ddi_qf` (H1), quality of `ddi_strict` (H2), prefill speed (H3). See §6 |
