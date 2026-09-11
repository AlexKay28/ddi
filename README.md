# ddi — Decoupled Document Inference

Research code for **DDI (Decoupled Document Inference)**: making RAG-style prefill cheaper by
*not* letting retrieved documents attend to each other.

> **Status: bounded preprint evidence finalized.** Core code + experiment pipeline in place after the
> [#81 refactor](https://github.com/AlexKay28/ddi/issues/81). The preprint focuses on
> three evidence areas — **C1** (quality estimates, including a fixed-manifest
> three-seed transfer study), **C3**
> (analytical cost and qualified timing observations), **C4** (cache tests and
> a restart-layout throughput estimate). See
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

- **Sparsity → cost model.** The block-diagonal mask removes attended pairs from the analytical
  prefill count. No numerical historical Flex result or Flex model-throughput claim is retained.
- **Cacheability hypothesis.** Under the restart-layout assumptions, document KV blocks can be
  content-addressable. The reported throughput estimate is limited to one corrected restart-layout
  harness and request shape.

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
not measuring retrieval. That control is what removed `kilt_nq` from the grid — `blind` stayed
within ~10-25 % of `full` across F1/presence/loss on both instruct models, i.e. parametric
memory was answering NQ questions without the passages. Its loader has since been deleted for a
separate reason (see §2).

---

## 2. Scope — models, datasets, paper claims

### Models

Preprint measurements run on the instruct variants of two 7-8B models:

- `Qwen/Qwen2.5-7B-Instruct` — see [#74](https://github.com/AlexKay28/ddi/issues/74)
- `meta-llama/Llama-3.1-8B-Instruct` — see [#75](https://github.com/AlexKay28/ddi/issues/75)

Base variants (`Qwen/Qwen2.5-7B`, `meta-llama/Llama-3.1-8B`) are supported by the same
inference / mask code but are not part of the preprint's measurement grid.

### Datasets

**Three**, and only three. `ddi.data.list_datasets()` returns exactly
`['hotpot_qa', 'musique', 'synthetic_qa']`, and that is what `quality.yaml`,
`cache_restart.yaml` and `finetune.yaml` name (single-sourced graph-side as
`nirvana/common/constant.EVAL_DATASETS`).

| dataset | shape | what it is for |
|---|---|---|
| `hotpot_qa` | 10 passages/query (2 gold + 8 distractors), multi-hop | the workhorse — enough distractors that document isolation can plausibly cost something |
| `musique` | 20 paragraphs/query, 2–4 hops, author-verified compositionality | the densest context, and the one where cross-document composition should hurt most |
| `synthetic_qa` | 3 generated docs/record, answer string planted in one | tiny, saturated, hermetic. **Needs no network** — the only loader that does not, which is what `make smoke` runs on |

Two datasets were **deleted**, not merely dropped from a list, and both loaders are gone:

- `kilt_nq` — its `info_contexts` were Wikipedia *titles*, not passage bodies (harness review
  R1). The documents did not contain the answers, so the whole DDI comparison on that column
  was between three ways of attending to nothing. Restoring it needs the KILT
  knowledge-source join first.
- `nq` — NQ Open ships no passages at all, so `info_contexts` was always empty, every DDI
  variant coerced to `full` (the runner logs `coerced variant ... -> 'full'`), and the
  aggregator dropped the rows again on the way out. It was a null column in the strict sense
  of measuring nothing.

Loading either now raises `KeyError: unknown dataset`. Any older config, cache file or
summary naming them predates this and is not reproducible.

C3 (timing) and C4 cache correctness/throughput use synthetic tokens — no dataset axis.

Multi-hop failure-envelope datasets (MoNaCo / 2WikiMultiHop / `ddi_stress` families) and PCW
baseline comparisons are **future work** — dropped from preprint scope in
[#82](https://github.com/AlexKay28/ddi/issues/82).

### Paper claims

The preprint asserts three claims. C5 (failure envelope) and C6 (distractor robustness)
remain out of scope per [#82](https://github.com/AlexKay28/ddi/issues/82).

| claim | what | evidence |
|---|---|---|
| **C1** | Quality estimates under `ddi_qf` and `full` | The matched Qwen/Llama study uses three trained seeds and frozen manifests; neither architecture reaches natural-QA parity, and the sole valid architecture contrast establishes no practical difference. Other legacy rows remain single-run estimates |
| **C3** | Analytical cost versus qualified timing observations | SDPA has no sparsity benefit from a 4D mask; the unhashable historical Flex timing table and its numerical claims are removed. V49 Flex timing is deferred and did not run because the V47 parity gate fails |
| **C4** | End-to-end per-document cache reuse and a restart control | The small-shape V51 diagnostic matches short greedy generation but fails numerical identity. V53 withdraws the 1.219x throughput claim: V51 did not establish exact token geometry or equal decode work. Local revalidation is not clean-revision evidence. The historical 1.72x result also remains withdrawn |

### Fine-tuning evidence is in scope; further expansion is deferred

The paper retains the measured self-distillation evidence produced by
`experiments/finetune.yaml` (`kind: finetune`): a student runs under `ddi_qf`
against a teacher under `full`, with KL on answer-position logits. The matched
V46 study covers three independently trained seeds for the measured
Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct checkpoints on frozen manifests.
Neither architecture reaches natural-QA parity. The sole valid natural-QA
architecture contrast is musique F1 and establishes no practical difference;
it does not establish equivalence or invariance.

VIBEQUEUE-48's powered quality expansion and larger-scale cells are deferred
future work, not missing prerequisites for this bounded characterization. They
may strengthen or extend a later paper only after they run with the same
reproducibility controls; the current paper makes no scale-trend claim.
`make finetune MODEL=<hf-id> [NUM_GPUS=N]` remains the training entry point for
reopening that work.

VIBEQUEUE-49 compiled Flex timing is likewise deferred optional future work,
not a missing prerequisite. The current proxy fails the required cosine and
token-argmax gates for every mask, so no timing began and no Flex model-throughput,
batching, or serving claim is admitted. Reopening it must follow the unchanged
parity-first protocol in [`EVIDENCE_TO_CLAIM.md`](EVIDENCE_TO_CLAIM.md).

---

## 3. Running it

### Where the code is

**Arcadia, branch `users/alexkay28/alignment-2445-ddi`, at `junk/alexkay28/ddi`.** Every
number this repo has published was produced from that branch. `github.com/AlexKay28/ddi` is
a mirror at most: the issue links throughout this README are a real historical trail and are
worth reading, but you cannot `git clone` your way to a reproduction, and the CI workflow
under `.github/` has never gated anything that ran on a GPU here.

```bash
arc checkout users/alexkay28/alignment-2445-ddi
cd $(arc root)/junk/alexkay28/ddi
```

### What each entry point needs

Read this before picking a command. "needs" means *will not work without*.

| entry point | GPU | HF Hub | YT | Nirvana | network |
|---|---|---|---|---|---|
| `make setup` | – | – | – | – | PyPI (or `uv`, if installed) |
| `make test` | – | – | – | – | – |
| `make lint` | – | – | – | – | – |
| **`make smoke`** | – | – | – | – | – |
| `make yt-fetch MODEL=… HF_HOME=…` | – | – | **yes** | – | yes |
| `make yt-fetch-records` | – | – | **yes** | – | yes |
| `make hf-fetch MODEL=…` | – | **yes** | – | – | yes |
| `make exp-<name> MODEL=…` | **yes** | – (weights must already be in `HF_HOME`) | – | – | – |
| `make finetune MODEL=…` | **yes** (8 for the real rows) | – (same) | – | – | – |
| `make delta MODEL=… DATASET=… METRIC=…` | – | – | – | – | – |
| `make nv-list` / `make nv-pin` | – | – | – | – | `nv-pin` reads `arc` locally |
| `make nv-check GRAPH=…` / `nv-publish` | – | – | – | **yes** | yes |
| `make repro-pin INSTANCE=…` | – | – | – | **yes** (read-only) | yes |
| `make nv-status` / `nv-guard` | – | – | – | **yes** (read-only) | yes |

Two consequences worth stating plainly, because the old version of this section did not:

- **`make exp-<name>` requires a GPU and refuses to fall back.** `ddi.run`'s CLI honours
  `require_gpu:`, which defaults to true, so a sweep cannot quietly run on CPU and emit rows
  indistinguishable from GPU rows. There is exactly one config that opts out —
  `experiments/smoke.yaml`.
- **`hf-fetch` does not work on the hosts this project runs on.** None of them have HF Hub
  access; that is why `yt-fetch` exists and why every job sets `HF_HUB_OFFLINE=1`. Use
  `hf-fetch` only from a networked machine when *adding* a model to the YT mirror.

### Steps

**1. Install.** `uv` is not required and is not present on most of these hosts.

```bash
make setup      # uv sync --extra dev if uv is on PATH, else pip install -r requirements-dev.txt
```

`requirements-dev.txt` deliberately omits `torch` (the CUDA pin is host-specific — see that
file's header) and pins `transformers==5.14.1`, which is what the GPU jobs actually ran, not
what `uv.lock` resolves. Install a torch wheel for your machine separately; a CPU-only one is
enough for steps 2 and 3.

**2. Check the code imports and the units pass.**

```bash
make test       # 535 passed, 28 skipped as of 2026-09-06
```

Network-, CUDA-, and real-model integration tests are skipped unless explicitly enabled;
the default suite must otherwise pass.

**3. Run the pipeline end to end, on CPU, in ten seconds.**

```bash
make smoke
```

This is the check `make test` cannot be: it drives `ddi.run` → `ddi.store` → `ddi.aggregate`
under all three masks, through a real `LlamaForCausalLM` forward pass and a real greedy
decode, and writes `logs/smoke_summary.json`. It needs **no GPU, no HF Hub, no YT and no
network** — it builds its own 2-layer random model and a byte-level tokenizer locally, from a
config, downloading nothing. See `experiments/smoke.yaml` and the `smoke` block in
`Makefile.exp` for why it is built rather than fetched.

Be precise about what it does and does not prove. Its *numbers are meaningless* — random
weights, so em/f1/presence are ~0 by construction and you cannot read a quality signal out of
it. What it proves is that the whole path **executes**: real tokenization through the chat
template, real masked forward passes, a real greedy decode under each variant, real rows in
the store, and a summary the aggregator can build. That is not a low bar here. The unit suite
skips every test that needs `transformers` on a host without it (see `tests/fake_models.py`),
which is the gap two mask bugs shipped through, and the decode crash that lost nine GPU runs
on 2026-08-06 was a tensor-alignment fault no unit test could see.

To catch a *quality* regression — the class the "prompt never mentioned the documents" bug
belonged to, where `em` was 0 in every cell of every sweep — point the same YAML at a real
model on a GPU: `python -m ddi.run experiments/smoke.yaml --model <hf-id>`. It is a
three-minute version of step 4.

**4. Get the weights, then run a real experiment.** Needs a GPU from here on.

```bash
HF_HOME=/mnt/models make yt-fetch MODEL=Qwen/Qwen2.5-7B-Instruct
make yt-fetch-records          # dataset record cache; loaders never touch the Hub on a hit

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/mnt/models \
  make exp-quality MODEL=Qwen/Qwen2.5-7B-Instruct
```

The target is `exp-quality`, after `experiments/quality.yaml` — `make exp-<name>` is a
pattern rule over `experiments/<name>.yaml`, so `make list-experiments` is the authoritative
list of what you can pass. It writes rows to `results/runs.jsonl` and a **JSON** summary to
`logs/quality_summary.json` (not Markdown). No YAML names a model; `MODEL=` supplies it, and
also scopes the summary to that model.

For the GPU-host recipe — container base, offline flags, one-experiment-per-GPU — see
[`experiments/SERVERLESS.md`](experiments/SERVERLESS.md). For what each YAML measures, see
[`experiments/README.md`](experiments/README.md).

**5. Run one on the GPU fleet.** `nirvana/graphs/*.py` declares one entry point per grid row;
publishing one creates an instance you then start from the Nirvana UI.

```bash
make nv-list                       # every publishable entry point
make nv-pin                        # the revision the next publish would ship, vs the branch tip
make nv-check GRAPH=<name>         # dry run: validates, publishes nothing
make nv-publish GRAPH=<name>       # the real thing; prints the instance URL
```

`publish -n` is `--dry-run`, **not** `--name` — which is why no target here takes a flag that
flips one into the other. Read the header of `Makefile.nirvana` before using it.

**6. Re-run exactly what produced a published number.**

Start from the number's Nirvana instance, not from the summary file:

```bash
make repro-pin INSTANCE=<workflow-instance-uuid>
```

It prints the instance's identity stamp and the `arc checkout <sha>` that reproduces the code
it ran, e.g.

```
graph    : ddi_diagnose_7b_backends
model    : Qwen/Qwen2.5-7B-Instruct
arc-ref  : be94ea06ef
date     : 2026-08-08
ticket   : VIBEQUEUE-2

reproduce the code this instance ran:
  arc checkout be94ea06ef
```

**The revision is not in the summary JSON.** `ddi.aggregate` stamps every summary with a
`git_sha` field, and for every Nirvana run that field reads the literal string
`not-a-git-repo`: the job executes from an `arc export`, a plain directory tree with no VCS
metadata for `ddi/provenance.py` to read. The only surviving record is the **instance
comment**, written at publish time by `nv-comment.py` out of
`nirvana/common/constant.ARC_REFERENCE`, in the format
`<graph> | <model> | <arc-ref> | <date> | <ticket> | <note>`. `make repro-pin` reads it back.

That also means an instance published *without* `make nv-publish` has no revision recorded
anywhere — `repro-pin` says so and exits non-zero rather than guessing. `publish` writes a
comment of its own whose Arcadia link points at `rev=-1`, which does not resolve.

Then, from that revision, re-run the row's own command. Each graph block shells out to plain
`make` inside the exported tree (`make yt-fetch MODEL=…`, `make finetune MODEL=…`,
`make exp-quality MODEL=…` — see `nirvana/graphs/scripts.py`), so step 4 above *is* the
reproduction, with `MODEL=` and any `--set` overrides taken from the stamp and the graph.

---

## 4. Glossary

| term | meaning |
|---|---|
| **DDI** | Decoupled Document Inference — the block-diagonal document mask scheme |
| **IC** | information context, i.e. one retrieved document |
| **TTFT** | time to first token; prefill latency (also called `FTT` in some older code paths) |
| **TPOT** | time per output token; decode latency |
| **SEP trick** | forcing the last prefix row of an otherwise-isolated mask to attend to everything, so a single token bridges the blocks |
| **`Δ`** | descriptive full-baseline spread. The available artifacts do not support using it as a mask-comparison parity threshold |
| **PCW** | Parallel Context Windows (Ratner et al., ACL 2023) — training-free independent context windows; closest prior work |
| **FiD** | Fusion-in-Decoder (Izacard & Grave, 2021) — encode passages independently, fuse in the decoder |
