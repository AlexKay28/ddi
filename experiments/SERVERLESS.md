# experiments/SERVERLESS.md

One-page recipe for running a DDI experiment on a GPU host — a Nirvana job's container, a
serverless pod (RunPod, Modal, Vast, Lambda), or a dev box with a card. The design assumes a
persistent volume for model weights and one experiment per GPU.

For the index of what each YAML measures, see [`README.md`](README.md). For the
capability table (which entry point needs GPU / YT / Nirvana / HF Hub), see
[`../README.md` §3](../README.md#what-each-entry-point-needs).

**Read this first: these hosts have no HuggingFace Hub access.** Not the Nirvana containers,
not the dev boxes. Every step below is built around that, which is why models and dataset
records come from a YT mirror and every run sets `HF_HUB_OFFLINE=1`. A recipe that starts
with `hf-fetch` is a recipe for a host you do not have.

---

## Recipe

### 1. Container base

```
nvidia/cuda:12.1.1-cudnn-runtime-ubuntu22.04
```

Chosen to match the CUDA 12.1 pin used by the project's torch build (`torch==...+cu121`
in `pyproject.toml`). The GPU boxes run driver 535.129.03 (CUDA 12.2 max) and cannot load a
`+cu130` wheel. Any newer CUDA runtime image works if it stays ABI compatible with the pinned
torch wheel; the safest choice is this exact tag.

### 2. Get the code and install

The code is in **Arcadia**, branch `users/alexkay28/alignment-2445-ddi`, at
`junk/alexkay28/ddi`. Nirvana jobs get it as an `arc export` of a pinned revision — see
`nirvana/common/constant.ARC_REFERENCE` and `make nv-pin`. By hand:

```bash
arc checkout users/alexkay28/alignment-2445-ddi
cd $(arc root)/junk/alexkay28/ddi
make setup
```

`make setup` uses `uv sync --extra dev` when `uv` is on PATH and falls back to
`pip install -r requirements-dev.txt` otherwise; **`uv` is not required and is not present on
most of these hosts.** `requirements-dev.txt` deliberately omits `torch` (the CUDA pin above is
host-specific) and pins `transformers==5.14.1` — what the GPU jobs actually ran, which is a
*different code path* through `ddi/model.py` from the 4.52.4 that `uv.lock` resolves. Install
a torch wheel for your machine separately.

### 2b. Confirm the pipeline works before spending a GPU on it

```bash
make smoke
```

CPU-only, no network, ~10s. It drives `ddi.run` → store → `ddi.aggregate` under all three
masks on a locally-constructed tiny model, and fails loudly if any of that is broken. Cheap
insurance against discovering a harness bug four hours into an 8-GPU allocation. The numbers
it produces are meaningless by construction — see [`smoke.yaml`](smoke.yaml).

### 3. Fetch the model weights (once per model)

```bash
HF_HOME=/mnt/models make yt-fetch MODEL=Qwen/Qwen2.5-7B-Instruct
```

Run this **once**, pointed at a persistent volume mounted at `/mnt/models`. `yt-fetch` pulls
the bytes from our YT mirror at
`//home/gena/alexkay28/tickets/ALIGNMENT-2445/models/<repo-name>` and reconstructs an
`$HF_HOME` cache layout (`refs/main` + a `yt-cache` snapshot dir) so `from_pretrained(MODEL)`
resolves offline afterwards.

Mirrored today (`yt --proxy=watt list //home/gena/alexkay28/tickets/ALIGNMENT-2445/models`):
base models `Qwen2.5-7B`, `Qwen2.5-7B-Instruct`, `Qwen3.5-4B-Instruct`, `Qwen3.6-27B-Instruct`,
`Llama-3.1-8B-Instruct`, plus the fine-tuned checkpoints written back by the finetune graphs
(`Qwen__…__ddi_qf__…`). Two naming conventions coexist and `yt-fetch` tries both — bare repo
suffix for base models, `/`→`__` for checkpoints.

**Adding a model to the mirror** is the one job `make hf-fetch` exists for: from a machine
that *does* have Hub access, `hf-fetch` it into a local `HF_HOME`, then run
`./upload_models_to_yt.sh`. Do not put `hf-fetch` in a GPU-host recipe.

**If `HF_HOME` is not a persistent volume,** the fetch is thrown away when the container exits
and every run re-fetches ~15 GB. Mount the volume.

### 3b. Restore the dataset record cache

```bash
make yt-fetch-records
```

Every `ddi.data` loader checks `results/splits/<name>__seed<seed>__<split>_records.jsonl`
first and never touches the Hub on a hit (see `ddi/data/records_cache.py`). This target
restores those files from
`//home/gena/alexkay28/tickets/ALIGNMENT-2445/data/records_cache`.

This is **not** optional on an offline host, and mirroring the raw HF dataset snapshot
instead would not help: `datasets.load_dataset()`'s offline path needs the library's own
prebuilt arrow cache, not the source files. The record cache sidesteps that entirely.

To prime it from scratch on a networked build node — the only place this loop can run —
and re-upload:

```bash
for name in hotpot_qa musique synthetic_qa; do
  python3 -c "from ddi.data import load; list(load('$name', 'val', 0))"
done
./upload_records_cache_to_yt.sh
```

Those three are the entire registry (`ddi.data.list_datasets()`), and are single-sourced
graph-side as `nirvana/common/constant.EVAL_DATASETS`. This loop used to name `kilt_nq` and
`nq` as well; both loaders have since been deleted, so the loop died on the third iteration.
Stale `kilt_nq_*` / `nq_*` files still sitting on the YT mirror are harmless — nothing can
load them — but do not read them as evidence those datasets are supported.

`synthetic_qa` needs no network at all; it is generated from `(split, seed)`.

### 4. Run the experiment (offline)

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/mnt/models \
  make exp-quality MODEL=Qwen/Qwen2.5-7B-Instruct
```

The two `*_OFFLINE=1` flags force `transformers` and `huggingface_hub` to fail loudly if any
code path tries a network call — which catches "silently re-downloading on every run" bugs.
They are set by the **operator**, not by the Makefile, so a misconfigured cache surfaces
immediately rather than being papered over.

The runner writes rows to `results/runs.jsonl` and `ddi.aggregate` emits
`logs/quality_summary.json` (JSON, not Markdown) from the YAML's `report:` block. Substitute
any name from `make list-experiments` and the matching `MODEL=`; the store is append-only, and
passing `MODEL=` also scopes the aggregated summary to that model.

Fine-tuning is the exception to the pattern: use `make finetune MODEL=… [NUM_GPUS=N]`, not
`make exp-finetune`. The training step may need `torchrun --nproc_per_node=N` while the
aggregate step stays single-process, and swapping the launcher for one step only is what that
target owns.

### 5. Verify before reporting anything

There is no per-cell log file — cell failures go to **stderr**, and the sweep isolates them so
one bad cell does not abort the rest. Two checks, in order:

1. **Exit status.** `ddi.run` exits non-zero if any cell failed (`_CellTracker.failed_total`),
   and `make` propagates it. A green `make exp-<name>` means every cell wrote rows. This was
   not always true: the runner used to print tracebacks, write nothing and exit 0, which is
   how a Nirvana finetune block went green with no checkpoint and the eval downstream scored a
   stale model.
2. **The summary's own counters.** Each table carries `backing_rows`, `distinct_run_ids`,
   `models` and `seeds`. A table with a missing dataset key is a cell that produced nothing;
   `backing_rows` far larger than the grid means the store carried historical rows for that
   model as well — `ddi.aggregate` filters by model only, not by dataset, mask, config hash or
   recency.

Expected non-failures: `fa2` cells on a host without flash-attn (see the gotcha below).

### 6. Reporting

Copy the summary JSON into the tracking ticket by hand. There is **no** `make publish-summary`
any more: it shelled out to `gh issue comment`, `gh` is installed on none of these hosts, and
it targeted GitHub while the numbers are reported into the Yandex ticket and the code lives in
Arcadia. See the tombstone comment in `Makefile.exp` if you are tempted to restore it.

When you report a number, report the instance it came from. The revision is **not** in the
summary JSON — `git_sha` reads `not-a-git-repo` for every Nirvana run, because the job executes
from an `arc export` with no VCS metadata. The instance comment is the record:
`make repro-pin INSTANCE=<uuid>` reads it back and prints the `arc checkout` that reproduces it.

---

## One experiment per GPU

**Convention:** run one `make exp-<name>` at a time on a given GPU. Do not launch a
timing grid and a quality grid concurrently on the same card. Do not fan out inside a
YAML in a way that keeps two model copies resident.

This is not enforced by code — the runner will happily start a second grid if you
launch one — but two operator failures both traced back to violating it:

- **[#74 (Qwen C1)](https://github.com/AlexKay28/ddi/issues/74#issuecomment-5115033578)**
  — OOM from a concurrent grid sharing the H100; root cause was two configs resident
  at once, not any single config's peak.
- **[#75 (Llama C1)](https://github.com/AlexKay28/ddi/issues/75)** — same failure
  mode, different model.

If you need to sweep multiple experiments, run them **sequentially** on one GPU or
**in parallel across separate GPUs** (one per pod). The wall-clock cost of serial
execution on one GPU is lower than the debugging cost of an OOM mid-grid.

---

## Gotchas

- **FA2 cells will fail unless `flash-attn` is installed.** Neither `uv sync` nor
  `requirements-dev.txt` installs it — the wheel is CUDA-version-sensitive and slow to build.
  If a YAML lists `fa2` in `backends:`, install FA2 explicitly per
  [#70](https://github.com/AlexKay28/ddi/issues/70), or skip those cells with
  `make exp-performance MODEL=… BACKENDS=sdpa`.
- **`HF_HOME` must be a persistent volume.** Ephemeral container storage means every
  run re-fetches the model; the `HF_HUB_OFFLINE=1` guard in step 4 will then fail
  fast, which is the intended signal.
- **`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set by the operator, not the
  Makefile.** If the drivers set them for you, a cache miss becomes a silent
  re-download instead of an error.
- **A model with no chat template cannot be evaluated by this harness.**
  `ddi/prompt.py::build_query_text` goes through `tokenizer.apply_chat_template`, so a base
  checkpoint without one raises before the first forward pass. Conversely, a reasoning-tuned
  template that opens a `<think>` block will burn the whole `max_new_tokens` budget on
  reasoning and score `em == 0` with non-zero `presence` — `ddi.prompt.thinking_disabled`
  detects that from the template source and passes `enable_thinking=False`.
- **A GPU is required and the runner will not fall back.** `require_gpu:` defaults to true and
  `ddi.run`'s CLI enforces it, so a sweep cannot quietly run on CPU and emit rows
  indistinguishable from GPU rows. `experiments/smoke.yaml` is the only config that opts out.
- **Gated model access.** `Llama-3.1-8B-Instruct` requires an HF token with access
  granted on the model page — relevant only when *adding* it to the YT mirror. Set
  `HF_TOKEN=...` for that `hf-fetch` step. Not applicable to `yt-fetch`, which never talks to
  the HF Hub.
