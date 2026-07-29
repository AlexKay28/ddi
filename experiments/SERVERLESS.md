# experiments/SERVERLESS.md

One-page recipe for running a DDI experiment on a serverless GPU (RunPod, Modal,
Vast, Lambda, etc.). The design assumes a persistent volume for HF weights and one
experiment per GPU.

For the index of what each YAML measures, see [`README.md`](README.md).

---

## Recipe

### 1. Container base

```
nvidia/cuda:12.1.1-cudnn-runtime-ubuntu22.04
```

Chosen to match the CUDA 12.1 pin used by the project's torch build (`torch==...+cu121`
in `pyproject.toml` / `uv.lock`). Any newer CUDA runtime image works if it stays ABI
compatible with the pinned torch wheel; the safest choice is this exact tag.

### 2. Clone and sync

```bash
git clone https://github.com/AlexKay28/ddi.git
cd ddi
uv sync
```

`uv sync` installs from `uv.lock` — no separate `pip install` step, and the resolver
is deterministic across runs. If the container image doesn't ship `uv`, install it
first with `pip install uv` or the standalone installer.

### 3. Prefetch the model weights (once per model)

```bash
HF_HOME=/mnt/models make prefetch MODEL=Qwen/Qwen2.5-7B-Instruct
```

Run this **once**, pointed at a persistent volume mounted at `/mnt/models`. The
`prefetch` target downloads the tokenizer + weights into `$HF_HOME`. Subsequent runs
that also set `HF_HOME=/mnt/models` will read from the volume instead of hitting the
HF hub.

Do this for every model an experiment references (both C1 YAMLs share Qwen and Llama
instruct; the C3 YAMLs use the same two).

**If `HF_HOME` is not a persistent volume,** the prefetch is thrown away when the
container exits and every run re-downloads ~15 GB. Mount the volume.

### 4. Run the experiment (offline)

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/mnt/models \
  make exp-qwen_instruct_c1
```

The two `*_OFFLINE=1` flags force `transformers` and `huggingface_hub` to fail loudly
if any code path tries a network call — this catches "silently re-downloading on
every run" bugs. They are set by the **operator**, not by the Makefile, so a
misconfigured cache surfaces immediately rather than being papered over.

The runner writes rows to `results/runs.jsonl` and emits
`logs/qwen_instruct_c1_summary.md` from the YAML's `report:` block.

### 5. Publish the summary to the tracking issue

```bash
make publish-summary SUMMARY=logs/qwen_instruct_c1_summary.md ISSUE=74
```

This posts the rendered Markdown (tables + provenance footer per
[#79](https://github.com/AlexKay28/ddi/issues/79)) as a comment on the tracking
GitHub issue. The mapping from YAML to issue number is in
[`README.md`](README.md#index).

---

## One experiment per GPU

**Convention:** run one `make exp-<name>` at a time on a given GPU. Do not launch a
timing grid and a quality grid concurrently on the same H100. Do not fan out inside a
YAML in a way that keeps two model copies resident.

This is not enforced by code — the runner will happily start a second grid if you
launch one — but the two most recent operator failures both traced back to violating
it:

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

- **FA2 cells will fail unless `flash-attn` is installed.** The default `uv sync`
  does not install it — the wheel is CUDA-version-sensitive and slow to build. If
  the YAML lists `fa2` in `backends:`, install FA2 explicitly on the container per
  [#70](https://github.com/AlexKay28/ddi/issues/70). Otherwise those cells will
  raise at import time.
- **`HF_HOME` must be a persistent volume.** Ephemeral container storage means every
  run re-downloads the model; the `HF_HUB_OFFLINE=1` guard in step 4 will then fail
  fast, which is the intended signal.
- **`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set by the operator, not the
  Makefile.** If the drivers set them for you, a cache miss becomes a silent
  re-download instead of an error.
- **Gated model access.** `Llama-3.1-8B-Instruct` requires an HF token with access
  granted on the model page. Set `HF_TOKEN=...` for the `prefetch` step only; runs
  themselves are fully offline.
