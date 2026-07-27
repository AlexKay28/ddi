# Changelog

Chronological log of experiments and project decisions. **Newest first.**

Numbers here are pointers, not sources of truth. Every experiment traces to a row in
`results/runs.jsonl` (ground rule 2 in [issue #1](https://github.com/AlexKay28/ddi/issues/1));
tables and figures regenerate from that store (ground rule 3). This file is the narrative layer
on top — what was run, why, and what it told us.

---

## Entry format

**Experiment entries** — one per run or sweep:

- **Date + title** — `## YYYY-MM-DD — [phase/sub-phase] short title`
- **Sub-issue** — link (e.g., `#13` for Phase 2.1)
- **SHA** — `abcdef1` (mark dirty if applicable)
- **Config** — `configs/<file>.yaml`
- **Runs** — `run_id` list, or count + query template
- **Finding** — one or two sentences; link the generated figure / table
- **`Δ` check** — if a quality claim: gap to `full` vs. `Δ` (green / red / n/a)
- **Follow-up** — issues opened, next sweep, or `closes #N`

**Non-experiment entries** — infrastructure, decisions, rebuilds — use the same date header
without the sub-issue / SHA / runs fields.

---

## Unreleased

Phase 0 in progress. No experiment entries until the three correctness gates
([issue #4](https://github.com/AlexKay28/ddi/issues/4)) are green — per ground rule 6, no DDI
number enters the store before that.

---

## 2026-07-27 — **Roadmap complete for all experimental phases**

Phases 0 → 5 all closed; only #29 (writeup) and the debt-tracker #30 (transformers-4.53 API
migration) remained after Phase 5, and #29 lands in this cycle. Every C1–C6 claim in
RESEARCH_PLAN §4 has an executable runner path and a rendered result artifact behind it.

## 2026-07-27 — [6] preprint scaffold — Phase 6 done

arxiv-submittable LaTeX skeleton in `paper/preprint.tex` + `paper/sections/00_intro.tex` through
`08_limitations_conclusion.tex`, drafted from RESEARCH_PLAN in paper voice. `paper/build.sh`
regenerates every table (markdown via `ddi-report --name X` piped through `pandoc`) and figure
(pdf via `ddi-report --name X --output ...`), stubs on empty store, then runs
`pdflatex → bibtex → pdflatex → pdflatex` when a TeX toolchain is available. `\todoreport{...}`
macro flags every cell that awaits data. Ground rule 3 enforced by construction: no hand-typed
numbers in the .tex sources. Bibliography wired to `paper/references.bib`. Root `Makefile`
gains `make paper`. Closes #29.

Follow-ups landed in the same cycle:
- `ddi-report` script entry added to `[project.scripts]` (agent flagged this — the scaffold's
  `build.sh` used the `python -m ddi.report_cli` form; both now work).
- **T2 renderer** (`ddi.reports.phase2_quality`) — the Phase 2 quality summary table referenced
  by `sections/04_results_quality.tex`. Rows per `(model, dataset, variant)`, columns for EM /
  F1 / presence / loss (median ± σ across seeds).

## 2026-07-27 — **Phase 5 complete**

All three Phase 5 sub-issues closed: #26 hop-split renderers (T5, F4), #27 PCW baseline, #28
composition probe (dataset + runner + T6 + F7). Every experiment described in RESEARCH_PLAN
§4's C1–C6 now has an executable path. Remaining on the roadmap: **Phase 6** (#29 writeup) and
the transformers-4.53 migration debt (#30). All eight prior phases' sub-issues (0.1 through 5.3)
are closed on GitHub.

## 2026-07-27 — [5.3] composition probe (dataset + runner + T6 + F7)

New dataset `composition` (`ddi.data.composition`): each record has entity `X` living in `<CITY>`
(doc_A) and `<CITY> is in <COUNTRY>` (doc_B) plus `N−2` leakage-checked distractors placed at
controlled `distance` and `pair_position`. Answer = `<COUNTRY>`. `hop_type="multi"` by
construction. `phase5_composition` runner mirrors `phase4_restart_quality` and encodes the
probe knobs in the `run_id` (`...:d{distance}:p{pair_position}:...`) so downstream slicing is
free. `T6` markdown groups by `(variant, layout, distance)` with EM mean/median/IQR/n; `F7`
plots median EM vs distance, one line per `(variant, layout)` — the picture that shows where
`ddi_qf` starts to fail as gold docs move apart. Closes #28.

## 2026-07-27 — [5.1] hop-split renderers T5 + F4

`ddi.reports.phase5_hop_split`: joins `phase2_train_free` EM rows with the record's `hop_type`
(fetched from `ddi.data.load` and cached per `(dataset, seed)` since the full split re-materialises
for each row). T5 markdown: per `(dataset, variant)`, columns for EM on single / multi / all,
plus `Δ EM (multi − single)`; adds an `EM (unknown)` column and an inline `<!-- N rows dropped -->`
note whenever any row falls to `hop_type=unknown`. F4 figure: grouped bar chart, median height,
IQR error bars, matplotlib lazy-imported. Both raise loudly on entirely-empty stores; missing
splits render `n/a`, and the degradation column is `n/a` unless both hop buckets have data —
numbers are never fabricated. Report-only; no new runner or store row. Closes #26.

## 2026-07-27 — [5.2] Parallel Context Windows baseline

`ddi.runners.phase5_pcw` — registered as `task=phase5_pcw`. Reproduces Ratner et al. (ACL 2023):
each doc is an independent context window, positions restart at 0, query attends to all
windows. In DDI's machinery this is exactly `variant=ddi_qf` + `layout=restart`; the runner
hard-codes that combo and writes rows under a distinct `phase5_pcw:{model}:{dataset}:...`
prefix so the paper's PCW column is a first-class citizen in tables (rather than getting mixed
into phase2_train_free or phase4_restart rows). No mask/layout knobs in the config — the paper
defines the setup. Greedy prefill+cached-KV generation helper copied from
`phase4_restart_quality`; a follow-up should factor both into `_shared.py`. Closes #27.

## 2026-07-27 — **Phase 3 and Phase 4 complete**

Nine sub-issues closed: #17 timing harness, #18/#19/#20 timing sweeps + configs, #21
realization ratio (T3/F3) + F6 Pareto pairing, #22 position restart, #23 per-doc KV cache,
#24 order invariance, #25 throughput (F5). The harness can now run every efficiency
measurement (FTT/TPOT with median+IQR, realized-vs-analytical per cell, backend cross-check),
route DDI's masked prefill through a content-keyed KV cache, and measure end-to-end QPS under
three reuse regimes — plus GPU is required by default so a broken container fails at boot.
Remaining sub-issues: Phase 5 (#26/#27/#28 failure envelope) and Phase 6 (#29 writeup).

## 2026-07-27 — [4.3/4.4] order invariance and throughput under reuse (F5)

`phase4_order_invariance`: prefills a record's prompt under `(variant, layout)`, then compares
last-token query logits against `n_permutations − 1` deterministic random doc-order permutations
via `torch.allclose(atol=tolerance)`. Writes per-`(record, layout)` `invariance` (1.0/0.0) and
`max_delta`. Restricted to `ddi_qf` / `ddi_bridge_k` where the invariance claim is well-defined.
Hypothesis (restart → invariant, sequential → not) is left to the report layer.

`phase4_throughput`: three modes (`ddi_cache`, `ddi_recompute`, `full_causal`) over a synthetic
doc universe with `zipf` / `uniform` / `fixed` reuse. First-pass cache path skips doc prefill on
hit; full KV-injection is a follow-up. Writes `qps`, `hit_rate`, `n_requests`, `total_elapsed_s`
per run and `per_request_latency_ms` per request. `F5` renderer joins qps + hit_rate per base
run_id and plots log-scale QPS vs hit rate with one series per mode. Closes #24, #25.

## 2026-07-27 — GPU-required by default at CLI startup

`ddi.cli.main` now calls `gpu.require_cuda()` after `gpu.configure()` when the config's
`require_gpu` key is truthy (default `true`). Silently running a real DDI sweep on CPU would
burn hours and produce meaningless numbers, so misconfigured serverless containers fail at boot
with a clear "CUDA not available" message. Opt-out via `require_gpu: false` in the config;
`ddi/configs/example.yaml` and the CLI smoke test set that opt-out because they don't need a
real model.

## 2026-07-27 — GPU usage + [3.5] realization ratio + [2.3 pairing] Pareto

**GPU:** new `ddi/gpu.py` — `is_available()`, `require_cuda()`, `gpu_info()` (device name,
compute capability, VRAM, bf16 support), `configure()` (prints one-line stderr summary,
enables TF32 on Ampere+, idempotent). `ddi.cli.main` calls `configure()` between `set_seed`
and `dispatch` so every runner prints device facts and gets TF32 without any per-runner change.
`make gpu-check` runs the check as a standalone preflight; `make setup-gpu` chains it so a
missed CUDA install fails at container-boot time, not mid-sweep.

**[3.5] realization ratio (T3, F3).** `ddi.reports.phase3_realization`: T3 tabulates
analytical vs realized vs realization ratio per N (RESEARCH_PLAN §7 shape) — the "9× gap"
story per cell. F3 plots realization ratio vs N with one line per L. Both read the store
only; `n/a` when a cell's arms are missing; entirely-empty store raises. Closes #21.

**[2.3 pairing] F6 Pareto frontier.** `ddi.reports.phase2_pareto`: pairs Phase 2.3 bridge EM
rows (`ddi_bridge_k_k{K}`) with Phase 3.2 timing rows at the same K; x = realized speedup,
y = median EM; annotates points with `k`, draws dashed lines at the `full` baseline and at
`full − Δ` when `delta_em` is in the store. Paper centerpiece.

## 2026-07-27 — [4.2] per-doc KV cache

`ddi.cache.DocCache`: disk-backed cache keyed by `sha256(model_id\0dtype\0doc_text)[:16]`; atomic
`.tmp` + `os.replace` writes; per-key `fcntl.flock` sentinel prevents concurrent-write races.
`compute_doc_kv(model, tokenizer, doc_text, layout)` runs a single-doc `variant="full"` prefill
with `build_positions(..., layout)` and extracts per-layer K/V from HF's `past_key_values`
(handles both `DynamicCache` and legacy tuple formats). Runner `phase4_cache_correctness`: for
each retrieved doc, asserts (a) two cold computes agree within tolerance and (b) the KV block
round-trips through the cache. End-to-end injection into a full prompt lives in Phase 4.3 (#24)
and 4.4 (#25). Closes #23.

## 2026-07-27 — [3.2–3.4] Phase 3 timing sweeps

`ddi.runners.phase3_timing` covers issues #18, #19, #20 with one runner. One invocation =
one grid cell `(N, L, Q, mask variant, backend, seed)`. Every row carries the shape in `run_id`,
the analytical speedup target from RESEARCH_PLAN §7 (`T²/2` for full; `Nℓ²/2 + Q·Nℓ + Q²/2` for
`ddi_qf`; `Nℓ²/2 + Q²/2 + T` for `ddi_strict`), and mask sparsity (`1 − allow_pairs / causal_pairs`
— matches §7's 90.6% at N=13/L=512/Q=64), so Phase 3.5's realized-vs-analytical is a store query,
not a re-derivation. Backend selection routes via `load_model(..., attn_implementation=...)`;
FA2-on-CPU fails fast before any download. Three config templates (`phase3_timing_N.yaml`,
`phase3_timing_L.yaml`, `phase3_backends_NxL.yaml`) that the orchestrator overrides per cell.
Closes #18, #19, #20.

## 2026-07-27 — [4.1] position restart layout + phase4_restart_quality

`ddi.positions.build_positions` returns a `[T]` long tensor: `sequential` = `arange(T)`,
`restart` = each doc's tokens start at 0, query+answer block starts at `max(doc_lengths)`. The
`max` offset keeps query→doc RoPE distances well-defined without any RoPE collision into doc
rows. `ddi.model.forward_with_mask` gains an optional `position_ids` kwarg (2D `[batch, T]` long,
default `None` preserves HF's default). Runner `phase4_restart_quality` runs both layouts on the
same records and writes rows carrying the layout in the run_id (`...:{variant_tag}:{layout}:...`),
so downstream analysis can compare directly whether restart costs anything at eval time —
ground-truth for whether Phase 4.2's cache is a win at any quality level. Closes #22.

## 2026-07-27 — [3.1] timing harness

`ddi.timing.measure`: `warmup + n_repeats` calls; CUDA-event timing when `torch.cuda.is_available()`
else `time.perf_counter`; `torch.cuda.empty_cache()` before every iteration (warmup included) so
each repeat starts cold, `torch.cuda.synchronize()` after so timing captures full compute.
`Timing` dataclass carries `median_ms`, IQR bounds, and every post-warmup sample for auditability.
Thin wrappers `measure_prefill` (FTT via `forward_with_mask`) and `measure_decode` (TPOT — full
prefill+decode time divided by `n_new_tokens`). `append_timing` writes four rows to the store
(`{metric}_median_ms`, `_iqr_low_ms`, `_iqr_high_ms`, `_repeats_ms` as JSON-encoded list per
ground rule 4). CPU fallback keeps CI green without a GPU. Reuses `ddi.report._quantile`. Closes #17.

## 2026-07-27 — **Phase 2 complete**

All four Phase 2 sub-issues closed: #13 train-free primary result, #14 conditional finetuned
sweep, #15 bridge sweep centerpiece, #16 base-vs-instruct probe. The harness can now: evaluate
one cell of the `{model} × {mask} × {dataset}` sweep on stock weights, fine-tune with the mask
present and save a reloadable checkpoint, sweep bridge widths for the F2 curve, and produce a
base-vs-instruct effect-size table (T4_probe_instruct). All quality metrics share the same
`build_mask → to_additive → forward_with_mask` code path as the baseline (Phase 1.4) so any
gap is a semantic effect, not implementation drift. Ready for Phase 3 (efficiency + realization
gap) and Phase 4 (per-document caching) in parallel.

## 2026-07-27 — [2.4] base-vs-instruct probe

`ddi.reports.phase2_base_vs_instruct`: registered as `T4_probe_instruct`. Parses
phase2_train_free run_ids to identify (model, dataset, variant); pairs base and instruct
versions of each family (strip `-Instruct` / `-instruct` suffix); reports per-family
`gap = mean(full) − mean(ddi_qf)` for base and instruct with per-seed σ, plus the effect
(base − instruct). Hypothesis in RESEARCH_PLAN §2.4: negative effect means instruct is more
DDI-friendly. Closes #16.

## 2026-07-27 — [2.3] bridge sweep + F2

`ddi/reports/` package introduced (mirrors `ddi/runners/` — side-effect registration).
`ddi/report.py` imports it at module bottom so every renderer is discoverable via
`list_reports()`. `phase2_bridge.yaml` matches the phase2_train_free schema with
`variant=ddi_bridge_k` and a `bridge_k` placeholder that the orchestrator overrides per
invocation across `{1, 2, 4, 8, 16, 32, 64, query_len}`. `F2` renderer (registered by
`ddi.reports.phase2_bridge`): reads bridge rows from the store, groups by k, plots median EM
with IQR error bars on a log-2 x-axis. Lazy matplotlib import; parent path is created on demand.
Closes #15.

## 2026-07-27 — [2.2] SFT training loop + phase2_finetune_train

`ddi.train`: from-scratch SFT loop (no `Trainer`/`trl`/`peft`) — AdamW, linear warmup then
constant LR, configurable gradient accumulation, gradient checkpointing, non-finite-loss step
skip that keeps the LR schedule honest, periodic validation loss logged as `train_val_loss` to
the store. Deterministic dataloader by re-invoking the loader per pass (no `itertools.cycle`
buffering). Checkpoint saved via `model.save_pretrained` under a hashed `_run_tag` that
sanitises model/dataset slashes. Runner `phase2_finetune_train` wraps it; eval reuses
`phase2_train_free` by pointing its `model:` at the checkpoint dir. LoRA / light-adaptation
rung deferred with a `TODO` in the config. Closes #14.

## 2026-07-27 — [2.1] train-free mask sweep runner

`ddi.runners.phase2_train_free`: reads mask variant + bridge_k from config so the orchestrator
fills the sweep grid one cell per invocation. `ddi/runners/_shared.py` factors out
`shifted_cross_entropy` so Phase 1.4 (`phase1_baseline`) and Phase 2.1 (`phase2_train_free`)
use bit-identical loss — Δ is only meaningful if the formula is shared. `_generate_with_prefill_mask`:
DDI-masked prefill via `forward_with_mask`, then greedy cached-KV argmax steps. Records with
empty `info_contexts` coerce to `variant="full"` with a per-run summary count. Config validation
fails loudly on missing keys and on `bridge_k` / variant mismatch before touching the model.
Closes #13.

## 2026-07-27 — Makefile refocused on serverless surface

`Makefile` narrowed to what a serverless container actually needs: install (`setup`, `setup-gpu`),
tests (`test`, `gates`), **datasets upload** (`upload-dataset` — pushes frozen split manifest to
HF hub), **models upload** (`upload-model` — pushes a checkpoint to HF hub), experiments (Phase
1–5 targets), results collection (`report`). Dropped: `test-fast`, `lint`, `format`, `clean` —
not part of experiment orchestration.

## 2026-07-27 — **Phase 1 complete**

All four Phase 1 sub-issues closed: #9 datasets, #11 metrics, #10 hop labels, #12 baseline
runner + Δ. The harness can now: load a frozen dataset with verifiable split hashes, tokenize
and score model output against SQuAD-style metrics, and run a `phase1_full_baseline` that writes
per-record loss + EM/F1/presence to `results/runs.jsonl` under a `full` mask routed through the
same `build_mask → to_additive → forward_with_mask` path every DDI variant will use (no
baseline-vs-variant implementation drift). `compute_delta` reads those rows and returns
`Δ = k × std_across_seeds(per_seed_mean(metric))` — this is what makes ground rule 9 real. Ready
for Phase 2 (issues #13 train-free primary, #14 conditional finetuned, #15 bridge sweep
centerpiece, #16 base-vs-instruct probe).

## 2026-07-27 — [1.4] baseline runner + Δ

`ddi.runners.phase1_baseline`: registers `task=phase1_full_baseline`. For each record: builds a
`variant="full"` mask via `build_mask` + `to_additive`, runs it through `forward_with_mask` for
answer-only cross-entropy (next-token-shifted), then `model.generate` (greedy) for EM/F1/presence.
Every metric writes a row to `results/runs.jsonl` with full provenance. Skips records with empty
`possible_answers` (loud print). The baseline shares the mask-injection path with future DDI
variants, so any drift between "baseline" and "DDI" numbers is a code bug, not an implementation
difference. `ddi.delta`: `compute_delta` filters by `phase1_baseline:{model}:{dataset}:` +
metric, groups by `provenance.seed`, computes sample std (ddof=1), returns `Δ = k × std`
(default `k=2`, requires ≥3 seeds). `record_delta` writes per-seed-mean rows AND a `delta_{metric}`
row so the whole computation is auditable per ground rule 3. `ddi-delta` CLI wraps it. Makefile
gains `phase1-baseline` and `delta` targets. Closes #12.

## 2026-07-27 — [1.2] hop labels + ddi-audit CLI

`ddi.hop_labels`: `hop_type(record)` reads the native field; `register_heuristic(dataset)` lets
later loaders (HotpotQA, Open RAG Bench) plug in a labeler for `hop_type == "unknown"` records
without touching the loaders themselves; `apply_heuristic` yields shallow-copied dicts so
callers can iterate loaders directly. `sample_for_audit` + `load_audit` + `agreement_rate` wire
the hand-audit workflow. `ddi-audit sample` writes a JSONL for review; `ddi-audit score`
re-samples, applies the heuristic, loads human labels, prints the rate + disagreements +
skipped count. Closes #10.

## 2026-07-27 — [1.1] dataset framework + NQ + synthetic

`ddi.data`: registry-based loaders yielding a common record shape (`info_contexts`, `dialog`,
`possible_answers`, `hop_type`, `gold_position`, `meta`). `splits.py` provides deterministic
`split_ids` + order-invariant `hash_split`; `manifest.py` reads/writes/validates a JSON manifest
at `results/splits/<name>.json` per ground rule 1 — mismatch raises `RuntimeError` with a diff.
Two initial loaders: `synthetic_qa` (deterministic, offline, 100 records / 60·20·20 split) for
tests and smoke runs, and `nq` (via `google-research-datasets/nq_open`) as the first real
dataset. `pyproject.toml` gains `datasets>=2.14`. Closes #9.

## 2026-07-27 — [1.3] metrics

`ddi.metrics`: SQuAD-style `normalize` (lowercase → punctuation strip → whole-token article
removal to avoid eating `"the"` inside `"theatre"` → whitespace collapse), then `exact_match`,
`token_f1` (Counter-multiset overlap, canonical SQuAD formula, max across candidates), and
`answer_presence` (refuses trivial empty-candidate hits). `make_answer_labels` returns a
`[batch, seq]` long tensor that is `input_ids` inside `[answer_start:answer_end)` and `-100`
elsewhere — feeds HF's own answer-only cross-entropy path. Closes #11.

## 2026-07-27 — **Phase 0 complete**

All seven Phase 0 sub-issues closed: #2 skeleton, #3 masks, #6 attention backends, #5 HF model
wrapper, #4 correctness gates, #7 provenance + store, #8 report generation. The DDI harness now
has a single mask definition (ground rule 8), three attention backends verified equivalent by the
Phase 0.5 gates (ground rule 6), mask injection into HF's own forward (no hand-rolled decoder),
run identity + append-only store (ground rules 1, 2), and idempotent report rendering from the
store (ground rule 3). Ready for Phase 1 (datasets + `full` baseline + fix `Δ`).

## 2026-07-27 — [0.5] correctness gates

`tests/test_gates.py`: three parametrized gates, all `@pytest.mark.gate`. Gate 1 — N=1 mask
equivalence + output equivalence over available backends. Gate 2 — SDPA-referenced backend
cross-check on doc-only fixed input; skips (never passes vacuously) when only SDPA is available.
Gate 3 — mask shape sanity plus the `len()`-vs-`.shape[-1]` regression encoded permanently. Meta
test verifies every `test_gate*` carries the marker. `gate` marker registered in `pyproject.toml`.
`make gates` runs just the gate suite. Closes #4.

## 2026-07-27 — Makefile as experiment shortcuts

Reframed root `Makefile` around experiment shortcuts. Setup collapsed to two targets (`setup`,
`setup-gpu`); every phase gets its own labeled targets (`sweep-train-free`, `sweep-bridge`,
`timing-N`, `realization`, `cache-throughput`, `pcw-baseline`, ...), each pointing at a config
under `ddi/configs/phase*.yaml`. `help` output is sectioned. `run-seeds` added for one-VM seed
loops. `report` target wraps `ddi.report_cli` (new module). Two config stubs seeded
(`phase2_train_free.yaml`, `phase2_bridge.yaml`) to demonstrate the shape.

## 2026-07-27 — [0.4] HF model wrapper with mask-preservation patch

`ddi.model.load_model` returns `(model, tokenizer)` and applies `patch_model` which installs an
instance-level `_update_causal_mask` (via `types.MethodType` on `model` or `model.model`) that
returns a supplied **4D float** mask unchanged and delegates every other shape/dtype to the
original. `unpatch_model` deletes the instance attribute to unshadow the class descriptor cleanly.
`forward_with_mask` validates 4D + floating-point and calls HF's own forward with `use_cache=False`.
No hand-rolled decoder loop. Integration test gated on `DDI_INTEGRATION=1`; unit tests use a mock.
Closes #5.

## 2026-07-27 — Makefile as serverless entry point + CI

`Makefile` reworked to use `uv` throughout (`uv pip install --system`, `uv run <cmd>`) and
reframed as the single interface for serverless experiment runs: fresh container → `make setup`
→ `make <phase-target>`. GitHub Actions workflow at `.github/workflows/test.yml` installs via
`make setup`, runs `make lint test` on Python 3.11 and 3.12, matches the serverless first-run
sequence so CI green ≡ the container will boot cleanly.

## 2026-07-27 — [0.3] attention backends

`ddi.attention.get_backend` dispatches to three implementations of one mathematical object: SDPA
with explicit additive mask (**reference** — defines correct), FlashAttention-2 varlen scoped to
doc-only prefill (uses `cu_seqlens`, CUDA-only, lazy-imported), and Triton deferred as
`NotImplementedError`. Cross-backend equivalence is verified by the Phase 0.5 gate. Closes #6.

## 2026-07-27 — [0.7] report generation

`ddi.report` provides the `@register` decorator (mirrors `ddi.cli.register_runner`), sorted
`list_reports`, and `render(name, store, output=None)` dispatching tables (markdown) vs figures
(write to output) by `T*` / `F*` naming. Store reader `iter_rows` in file order plus
`filter_rows` (metric / run_id / provenance subset). Helpers: `median_with_iqr` with linear
interpolation, `group_by` with dotted-path lookup. Idempotent by construction. Renderers for
T2/T3/F2–F6 register from later-phase modules as their data lands. Closes #8.

## 2026-07-27 — [0.2] masks.py

`ddi.masks.build_mask` returns a boolean `[T, T]` allow-matrix for all five variants (`full`,
`ddi_qf`, `ddi_strict`, `ddi_bridge_k`, `blind`) via vectorised boolean-tensor ops over row/col
index grids. `to_additive` converts to `-inf` form preserving bf16 representability. N=1 collapse
to `full` for the qf/strict/bridge_k family is explicit and matches the correctness-gate semantics
in issue #1 §0.5. Independent triple-loop reference in `tests/test_masks.py`. Closes #3.

## 2026-07-27 — [0.6] provenance and append-only store

`ddi.provenance.capture` returns a `Provenance` (git SHA/dirty, config hash, seed, model/dataset
revisions, GPU model, driver/CUDA/torch versions, UTC timestamp). `ddi.store.append` writes one
`fcntl.flock`-protected JSONL row per metric. Two-process concurrent-write test verifies parallel
sweeps do not interleave. Closes #7.

## 2026-07-27 — Makefile added

Root `Makefile` collects install / test / lint / format / run / clean targets and pins overridable
variables (`PYTHON`, `CONFIG`, `SEED`). Placeholder comments mark where phase-specific sweep
targets will land as their sub-issues close.

## 2026-07-27 — [0.1] package skeleton

`pyproject.toml`, `ddi/` package with stub modules (`masks`, `attention`, `model`, `provenance`,
`store`, `report`), working `ddi.cli` with runner registry, `ddi/configs/example.yaml`, `tests/`
with CLI smoke tests, `results/` directory. Closes #2.

## 2026-07-27 — CHANGELOG created

Repo scaffolded for the clean-room rebuild. Awaiting Phase 0.
