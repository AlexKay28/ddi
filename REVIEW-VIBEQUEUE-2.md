# VIBEQUEUE-2 — methodology and harness review

Scope: the research plan in VIBEQUEUE-2 and the `ddi/` harness that implements it.
Question asked: *can we fine-tune a model so that it performs the same under the
`ddi_qf` mask as under `full`* — and is the current setup capable of finding that
answer, or of recognising it if it happened?

**Verdict: no, not as it stands.** Three independent defects each break the
measurement on their own, and all three push in the same direction — they shrink the
difference between "the model can read the documents" and "it cannot" until nothing
is measurable. Two of them make a *false success* the most likely reported outcome.
All are fixed below; the numbers currently in the ticket should be discarded.

---

## 1. The gold methodology

### 1.1 What the claim actually is

"`ddi_qf == full`" is two conditions, not one. Reporting only the first is how a
broken model passes:

| | condition | catches |
|---|---|---|
| **Parity** | tuned model: `ddi_qf` ≈ `full` | the mask costing quality |
| **Preservation** | tuned model under `full` ≈ **base** model under `full` | a model that "achieved parity" by getting worse at everything |

A model that collapses under `full` down to its own `ddi_qf` level satisfies parity
perfectly and is worthless. Only preservation catches it.

### 1.2 The measurement

Per (dataset, metric), with every value oriented so higher is better (`loss` negated):

```
S = full_base − blind_base                      headroom: what the documents are worth
P = (full_tuned  − ddi_qf_tuned) / S            parity gap        → 0
R = (base_full   − full_tuned)   / S            preservation gap  → 0
E = (base_full   − ddi_qf_tuned) / S            end-to-end gap    → 0
L = (blind_tuned − blind_base)   / S            leakage           → 0
```

`E = P + R` identically, so every failure is attributable to one half or the other.
As ratios: `closure = 1 − P`, `preservation = 1 − R`, both targeting **1.0**.

Decision table:

| P | R | verdict |
|---|---|---|
| ≈ 0 | ≈ 0 | **success** — the mask is free and the model is intact |
| ≈ 0 | > 0 | parity by regression — not bankable |
| > 0 | ≈ 0 | no effect — model fine, mask still costs quality |
| > 0 | > 0 | harmful |
| any | any, `L` ≠ 0 | **invalid** — `blind` moved, audit the split |

Implemented in `ddi/parity.py`; `python -m ddi.parity --store … --base-model … --tuned-model …`.

### 1.3 The admissibility gate — the part that was missing

**`S` is the denominator of everything, and a cell where `S` is at the noise floor
cannot support a parity claim in either direction.** All four ratios become
noise/noise, swing wildly, and look like signal. Before any ratio is quoted:

- `S` must exceed `k·Δ` (default k=3), with Δ from `ddi.delta.compute_delta` — the
  repo's own ground rule 9: *"before any DDI number can be called parity or
  non-parity, Δ must be fixed from the `full` baseline's seed-to-seed variance
  alone."*
- With no Δ available, an absolute floor (`min_span`, default 0.05) applies.
- A cell failing the gate is reported as `INADMISSIBLE`, not as a number.

### 1.4 What a healthy measurement looks like

Sanity expectations on the **base** model, before any training:

| mask | expectation | if violated |
|---|---|---|
| `full` | clearly the best; `em` well off the floor on hotpot/musique | prompt or answer-extraction is broken, not the model |
| `ddi_qf` | between `full` and `blind` — **the gap being studied must exist** | nothing to close; the experiment is vacuous at this operating point |
| `blind` | clearly worst — it cannot see documents at all | the mask is leaking; every parity number is invalid |

If `full ≈ blind`, the dataset is not testing document use and no amount of
fine-tuning will produce an interpretable result.

---

## 2. Are the metrics adequate? — **No**

From the ticket's own row 3 (500 records, `ddi_qf-all4`):

| metric | full | ddi_qf | blind | headroom `S` |
|---|---|---|---|---|
| em (all datasets) | 0.0000 | 0.0000 | 0.0000 | **0.0000** |
| f1 (hotpot_qa) | 0.0432 | 0.0416 | 0.0415 | **0.0017** |
| presence (hotpot_qa) | 0.1980 | 0.1900 | 0.1920 | **0.0060** |
| loss (hotpot_qa) | 5.6881 | 5.7122 | 7.2756 | 1.5875 |

`em` is a hard zero in **every** cell — 4 datasets × 3 masks × 2 models × 500
records. `f1` and `presence` have headroom smaller than any plausible Δ. Only `loss`
has real dynamic range, and it is a proxy: a cross-entropy gap can close while the
model still never emits the answer string.

Two consequences:

1. No quality-metric parity claim is currently possible in either direction.
2. On the one metric that does have range, **the gap is already closed before any
   training**. Closure `(blind − ddi_qf)/(blind − full)` on loss: base model on
   kilt_nq **99.9 %** (row 5: 8.8656 / 8.8659 / 9.2973); tuned model 98.5 % hotpot,
   99.8 % musique, 100.1 % kilt_nq, 98.6 % synthetic. There is nothing for the
   finetune to close at this operating point, so row 2 is a no-op — not a failed
   method.

Also anomalous: on synthetic_qa row 3, `full` scores **worse than `blind`** (f1
0.0000 vs 0.0111, presence 0.0000 vs 0.1500). That ordering is impossible if the
documents help and the masks are correct. It is consistent with issue **R2** below.
Note also that synthetic_qa's val split is only **20 records** (issue R7), so
`max_records: 500` was evaluating 20.

---

## 3. Are the metrics computed correctly? — **No**, and the documents are largely not there

This was the specific thing to check, and it is where the review found the worst
defects. Findings by dataset, from the committed record caches in `results/splits/`:

| dataset | val records | documents | healthy? |
|---|---|---|---|
| hotpot_qa | 3702 | 10 real paragraphs on 3668 of them, 300–500 chars each | **yes** |
| musique | — | `"{title}: {paragraph_text}"`, real bodies | yes (loader inspected) |
| synthetic_qa | 20 | 3 × 12 random words + planted answer token | yes, but tiny (R7) |
| kilt_nq | 1418 | **Wikipedia titles, not passages** | **no** (R1) |

---

## 4. Issues

Severity: **C** = breaks the research question outright, **H** = high, **M** = medium.

### R1 — `kilt_nq` documents are Wikipedia titles, not passages · **C** · fixed

`ddi/data/kilt_nq.py::_passage_text` fell back to `f"[{title}]"` when a provenance
entry carried no inline `text` — a placeholder added "so downstream code can still
count passages while a follow-up wires a real wiki lookup". The follow-up never
landed, and `facebook/kilt_tasks` provenance carries **no** inline text (the bodies
live in the separate KILT knowledge source). So every kilt_nq document ever seen by
this harness was a bare title of 6–61 characters:

```
["[Malware]", "[Malware analysis]"]
["[South Asia]", "[Asia]"]
["[2017 Wimbledon Championships – Women's Singles]"]
```

The fallback also silently defeated `_normalize`'s own "drop provenance entries with
no extractable text" filter, which could never fire for a row that had a title.

Effect: kilt_nq is a closed-book dataset dressed as a retrieval dataset. `full` and
`blind` see the same information, `S` collapses, and the measured DDI gap is pinned
near zero for reasons unrelated to the mask. This alone explains kilt_nq's tiny loss
span (0.29 vs hotpot's 1.59) and its 100.1 % closure.

*Fix:* no more placeholder (`DDI_KILT_TITLE_PLACEHOLDER=1` restores it for
reproducing historical rows only); `_check_passage_coverage` raises when under half a
split has passage text, naming the KILT knowledge-source join as the real fix.
**kilt_nq now fails loudly instead of producing fake documents — it is unusable for
this research until the passage join is written.**

### R2 — `blind` is not blind during generation · **C** · fixed

`ddi/inference.py::generate` applied the DDI mask to the **prefill only**. Each
decode step called `model(input_ids=next_token, past_key_values=past)` with no
`attention_mask`, so HF built a default full-causal mask over the whole KV cache and
every generated token after the first could read the documents — regardless of
variant. For `ddi_qf`/`full` this is harmless (`answer_sees_doc` is True anyway,
`masks.py:104`); for `blind` it contradicts `masks.py:101-102`, which sets
`answer_sees_doc` to all-False.

`blind` is the null channel that every parity claim is normalised against, so a leaky
`blind` erases the `full − blind` headroom in exactly the generation metrics (`em`,
`f1`, `presence`) the research needs — while `loss`, computed through a properly
masked single forward, kept showing the real separation. **That disagreement between
the two measurement paths is the fingerprint of this bug**, and it also explains
synthetic_qa's `blind > full` inversion.

*Fix:* the allow-matrix is now built once with `answer_len=max_new_tokens` and the
correct row is sliced per decode step. Verified: the prefill block is bit-identical
to the old mask (so `full`/`ddi_qf` results are unchanged), and `blind` answer rows
now see the query and prior answer tokens but no document tokens.

### R3 — the prompt never mentions the documents, and never asks for an answer · **C** · fixed

The loaders build `dialog = [{user: question}, {assistant: answer}]` — the bare
question. `_quality_cell` then chat-templates that and concatenates the documents as
raw tokens *in front of* the templated conversation, so the model sees wiki text,
then the template's opening (for Qwen, an auto-inserted system block), then a
question that never refers to any documents and never asks for a short span. Greedy,
64 tokens.

`em` scores an exact span match. A verbose chat answer scores 0 by construction —
hence `em = 0.0000` everywhere and `f1` pinned at ~0.04. The synthetic_qa loader had
already discovered this independently; its comment records that without an explicit
"the passages above are the only source of truth" line the models *"never look at the
passages at all"* — and synthetic_qa is the only dataset whose question says so.

*Fix:* new `ddi/prompt.py` is the single source of prompt truth for all three call
sites (`run.py::_quality_cell`, `finetune.py::_encode_record`,
`perplexity.py::compute_perplexity`) — which previously duplicated the same eight
lines and could drift apart. It names the passages and demands a bare span. The
instruction rides inside the query block because `build_mask` requires the documents
to be a prefix; a system turn before them would break that contract.
`DDI_PROMPT_INSTRUCTION=0` restores the old prompt.

### R4 — a quarter of training does nothing · **H** · fixed

`build_mask` collapses the whole `ddi_qf`/`ddi_strict`/`ddi_bridge_k` family to plain
`full` when there is at most one document (`masks.py:79-80`) — correct, there is
nothing to isolate. But then student and teacher masks are *identical*, the KL is
exactly zero, and the record contributes nothing but two 4B-model forward passes.

Measured on the committed caches:

```
kilt_nq_n3 train : 87,372 records — 76,945 single-doc + 10,427 with no documents = 100.0%
kilt_nq_n3 val   :  1,418 records —    788 single-doc                             =  55.6%
hotpot_qa  train : 90,447 records —      0 single-doc                             =   0.0%
```

In a four-dataset round-robin mix, roughly a quarter of all training steps were
guaranteed-zero gradients.

*Fix:* `_prefilter_records(records, min_docs=2)` drops them before DDP sharding (so
ranks stay in lockstep) and logs the count. Config key `min_docs`, default 2;
`min_docs=1` restores the old behaviour.

### R5 — the KL loss is answer-length-weighted, not a mean · **M** · fixed

`F.kl_div(..., reduction="batchmean")` divides by `input.shape[0]`. The logits were
`[1, L, V]`, so it divided by 1 — the loss was the **sum** over the `L` answer
positions, not the mean the comment claimed. Gradient magnitude scaled with answer
length, so long-answer records dominated every accumulation window.

*Fix:* flatten to `[L, V]` before the reduction.

### R6 — no parity metric existed · **H** · fixed

Nothing in the repo computed the closure or preservation ratios, so "did we find it?"
had no answer to read. `ddi/aggregate.py` emits per-mask tables and leaves the
comparison to the reader — which is how "f1 full 0.0414 vs ddi_qf 0.0407, equal!"
becomes a success story.

*Fix:* `ddi/parity.py` (§1.2), including the admissibility gate that refuses to
answer when the headroom is at the noise floor.

### R7 — synthetic_qa is too small, and its document count is hardcoded · **M** · fixed

`_N_RECORDS = 100` with `val_frac = 0.20` gives **20 val records**. A metric over 20
records moves in steps of 0.05 and cannot resolve a parity claim, so synthetic_qa
could not serve as one of the "4 quasi-independent datasets" the plan relies on for
significance. `_DOCS_PER_RECORD = 3` was hardcoded, which the ticket already flagged
as blocking rows 14-15.

*Fix:* `DDI_SYNTHETIC_N_RECORDS` and `DDI_SYNTHETIC_DOCS_PER_RECORD` env knobs.

### R8 — the parity test the repo already had is disabled by the plan · **H** · plan change

`ddi/delta.py` implements exactly the needed non-inferiority margin — `Δ = k·std`
from the `full` baseline's seed-to-seed variance — and its docstring calls this
ground rule 9: parity cannot be claimed without it. It requires `_MIN_SEEDS = 3`.
The ticket drops seeds as an axis, so `compute_delta` raises.

The research question *is* a parity claim, so this rule fires by definition; the
ticket's own escape clause ("if a decision rests on a small delta, re-run that row at
2-3 seeds") is guaranteed to trigger. Since `seed` also selects the train/val/test
partition, three seeds of the **base** model measure split-to-split variance — which
is the right noise floor, and it is three eval-only 1-GPU runs, the cheapest thing in
the grid.

*Not a code change* — a plan change. `ddi/parity.py` accepts a `delta` per cell.

### R9 — base and tuned models can silently average into one cell · **M** · guarded

`ddi.aggregate` keys the model off `run_id` field 1, which is the config's `model:`
string, not the checkpoint (`aggregate.py:97-110`). Row 5 got two separate columns
because it set `model:` to a suffixed name with `checkpoint:` pointing at the
weights. That works — but if any row forgets, both checkpoints land in one cell and
`R` (preservation) comes out 0 **by construction**, i.e. the second half of the claim
always passes.

*Fix:* `compute_parity` refuses when `base_model == tuned_model`, with the reason
spelled out. A `quality_ckpt.yaml` that fails loudly when `checkpoint:` is set and
`model:` is not unique would close it properly.

### R10 — the passage-count axis cannot be run on kilt_nq · **H** · plan change

Rows 5-7 (`DDI_KILT_N_PASSAGES=2/10/20`) carry "the core DDI claim" — the gap should
widen with document count. But KILT provenance is thin: at the N=3 configuration the
val split delivers 1/2/3 passages to 788/420/210 records — 55 % get a single
passage, and **nothing reaches even 3 reliably**. Raising N to 10 or 20 cannot
manufacture provenance entries that are not there, so rows 6-7 as specified would
measure a document count that never changes. Compounded with R1, they would measure
the count of *titles*.

*Recommendation:* move the document-load axis to hotpot_qa (10 real paragraphs on
99 % of records) or to synthetic_qa (now configurable, and the only dataset with
controlled gold placement).

---

## 5. Ticket review

What the plan gets right: the three-mask reading rule, `blind` as a null channel,
`full`-flat as a non-negotiable guard, the honesty about significance without seeds,
the refusal to run a full cross product, and the explicit "stop and ask" triggers.
The step structure is sound.

Where it needs to change:

1. **Step 1's gate is the whole ballgame and is currently failing.** *"If `blind` is
   not far below the other two, the eval harness itself is wrong — fix that before
   training anything."* On the posted numbers `blind ≈ full` on every quality metric.
   The plan already said to stop here.
2. **Add preservation as a first-class condition** (§1.1). The plan states "full flat
   vs base" in prose but nothing computes it, and R9 could make it always pass.
3. **Restore 3 seeds on the base row** (R8) — otherwise "equal" is not a claim.
4. **Reorder.** Rows 6-7 before the training rows 8-13: establish the gap exists at
   high document count before spending 8-GPU jobs closing a gap that measures 99.9 %
   closed. And move that axis off kilt_nq (R10).
5. **Regenerate `baseline.json` after the fixes.** `full_base` and `blind_base` are
   the denominator of every number above; they cannot come from a different harness
   version than the numerator.
6. **Retire the existing result comments.** Rows 3 and 5 were produced with R1+R2+R3
   live. They are not "noisy", they are measuring something else.

---

## 6. Verdict

The plan is looking for the right thing and describes the right guards, but the
harness underneath could not have found it:

- on `em`/`f1`/`presence` there was **no dynamic range at all** (R2 + R3), so parity
  was unmeasurable;
- on `loss`, the only metric with range, the gap **was already closed before
  training** — and one of the four datasets had no documents in the first place (R1),
  which is part of why;
- a quarter of the training compute produced exactly zero gradient (R4);
- and nothing in the repo computed the two-condition criterion (R6), so the most
  likely reported outcome was a **false success** read off two near-identical
  floor-level numbers.

None of that is a statement about the method. It is a statement about the
instrument. After the fixes the instrument can answer the question; the first thing
it should be pointed at is whether the gap exists at all on hotpot_qa, which is the
one dataset that was healthy throughout.

## 7. Next steps, in order

1. Regenerate the hotpot_qa/musique/synthetic_qa record caches and re-run row 1 (base
   model, 3 seeds) with the fixed harness. Commit `baseline.json`.
2. Read `S` per (dataset, metric). If `S(em)`/`S(f1)` are still at the floor on
   hotpot_qa, the remaining problem is prompt/answer format, not the mask — stop and
   fix that first.
3. Compute Δ (`python -m ddi.delta`), feed it to `python -m ddi.parity`.
4. Only then: does `ddi_qf` sit below `full` on the base model, and does the gap widen
   with document count on hotpot_qa? That is the existence check for the phenomenon.
5. If yes — the training rows are meaningful, and the parity/preservation pair is the
   number to report. If no — the honest write-up is that at this operating point the
   `ddi_qf` mask is already free, which is itself a publishable result.

## Appendix — changed files

| file | change |
|---|---|
| `ddi/prompt.py` | **new** — single source of prompt truth (R3) |
| `ddi/parity.py` | **new** — parity/preservation + admissibility gate (R6) |
| `ddi/inference.py` | mask threaded through every decode step (R2) |
| `ddi/data/kilt_nq.py` | no title placeholders; coverage check raises (R1) |
| `ddi/data/synthetic.py` | record count + docs-per-record configurable (R7) |
| `ddi/finetune.py` | KL reduction (R5); zero-gradient records filtered (R4); shared prompt |
| `ddi/perplexity.py`, `ddi/run.py` | shared prompt; `min_docs` config key |
| `tests/test_parity.py`, `tests/test_prompt.py` | **new** — 30 tests |
| `tests/test_data_kilt_nq.py`, `tests/test_finetune.py`, `tests/test_perplexity.py` | updated for the new behaviour |
