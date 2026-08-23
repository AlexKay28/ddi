"""Every publishable VIBEQUEUE-2 graph: the gates, the finetunes, the parity and doc sweeps.

35 entry points across three base models. `make nv-list` enumerates them (this file plus
`ddi_diagnose.py` are the only two modules that declare any); `make nv-check GRAPH=<name>`
validates one without publishing it.

Only two shapes exist, and `build_ddi_pipeline` picks between them on one argument:

* **eval-only** — pass `eval_models`, get one 1-GPU eval per listed model and one Tracker
  comment. No training, so nothing here costs an 8-GPU job. If a graph built from one of
  these ever shows an 8-GPU block, `eval_models` was not passed and the builder fell back
  to the finetune path.
* **finetune** — pass `seeds` + a reserved `ft_suffix`, get one 8-GPU finetune plus a 1-GPU
  eval of exactly the checkpoint it wrote. Eight of the 35 are this shape; they are the
  entire GPU cost of the file.

| entry point(s)                          | model | shape     | what it varies                     |
|-----------------------------------------|-------|-----------|------------------------------------|
| `ddi_row19_rebaseline_seed{0,1,2}`      | 4B    | eval-only | seed -- fixes Δ, the noise floor   |
| `ddi_row20_gap_{2,5,10}docs`            | 4B    | eval-only | hotpot paragraphs: gap present?    |
| `ddi_row2_all_datasets`                 | 4B    | finetune  | -> `__all4`, the reference run     |
| `ddi_row{8,9,10}_*`                     | 4B    | finetune  | the *training* `datasets=` list    |
| `ddi_all4_seed{0,1,2}`                  | 4B    | finetune  | seed -> `__all4_s{0,1,2}`, tuned Δ |
| `ddi_tuned_gap_{2,5,10}docs`            | 4B    | eval-only | hotpot paragraphs, 500 records     |
| `ddi_parity_eval_500`                   | 4B    | eval-only | base+tuned paired, 500 records     |
| `ddi_parity_eval_500_all4_s0`           | 4B    | eval-only | the same, on `__all4_s0`           |
| `ddi_probe_no_dsmt`                     | 4B    | eval-only | drops DSMT_* -- infra probe        |
| `ddi_qwen25_7b_rebaseline_seed{0,1,2}`  | 7B    | eval-only | row 22: row 19's shape on the 7B   |
| `ddi_qwen25_7b_gap_{2,5,10}docs`        | 7B    | eval-only | row 23: row 20's shape on the 7B   |
| `ddi_qwen25_7b_finetune_all4`           | 7B    | finetune  | row 2's shape -> `__all4`          |
| `ddi_parity_eval_500_qwen25_7b`         | 7B    | eval-only | the parity pair on the 7B          |
| `ddi_qwen36_27b_rebaseline_seed{0,1,2}` | 27B   | eval-only | row 24: row 19's shape on the 27B  |
| `ddi_qwen36_27b_gap_{2,5,10}docs`       | 27B   | eval-only | row 25: row 20's shape on the 27B  |
| `ddi_qwen36_27b_finetune_all4_frozen`   | 27B   | finetune  | -> `__all4_frozen8`, FROZEN TRUNK  |
| `ddi_parity_eval_500_qwen36_27b`        | 27B   | eval-only | the parity pair on the 27B         |

Rows 19 and 20 come first in the ticket's critical path and are the cheapest things in it —
1-GPU evals of the *untrained* base model, no training at all. Row 19 fixes Δ, the noise
floor without which no parity claim is admissible; row 20 asks whether the `ddi_qf` -
`full` gap this ticket exists to close is even present at this operating point. Both are
gates: a bad answer from either stops the grid rather than feeding it. Rows 22-25 are those
same two gates on the other two base models, which is what turns a single point into a
trend.

Rows 8-10 differ from each other in exactly two coupled values: the train `datasets=` list
and the checkpoint name. They live in one dict (`_MIX`) for that reason — letting them be
edited independently is how a row trains one mix and evaluates another row's weights. (Row
11, the single-hop complement of row 10, was dropped on 2026-08-07: it trained on kilt_nq,
whose loader now raises, so it could only ever have failed partway through.)

**Checkpoint names are reserved, not merely chosen.** The name IS the checkpoint directory,
so a repeat silently overwrites another row's weights. Every destination this file writes
is claimed at import through `reserve_checkpoint` at the top, and `build_ddi_pipeline`
refuses to wire a finetune whose destination was never claimed or that targets one of the
protected pre-review checkpoints. That check used to be a hand-maintained table in this
file, which three graphs in sibling modules bypassed entirely; see the guard block in
`ddi_pipeline.py`. See "Models" in the ticket for the authoritative names.

Rows 3, 4 and 5 were deleted in VIBEQUEUE-16 — see the note where they used to sit, above
rows 8-10.

Publish and start one of these::

    cd nirvana && ya make -r
    make nv-pin                                    # what the next publish would ship
    make nv-check   GRAPH=ddi_parity_eval_500      # validates, publishes NOTHING
    make nv-publish GRAPH=ddi_parity_eval_500      # the real thing; prints the instance URL

Deliberately not the `./bin/bin publish -n <fqn>` form this docstring used to show: `-n` is
`--dry-run`, not `--name`, so that line validates and publishes nothing. `Makefile.nirvana`
calls the trap out in its own header.

`publish` pins the new instance to this file's DEFAULT `arc_reference`, not to your working
copy — check the instance's global before starting it.
"""

import vh3

from junk.alexkay28.ddi.nirvana.common import constant
from junk.alexkay28.ddi.nirvana.graphs.ddi_pipeline import (
    build_ddi_pipeline,
    reserve_checkpoint,
)

# --- every checkpoint this file writes, declared once ---------------------------------
#
# `reserve_checkpoint` runs HERE, at import, and returns the suffix it just claimed -- so
# each name below is written exactly once and the declaration cannot drift from the value
# the row trains into. `build_ddi_pipeline` refuses to wire a finetune whose destination
# was not reserved, which is what turns "a new training row forgot to register itself"
# from a silent overwrite into a failed `nv-check`. See the guard block in `ddi_pipeline.py`
# for why the check lives in two halves.
#
# The base models are declared before the suffixes because a destination is `<model><suffix>`
# and the reservation needs both; the rows that use the 7B and 27B are declared much further
# down, but a reservation is not a row.

# The Models table's names, resolved to what `yt-fetch` takes. `ddi_qf-all4` keeps the
# legacy `__seed0` suffix because the run that produced it was already in flight when the
# table was written; the YT node is the `__`-escaped form of the id below.
_BASE = constant.BASE_MODEL  # `qwen35-4b-base`
_ALL4 = constant.ft_model(0)  # `ddi_qf-all4`, the PRE-review weights (`__ddi_qf__seed0`)

# The other two base models in the grid.
_QWEN25_7B = "Qwen/Qwen2.5-7B-Instruct"
_QWEN36_27B = "Qwen/Qwen3.6-27B-Instruct"

# Which numbered grid row each base model's two gates ARE.
#
# `_rebaseline_row` and `_gap_row` used to spell 19 and 20 into their title and `row:` line
# as literals, and the six 7B and six 27B graphs rendered those verbatim -- so twelve
# comments on one ticket announced themselves as rows 19/20 while this file's own section
# headers called the same graphs 22-25. The row number is the first thing a reader keys on
# to find what ran, and they did not launch it, so that is a falsified record, not a typo.
#
# Looked up from the model, not passed in by the entry point: the number is a property of
# (which model, which gate) and of nothing else, so an entry point able to pass one would
# be the same bug with an extra place to make it.
#
# 22/23 for the 7B and 24/25 for the 27B are the ticket's numbering, not invented here --
# see the `91f63399f1` paragraph in `nirvana/common/constant.py`.
_GATE_ROWS = {
    # base model: (re-baseline row, "does the gap exist?" row)
    _BASE: (19, 20),
    _QWEN25_7B: (22, 23),
    _QWEN36_27B: (24, 25),
}

# Row 2's rerun writes here instead of over `__ddi_qf__seed0`. The ticket's Models table
# says so ("rename to `__all4` on the rerun"), and the reason is not tidiness: the finetune
# script does `rm -rf "$DEST_CKPT"` before training, so pointing the rerun at the legacy
# name would destroy the only existing reference weights the moment it started — before it
# had produced anything to replace them, and irrecoverably if it then failed. Under a new
# name the old checkpoint survives as a fallback and the two stay tellable apart in the
# store, which is the whole point of R9. (`__ddi_qf__seed0` is protected outright now: the
# builder refuses to wire a finetune into it whatever a row asks for.)
#
# The 7B's row-2 shape reuses this suffix deliberately -- the suffix names what the row
# *varies* (the training mix), and that row varies only the base model, which the derived
# path already carries. Two reservations rather than one call because they are two rows and
# the owner label is what the collision message prints.
_ALL4_SUFFIX = reserve_checkpoint("__ddi_qf__all4", _BASE, owner="row 2")
reserve_checkpoint(_ALL4_SUFFIX, _QWEN25_7B, owner="row 2 / Qwen2.5-7B")
_ALL4_RERUN = constant.BASE_MODEL + _ALL4_SUFFIX

# The 27B trains under a frozen trunk, so its checkpoint does NOT reuse `__ddi_qf__all4`.
# The suffix names what the row varies, and this row varies the recipe, not just the base
# model: only the top 8 of 64 decoder layers are updated. Calling it `all4` would put a
# partially-tuned checkpoint under a name every other row reads as "full finetune, all four
# datasets", which is the confound the naming convention exists to prevent.
_ALL4_FROZEN_LAYERS = 8
_ALL4_FROZEN_SUFFIX = reserve_checkpoint(
    "__ddi_qf__all4_frozen%d" % _ALL4_FROZEN_LAYERS,
    _QWEN36_27B,
    owner="row 2 / Qwen3.6-27B",
)

# The three seeds of row 2's configuration on the clipped trainer, which is where the
# TUNED side's Δ comes from. Defined here, once, and indexed by seed further down by
# `_all4_seed_row` -- it used to be built twice, ~450 lines apart, once inside the guard's
# table and once for the rows themselves. They agreed, and if they ever stopped agreeing
# the guard would silently stop covering the three most expensive rows in the file. See
# "making the parity claim measurable" below for why these rows exist at all.
_ALL4_SEED_SUFFIXES = tuple(
    reserve_checkpoint(
        "__ddi_qf__all4_s%d" % s, _BASE, owner="row 2 seed %d (tuned Δ)" % s
    )
    for s in (0, 1, 2)
)

# Every row states these back, because the comments land next to a dozen others on one
# ticket and "which eval was this" is not recoverable from the numbers.
_FIXED = constant.FIXED_NOTE_LINES


# --- row 19: the re-baseline that unblocks the grid -----------------------------------
#
# Three eval-only cells on the *untrained* base model, differing in nothing but `seed`.
# Seed is not merely an RNG seed here: manifests are keyed `{name}__seed{seed}`
# (`ddi/data/manifest.py`), so it selects the train/val/test partition. Three seeds on an
# untrained model therefore measure split-to-split variance — which is exactly the noise
# floor Δ is defined as, and the reason this costs three 1-GPU evals and no training.
#
# Δ is then `python -m ddi.delta_cli`, which reads the `full` cells of these three runs and
# refuses below three seeds. Until it has them, every parity number in the grid is
# inadmissible by the ticket's own gate, which is why row 19 blocks everything.
#
# One graph per seed rather than one graph with three cells: `eval_sets` is shared by
# every cell of a `build_ddi_pipeline` call, so a single graph cannot vary `seed` across
# them. Three graphs also means a seed can be re-run alone without repaying for the other
# two.
_REBASELINE_SEEDS = constant.SEEDS

# Every note that lists the eval datasets renders this, so the list is written once.
# Four notes previously carried their own hardcoded copy naming kilt_nq, and all four went
# stale the day it was dropped from the YAML -- the reports then described twelve cells
# while nine ran. `constant.EVAL_DATASETS` is checked against the YAML by a test.
_DATASETS_NOTE = constant.DATASETS_NOTE


def _superseded_by_r13(model: str, unit: str) -> tuple[str, ...]:
    """Notes marking a Qwen3.x cell's pre-671bb3499d numbers void, or nothing at all.

    Only the Qwen3.x checkpoints ship a reasoning chat template, so only they were hit by
    R13: `ddi/prompt.py` did not pass `enable_thinking=False`, the generation prompt opened
    a `<think>` block, and with `max_new_tokens` at 64 the whole budget went to reasoning.
    Every generation metric on those runs describes a model that never emitted a bare span.

    Gated on the model rather than stamped on every graph, because saying this on the
    Qwen2.5-7B runs would be false -- that sweep has no thinking template, was never
    affected, and is the only clean generation data in the grid. Marking it void would
    throw away the one result the ticket can currently stand on.
    """
    if "Qwen3" not in model:
        return ()
    return (
        "",
        "Supersedes the 2026-08-06 run of this %s, which is void: the prompt builder did" % unit,
        "not pass `enable_thinking=False`, so this model opened a `<think>` block and spent",
        "all 64 generation tokens reasoning. It scored em 0.0000 in every cell, and any Δ",
        "or parity claim derived from it describes a model that never answered.",
        "R13, fixed in 671bb3499d -- which is the revision this run is pinned to.",
    )


def _rebaseline_row(seed: int, *, arc_reference, path, arc_token, ticket, model: str = _BASE):
    """One 1-GPU eval of an untrained model on a single seed's split."""
    # 19 on the 4B, 22 on the 7B, 24 on the 27B -- read once and rendered into both the
    # title and the `row:` line, so no graph can name two different rows.
    row = _GATE_ROWS[model][0]
    build_ddi_pipeline(
        arc_reference=arc_reference,
        path=path,
        arc_token=arc_token,
        ticket=ticket,
        eval_models=(model,),
        # `seed` is a top-level key of quality.yaml, so it rides the same `--set` path as
        # every other override; `ddi.run` reads it as `cfg["seed"]` and it lands in each
        # row's provenance, which is what `ddi.delta` groups on.
        eval_sets=("seed=%d" % seed,),
        title="-= VIBEQUEUE-2 row %d: harness re-baseline, %s, seed %d =-" % (row, model, seed),
        notes=(
            "row:       %d (harness re-baseline)" % row,
            "model:     `%s` (untrained)" % model,
            "seed:      %d of %s" % (seed, ", ".join(str(s) for s in _REBASELINE_SEEDS)),
            "eval size: max_records=100 (quality.yaml as committed)",
            _DATASETS_NOTE,
            *_FIXED,
            "override:  --set seed=%d" % seed,
            "",
            "First numbers on the fixed harness: the decode mask no longer leaks documents",
            "into `blind` (R2), the prompt names the passages and asks for a bare span (R3),",
            "and kilt_nq raises instead of fabricating title-only documents (R1). Numbers",
            "from before 2026-08-06 are not comparable to these.",
            *_superseded_by_r13(model, "row"),
            "",
            "Run all three seeds, then compute the margin (one line, no continuations --",
            "graph notes cannot carry a backslash):",
            # `make delta` (Makefile.exp), not `python -m ddi.delta`. `ddi.delta` is the
            # library half -- no `main()`, no `__main__` guard -- so `python -m ddi.delta`
            # imported it, ran nothing and exited 0. A reader takes that for "no output,
            # nothing to report" rather than "wrong module", which is the worst way for a
            # published recipe to fail. The target wraps `ddi.delta_cli` and defaults
            # --store to results/runs.jsonl, so this still fits on one line.
            #
            # `% model`, not `% _BASE`: the 7B and 27B rows render this same note, and the
            # hardcoded 4B told their readers to borrow another model's noise floor --
            # which this file argues at length is assuming the answer.
            "  make delta MODEL=%s DATASET=hotpot_qa METRIC=f1" % model,
            "Δ comes from the `full` cells' seed-to-seed spread alone (ground rule 9), and",
            "no parity claim anywhere in this grid is admissible until it exists.",
            "",
            "Gate: if `blind` is not clearly worst, the harness is still wrong -- stop. If",
            "`full` - `blind` is at the floor on hotpot_qa, the remaining problem is prompt",
            "or answer format, not the mask.",
        ),
    )


@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_row19_rebaseline_seed0(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2 row 19: re-baseline the base model on seed 0."""
    _rebaseline_row(0, arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket)


@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_row19_rebaseline_seed1(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2 row 19: re-baseline the base model on seed 1."""
    _rebaseline_row(1, arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket)


@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_row19_rebaseline_seed2(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2 row 19: re-baseline the base model on seed 2."""
    _rebaseline_row(2, arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket)


# --- row 20: does the gap this whole ticket is about actually exist? ------------------
#
# The cheapest decisive row in the grid. Everything downstream assumes `ddi_qf` sits
# measurably below `full` on the *base* model and that the distance grows with document
# count. If it does not, there is nothing for a finetune to close and the 8-GPU branch is
# vacuous — so this runs before any of it, on the base model only, eval-only, 1 GPU.
#
# hotpot_qa rather than kilt_nq: KILT provenance is thin and (before R1) was titles, so
# `DDI_KILT_N_PASSAGES` could not manufacture documents that were never there (R10).
# hotpot carries 10 real paragraphs on 3668 of 3702 val records, so 2 / 5 / 10 is a real
# sweep on real text.
#
# `DDI_HOTPOT_N_PARAGRAPHS` keeps every supporting paragraph and drops distractors first,
# so lowering the count does not quietly make records unanswerable — that would read as a
# document-count effect while actually being an answerability effect.
_GAP_DOC_COUNTS = (2, 5, 10)


def _gap_row(n_docs: int, *, arc_reference, path, arc_token, ticket, model: str = _BASE):
    """One 1-GPU untrained-model eval on hotpot_qa at a fixed paragraph count."""
    superseded = _superseded_by_r13(model, "cell")
    # 20 on the 4B, 23 on the 7B, 25 on the 27B. See `_GATE_ROWS`.
    row = _GATE_ROWS[model][1]
    build_ddi_pipeline(
        arc_reference=arc_reference,
        path=path,
        arc_token=arc_token,
        ticket=ticket,
        eval_models=(model,),
        eval_sets=("datasets=[hotpot_qa]",),
        eval_env={"DDI_HOTPOT_N_PARAGRAPHS": str(n_docs)},
        title="-= VIBEQUEUE-2 row %d: does the gap exist? %s, %d documents =-"
        % (row, model, n_docs),
        notes=(
            "row:       %d (does the gap exist?)" % row,
            "model:     `%s` (untrained)" % model,
            "documents: %d paragraphs per record (of %s)"
            % (n_docs, "/".join(str(n) for n in _GAP_DOC_COUNTS)),
            "eval size: max_records=100 (quality.yaml as committed)",
            "datasets:  hotpot_qa only",
            *_FIXED,
            "env:       DDI_HOTPOT_N_PARAGRAPHS=%d (default: all 10)" % n_docs,
            "override:  --set datasets=[hotpot_qa]",
            "",
            "The question: on the *base* model, is `ddi_qf` measurably below `full`, and",
            "does the distance widen from 2 -> 5 -> 10 documents? That widening is the core",
            "DDI claim -- isolating documents from each other should cost more as there are",
            "more of them to isolate.",
            "",
            "Gate: if the gap is inside Δ at every document count, the phenomenon does not",
            "exist at this operating point. Write that up and stop -- do not spend 8-GPU",
            "jobs closing a gap that is already closed. On the superseded numbers, closure",
            "on `loss` was already 98.5-100.1 %, which is what makes this the first thing",
            "to check rather than an afterthought.",
            "",
            "Read the three counts together; a single count decides nothing.",
            *superseded,
        ),
    )


@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_row20_gap_2docs(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2 row 20: base-model gap on hotpot_qa at 2 documents."""
    _gap_row(2, arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket)


@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_row20_gap_5docs(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2 row 20: base-model gap on hotpot_qa at 5 documents."""
    _gap_row(5, arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket)


@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_row20_gap_10docs(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2 row 20: base-model gap on hotpot_qa at 10 documents."""
    _gap_row(10, arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket)


# --- row 2's finetune, written once for the rows that run it --------------------------
#
# `ddi_row2_all_datasets`, its 7B and 27B counterparts and the three-seed Δ set are one
# call shape — train on every dataset at one seed into one reserved checkpoint, then eval
# what came out — plus a note header that was written out four times.
#
# Two of those header lines are genuinely NOT shared, so they are parameters rather than
# something forced into one shape:
#
#   * `seed_in_train` — the single-seed rows put `seed=0` in the `train:` line; the Δ-set
#     rows name their seed in the `row:` line and would otherwise say it twice.
#   * `eval_size` — the Δ-set rows carry no `eval size:` line at all, though they eval at
#     quality.yaml's 100 records like the rest. That looks like an omission, but supplying
#     it changes what three published graphs say, which is not this refactor's call.
def _all4_finetune_row(
    *,
    arc_reference,
    path,
    arc_token,
    ticket,
    suffix: str,
    label: str,
    title: str,
    why: tuple[str, ...],
    model: str = _BASE,
    seed: int = 0,
    train_extra: tuple[str, ...] = (),
    ft_sets: tuple[str, ...] = (),
    ft_env: dict | None = None,
    seed_in_train: bool = True,
    eval_size: bool = True,
):
    """One 8-GPU row-2 finetune + one 1-GPU eval of exactly the checkpoint it wrote.

    * `suffix` — the reserved checkpoint suffix. The destination and the `model:` note
      line are both `model + suffix`, so the note cannot name weights the row does not
      train.
    * `label` — everything after `row:       2 ` on the first note line. The row number
      itself is written here, once.
    * `train_extra` — extra `train:` continuation lines, between the parameter line and
      `min_docs` (the 27B's frozen-trunk settings are the only user).
    """
    build_ddi_pipeline(
        arc_reference=arc_reference,
        path=path,
        arc_token=arc_token,
        ticket=ticket,
        model=model,
        # One seed per checkpoint name, always: `build_ddi_pipeline` refuses `ft_suffix`
        # with more than one seed, because N seeds into one directory keeps only the last.
        seeds=(seed,),
        ft_suffix=suffix,
        ft_sets=ft_sets,
        ft_env=ft_env,
        title=title,
        notes=(
            "row:       2 %s" % label,
            "model:     `%s`" % (model + suffix),
            "train:     %s, %slr=1.0e-5, kl_T=1.0, epochs=1, 8-GPU DDP"
            % (", ".join(constant.EVAL_DATASETS), "seed=%d, " % seed if seed_in_train else ""),
            *train_extra,
            "           min_docs=2 -- single-document records are dropped (R4)",
            *(("eval size: max_records=100 (quality.yaml as committed)",) if eval_size else ()),
            _DATASETS_NOTE,
            *_FIXED,
            "",
            *why,
        ),
    )


# --- row 2: the reference finetune, rerun on the fixed harness -----------------------
#
# One 8-GPU finetune on every train dataset + one 1-GPU eval of what it produced. This is the
# checkpoint every later row is read against, so it is the only training row that has to
# happen whatever the rest of the grid does.
#
# The rerun exists because the previous one predates the review: it trained under the bare
# question prompt (R3), with a length-weighted KL (R5), and — in a four-dataset round-robin
# — roughly a quarter of its steps contributing exactly zero gradient (R4). Those are all
# training-side defects, so no amount of re-evaluating the old weights recovers from them.
#
# `min_docs=2` is now the default rather than an override here, so the zero-gradient records
# are dropped by the config the job already loads.
@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_row2_all_datasets(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2 row 2: retrain the reference finetune on every train dataset."""
    # No ft_sets: the full dataset list is finetune.yaml's own default, and row 2 *is* the
    # default configuration. Rows 8-10 are the ones that restrict the mix.
    _all4_finetune_row(
        arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket,
        suffix=_ALL4_SUFFIX,
        label="(all datasets, rerun)",
        title="-= VIBEQUEUE-2 row 2: all datasets, retrained on the fixed harness =-",
        why=(
            "The reference finetune, retrained. The previous weights (`%s`)" % _ALL4,
            "predate the review and are kept only as a fallback -- they were trained under",
            "the bare-question prompt (R3), a length-weighted KL (R5), and a mix in which",
            "about a quarter of steps produced exactly zero gradient (R4). Those are all",
            "training-side, so re-evaluating the old weights cannot repair them.",
            "",
            "This row writes a NEW checkpoint name rather than overwriting the old one: the",
            "finetune script removes its destination before training, so reusing the legacy",
            "name would have destroyed the only reference weights that exist at the moment",
            "training started.",
            "",
            "Read against row 19's baseline with `python -m ddi.parity`. Parity alone is not",
            "a result -- preservation has to hold too, or the model reached parity by getting",
            "worse at everything.",
        ),
    )


# Rows 3, 4 and 5 used to sit here. All three were deleted in VIBEQUEUE-16.
#
# Rows 3 (`max_records=500`) and 4 (`max_records=null`) were descoped. Their stated purpose
# was the "rows 1/3/4 resolution spread" -- the same checkpoint at 100 / 500 / all records,
# read together as the resolution of the eval. Row 1 has no graph anywhere in this project,
# so the spread has no near end and cannot be measured; two thirds of a three-point spread
# is not a smaller version of the measurement, it is a different one. `ddi_parity_eval_500`
# already scores the same checkpoint at 500 records AND pairs it with the base model in one
# graph, on one split, which is strictly more than row 3 produced -- this file's own notes
# say so, in `_mix_notes` and in the parity row's rationale.
#
# Row 5 (two passages, `DDI_KILT_N_PASSAGES=2`) was impossible, not merely descoped. It ran
# `--set datasets=[kilt_nq]`, and the kilt_nq loader raised by design rather than fabricating
# title-only documents (R1). Crucially it validated the CACHED path too -- `load_kilt_nq`
# buffered a prefix of the cache and ran `_check_passage_coverage` on it -- and `kilt_nq_n2`
# was derived by truncating `kilt_nq_n5`, which holds exactly those pre-review "[title]"
# placeholders. So the cache the row was built for was the thing the loader rejected, and the
# row would have queued two 1-GPU cells and died at its first and only dataset. The dataset
# has since been removed from `ddi/data/` altogether, which settles it: there is no loader
# for this row to call. Its `kilt_nq_n2` cache-construction notes went with it; the cache is
# still on YT if a real passage-body join ever arrives.


# --- rows 8-10: the dataset-mix training branch ---------------------------------------
#
# Each row is `datasets=` on the *training* config plus the checkpoint that override
# produces. The two are one entry here on purpose: they are the only thing that varies,
# and letting them be edited independently is how a row trains one mix and evaluates
# another row's weights.
#
# `checkpoint` is the suffix appended to `constant.BASE_MODEL`, so the directory and the
# YT node are `Qwen__Qwen3.5-4B-Instruct<suffix>` — exactly the names in the ticket's
# Models table. `datasets` is written without a space after the comma because Make splits
# `SETS` on whitespace (`scripts._validate_sets`); YAML parses the flow sequence the same
# either way.
_MIX = {
    8: {
        "checkpoint": "__ddi_qf__synth",
        "datasets": "[synthetic_qa]",
        "title": "synthetic only",
        "why": (
            "Hermetic and the cheapest of the four to train, which is why the ticket runs",
            "it first as a shakedown of the `--set datasets=` training path. Read on",
            "hotpot_qa and musique (held out) -- NOT on synthetic_qa, which is the",
            "trained-on column and is saturated: `full` and `ddi_qf` both score exactly",
            "1.0000 on em/f1/presence, so it cannot move in either direction. It also",
            "yields only 57 train records after the val and held-out splits, which was",
            "enough for one epoch of ~7 optimizer steps and shifted holdout perplexity by",
            "1.3e-5 -- so read a null result here as an underpowered run, not as evidence",
            "that ddi_qf finetuning does not transfer. Row 2 answers that.",
        ),
    },
    9: {
        "checkpoint": "__ddi_qf__hotpot",
        "datasets": "[hotpot_qa]",
        "title": "hotpot only",
        "why": (
            "The transfer row: 3 of the 4 eval datasets are held out. A gain that shows up",
            "only on hotpot_qa is memorisation of that corpus, not a property of the mask.",
        ),
    },
    10: {
        "checkpoint": "__ddi_qf__multihop",
        "datasets": "[hotpot_qa,musique]",
        "title": "multi-hop only",
        "why": (
            "Both training datasets are `hop_type: multi`, and with kilt_nq gone they are",
            "also the only two eval datasets carrying signal -- synthetic_qa is saturated.",
            "Read against row 2 (trains on everything): row 10 asks whether restricting the",
            "mix to multi-hop costs anything, now that its intended complement (row 11,",
            "single-hop) has been dropped as unrunnable.",
        ),
    },
}
# Row 11 ("single-hop only", `__ddi_qf__single`) was dropped on 2026-08-07 rather than left
# defined-but-unlaunchable. It trained on `[kilt_nq, synthetic_qa]`, and kilt_nq's loader
# raises rather than fabricating title-only documents (R1) -- so the row would have consumed
# an 8-GPU slot, trained on one dataset, and then aborted. Its purpose is gone too: it was
# the complement of row 10 over *four* datasets, and with kilt_nq removed the single-hop
# side is synthetic_qa alone, which is saturated (`full` == `ddi_qf` == 1.0000) and cannot
# show a mix effect in either direction.

# Rows 8-10 write weights too, so their destinations are reserved like every other. Derived
# straight from `_MIX` rather than restated, so adding a row to that dict is the whole edit
# -- and a row added to `_MIX` without a reservation could not happen, because there is only
# one place the checkpoint name is written.
for _row, _spec in sorted(_MIX.items()):
    reserve_checkpoint(_spec["checkpoint"], _BASE, owner="row %d" % _row)
del _row, _spec


def _mix_notes(row: int) -> tuple[str, ...]:
    spec = _MIX[row]
    model = constant.BASE_MODEL + spec["checkpoint"]
    return (
        "row:       %d (%s)" % (row, spec["title"]),
        "model:     `%s`" % model,
        "train:     datasets=%s (default: %s)"
        % (spec["datasets"], ", ".join(constant.EVAL_DATASETS)),
        "           seed=0, lr=1.0e-5, kl_T=1.0, epochs=1, full split, 8-GPU DDP",
        "eval size: max_records=100 (quality.yaml as committed, not overridden)",
        _DATASETS_NOTE + "  (every eval cell, always -- only *training* is narrowed)",
        *_FIXED,
        "override:  --set datasets=%s on the *training* config only" % spec["datasets"],
        "",
        *spec["why"],
        "",
        # `_ALL4_RERUN`, not `constant.ft_model(0)`. Row 2 trains `__ddi_qf__all4`;
        # `ft_model(0)` is `__ddi_qf__seed0`, which this file labels the PRE-review weights
        # thirty lines from the top. Pointing rows 8-10 at those made every one of the three
        # 8-GPU training rows post a note telling the reader to compare against a checkpoint
        # trained under the bare-question prompt (R3), a length-weighted KL (R5) and ~25 %
        # zero-gradient steps (R4) -- the exact defects row 2's rerun exists to escape.
        "Compare against row 2 (`%s` = ddi_qf-all4, the rerun) at the same eval size, and"
        % _ALL4_RERUN,
        # The bar used to be "the rows 1/3/4 resolution spread". There is no row 1 graph in
        # this file, so that spread has no 100-record arm and will not be measured; a note
        # that names an unmeasurable bar sets no bar at all. These rows eval at 100 records,
        # so state their own resolution and name the row that actually buys a finer one.
        "read a delta as real only if it clears the resolution of a 100-record eval -- the",
        "binomial 95% CI half-width on a proportion is ~0.08-0.10 there. Anything finer has",
        "to be re-measured at 500 records, which is what `ddi_parity_eval_500` does: one",
        "graph scoring base and tuned 4B on the same split, which cuts that interval ~2.2x.",
    )


def _mix_row(row: int, *, arc_reference, path, arc_token, ticket):
    """One 8-GPU finetune on the row's mix + one 1-GPU eval of the checkpoint it wrote."""
    spec = _MIX[row]
    build_ddi_pipeline(
        arc_reference=arc_reference,
        path=path,
        arc_token=arc_token,
        ticket=ticket,
        # seed=0 is the ticket's fixed value; seeds are not an axis in this grid. It still
        # goes through `--set seed=`, because it also selects the train/val/test partition.
        seeds=(0,),
        ft_suffix=spec["checkpoint"],
        ft_sets=("datasets=%s" % spec["datasets"],),
        # No eval_sets: rows 8-10 vary training only. The eval stays the ticket's fixed
        # one so these four and row 2 are the same measurement on different weights.
        title="-= VIBEQUEUE-2 row %d: %s (train datasets=%s) =-"
        % (row, spec["title"], spec["datasets"]),
        notes=_mix_notes(row),
    )


@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_row8_synth_only(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2 row 8: train on synthetic_qa only, eval the resulting checkpoint."""
    _mix_row(8, arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket)


@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_row9_hotpot_only(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2 row 9: train on hotpot_qa only, eval the resulting checkpoint."""
    _mix_row(9, arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket)


@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_row10_multihop_only(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2 row 10: train on hotpot_qa + musique, eval the resulting checkpoint."""
    _mix_row(10, arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket)


# --- rows 22-23: the same two gates on Qwen2.5-7B -------------------------------------
#
# (The 27B's half of the same pair is rows 24-25, much further down; `_GATE_ROWS` at the
# top is where all four numbers live, and the graphs below take theirs from it.)
#
# Rows 19 and 20 answer "is the harness sane" and "does the gap exist" for one model at one
# size. Neither answer is necessarily a property of DDI rather than of Qwen3.5-4B: a 4B
# model that barely uses its documents would show a small `full` - `blind` headroom no
# matter how good the mask is, and a closed gap could just as easily mean the model was
# never reading the passages. Two more sizes turn a single point into a trend.
#
# Both are already mirrored under YT_MODELS_ROOT (verified 2026-08-06), so `yt-fetch`
# resolves them by basename exactly as it does the 4B. They are untrained base models, so
# these are eval-only 1-GPU jobs and nothing here trains anything.
#
# One seed only, not three. Delta is a per-model constant and row 19 pays three evals for
# it on the reference model; repeating that on every size would triple the cost of a
# question these rows do not ask. What they ask is whether the *ordering* holds
# (`full` > `ddi_qf` > `blind`) and whether the gap widens with document count -- both
# read within a model, so they do not need this model's own noise floor to be measured
# first. If one of them produces a gap worth trusting, run its other two seeds then.
@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_qwen25_7b_rebaseline_seed0(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2: re-baseline Qwen2.5-7B-Instruct on seed 0 (row 19 shape, one seed)."""
    _rebaseline_row(
        0, arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket,
        model=_QWEN25_7B,
    )


# Seeds 1 and 2 exist so the 7B gets a Δ of its own, for the same reason the 27B got its
# pair: `ddi.delta` refuses below three seeds by design, because a single seed has no
# spread to measure a noise floor from. Without these two, the 7B is the one model in the
# Phase 1 size trend whose gaps cannot be judged against its own seed-to-seed variance.
#
# The 4B's Δ is NOT a substitute. Δ is a per-model quantity measured from that model's own
# `full` cells; borrowing another model's would be assuming the answer.
@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_qwen25_7b_rebaseline_seed1(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2: re-baseline Qwen2.5-7B-Instruct on seed 1 (row 19 shape)."""
    _rebaseline_row(
        1, arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket,
        model=_QWEN25_7B,
    )


@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_qwen25_7b_rebaseline_seed2(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2: re-baseline Qwen2.5-7B-Instruct on seed 2 (row 19 shape)."""
    _rebaseline_row(
        2, arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket,
        model=_QWEN25_7B,
    )


@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_qwen25_7b_gap_2docs(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2: Qwen2.5-7B-Instruct gap on hotpot_qa at 2 documents (row 20 shape)."""
    _gap_row(
        2, arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket,
        model=_QWEN25_7B,
    )

@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_qwen25_7b_gap_5docs(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2: Qwen2.5-7B-Instruct gap on hotpot_qa at 5 documents (row 20 shape)."""
    _gap_row(
        5, arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket,
        model=_QWEN25_7B,
    )

@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_qwen25_7b_gap_10docs(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2: Qwen2.5-7B-Instruct gap on hotpot_qa at 10 documents (row 20 shape)."""
    _gap_row(
        10, arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket,
        model=_QWEN25_7B,
    )


# --- row 2's shape on Qwen2.5-7B ------------------------------------------------------
#
# The reference finetune, retrained on the next size up: one 8-GPU finetune on every train
# dataset + one 1-GPU eval of what it produced. Nothing here is new machinery --
# `build_ddi_pipeline` has taken `model=` since it was written, and `finetune_script` builds
# `DEST_CKPT` from that same value -- so this is row 2 with one argument changed.
#
# The checkpoint is `Qwen__Qwen2.5-7B-Instruct__ddi_qf__all4`. It reuses row 2's suffix
# deliberately: the suffix names what the row *varies* (the training mix), and this row
# varies nothing but the base model, which the path already carries. The `reserve_checkpoint`
# call at the top of this file claims it under this model, which is what lets the builder
# wire the cell at all.
#
# Read against rows 22/23, not against row 19 -- Δ is a per-model constant and row 19
# measures the 4B's. Row 22 is a single seed, so the 7B has no measured noise floor yet: a
# parity claim on this checkpoint needs rows 22's other two seeds first. What it can support
# without them is the *ordering* and the direction of the finetune's effect, read within the
# model, which is what rows 22/23 were run for.
#
# Memory: teacher + student are both full bf16 replicas per rank, plus bf16 grads and
# AdamW8bit states -- about 8 bytes per parameter, so ~61 GB of the 141 GB card for a 7.6B
# model. Comfortable. DDP shards data, never parameters, so this is a per-rank figure and
# adding GPUs does not reduce it.
@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_qwen25_7b_finetune_all4(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2: row 2's finetune on Qwen2.5-7B-Instruct, all train datasets."""
    _all4_finetune_row(
        arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket,
        model=_QWEN25_7B,
        suffix=_ALL4_SUFFIX,
        label="(all datasets), on the 7B base model",
        title="-= VIBEQUEUE-2 row 2 shape: Qwen2.5-7B, all datasets =-",
        why=(
            "Row 2 with the base model changed and nothing else, so the finetune is",
            "comparable to the 4B's cell by construction rather than by argument.",
            "",
            # The four row numbers come out of `_GATE_ROWS`, not out of this string. They
            # are the 7B's two gates and the 4B's re-baseline, and this note used to be
            # the only place in the file that spelled 22/23 -- while the graphs those
            # numbers refer to were still titling themselves rows 19/20.
            "Read against rows %d/%d, NOT against row %d: row %d measures the 4B's Δ, and Δ"
            % (_GATE_ROWS[_QWEN25_7B] + (_GATE_ROWS[_BASE][0],) * 2),
            "is per-model. Row %d is one seed, so this model has no measured noise floor --"
            % _GATE_ROWS[_QWEN25_7B][0],
            "the ordering (`full` > `ddi_qf` > `blind`) and the direction of the finetune's",
            "effect are readable now; a parity claim needs row %d's other two seeds first."
            % _GATE_ROWS[_QWEN25_7B][0],
            "",
            "Parity alone is not a result -- preservation has to hold too, or the model",
            "reached parity by getting worse at everything.",
        ),
    )


# --- does closure survive more documents? ----------------------------------------------
#
# The decisive experiment for Phase 2's central hypothesis, and it is decisive precisely
# because it is WITHIN one dataset.
#
# Phase 2 measured how much of the mask's cost training removes: 93.2 % on hotpot_qa
# (10 documents per record) but only 62.1 % on musique (20). The base mask cost is flat
# between those two -- 0.5471 vs 0.5436 of headroom -- so the cost has plateaued while
# trainability has not. That is the sharpest result in the phase, and it rests on TWO points
# in which document count is perfectly confounded with dataset identity, domain and hop
# structure (musique is 2-to-4-hop, hotpot 2-hop). "musique is simply harder" explains the
# same numbers.
#
# Varying `DDI_HOTPOT_N_PARAGRAPHS` inside hotpot_qa holds all of that constant and leaves
# document count as the only moving part. Phase 1 already ran the base model at 2/5/10, so
# the axis exists -- what is missing is the tuned model on the same axis.
#
# Both models are scored in ONE graph per document count, at 500 records. Same graph means
# the same split, which is what makes "base cost vs tuned residual at this count" a paired
# comparison rather than two measurements that happen to share a label. 500 rather than
# quality.yaml's 100 because the tuned residual is the small quantity here: at n=100 the
# 95 % interval is +-0.11-0.14, which is wider than the effect being measured.
_TUNED_SWEEP_DOCS = (2, 5, 10)


def _tuned_gap_row(n_docs: int, *, arc_reference, path, arc_token, ticket):
    """Base AND tuned 4B on hotpot_qa at a fixed paragraph count, 500 records."""
    build_ddi_pipeline(
        arc_reference=arc_reference,
        path=path,
        arc_token=arc_token,
        ticket=ticket,
        eval_models=(_BASE, _ALL4_RERUN),
        eval_sets=("datasets=[hotpot_qa]", "max_records=500"),
        eval_env={"DDI_HOTPOT_N_PARAGRAPHS": str(n_docs)},
        title="-= VIBEQUEUE-2: closure vs documents, %d docs, base vs tuned 4B =-" % n_docs,
        notes=(
            "question:  does the finetune's closure survive more documents?",
            "models:    `%s` (base)" % _BASE,
            "           `%s` (row 2 finetune)" % _ALL4_RERUN,
            "documents: %d paragraphs per record (of %s)"
            % (n_docs, "/".join(str(n) for n in _TUNED_SWEEP_DOCS)),
            "eval size: max_records=500",
            "datasets:  hotpot_qa only",
            *_FIXED,
            "env:       DDI_HOTPOT_N_PARAGRAPHS=%d (default: all 10)" % n_docs,
            "override:  --set datasets=[hotpot_qa] max_records=500",
            "",
            "Read the three counts together; a single count decides nothing. The quantity",
            "of interest is not the gap but the FRACTION of the base gap that survives",
            "training, at each document count.",
            "",
            "Why this row exists: across datasets, training removes 93.2 % of the mask's",
            "cost at 10 documents (hotpot_qa) and 62.1 % at 20 (musique), while the base",
            "cost itself is flat between them. But document count is confounded there with",
            "dataset, domain and hop structure. Varying documents INSIDE hotpot_qa holds",
            "those constant, so this is the only version of the measurement that can",
            "attribute the effect to document count at all.",
            "",
            "Both models are scored in one graph so the comparison is paired on one split.",
        ),
    )


@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_tuned_gap_2docs(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2: base vs tuned 4B on hotpot_qa at 2 documents, 500 records."""
    _tuned_gap_row(2, arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket)


@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_tuned_gap_5docs(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2: base vs tuned 4B on hotpot_qa at 5 documents, 500 records."""
    _tuned_gap_row(5, arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket)


@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_tuned_gap_10docs(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2: base vs tuned 4B on hotpot_qa at 10 documents, 500 records."""
    _tuned_gap_row(10, arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket)


# --- making the parity claim measurable -----------------------------------------------
#
# Row 2 on the 4B closed the mask gap: `ddi_qf` f1 on hotpot_qa went 0.3993 -> 0.7888 and
# `full` did not degrade. The direction is far outside any noise. The *verdict* is not:
#
#   * the eval is `max_records=100`, where the binomial 95% CI half-width is about 0.08-0.10
#     on a proportion. The residual parity gap is 0.01 absolute em, and the `blind` rise that
#     trips the leakage gate is 0.05 absolute -- both inside that interval. At this eval size
#     "the gap is small" is measurable and "the gap is zero" is not.
#   * the tuned model has ONE seed, so Δ exists only for the base model. A parity claim needs
#     a noise floor on both sides.
#
# These two rows fix exactly those two holes, and nothing else about the experiment changes.
#
# (`_ALL4_SEED_SUFFIXES`, the three checkpoint names the Δ set writes, is declared and
# reserved at the top of this file with every other checkpoint name. It used to be built a
# second time here.)


def _parity_500_row(
    *,
    arc_reference,
    path,
    arc_token,
    ticket,
    base: str,
    tuned: str,
    size: str,
    eval_size: str,
    why: tuple[str, ...],
    recipe: str = "",
    tuned_note: str = "",
    caveat: tuple[str, ...] = (),
):
    """One base-vs-tuned column pair at max_records=500, for one base model.

    Four rows are this, and everything mechanical about them is identical: the two-model
    `eval_models`, the single override, and the `models:` / `eval size:` / datasets /
    fixed-settings / `override:` header. Written here once.

    The prose is NOT shared, and `why` is passed verbatim per row because the four argue
    four different things -- the CI arithmetic that motivates 500 records at all, which
    *trainer* a number and its error bar come from, and the 7B's and 27B's eval-size
    comparability plus a caveat about their own checkpoint. Folding those together would
    mean deleting caveats, which is the failure mode this file has already paid for.

    * `size` — "4B" / "7B" / "27B", the title's model-size label.
    * `recipe` — the title's parenthetical qualifier for the TUNED checkpoint, written
      only here. Two of the four score a checkpoint that is not a plain full finetune on
      the current trainer, and the title is where a reader skimming a dozen comments on
      one ticket sees that or does not.
    * `tuned_note` / `caveat` — trailing annotation on the tuned model's line, and any
      `^ ...` lines under it for a qualifier too long to ride on it.
    * `eval_size` — the tail of the `eval size:` line: what 500 records is being compared
      against for this row.
    """
    build_ddi_pipeline(
        arc_reference=arc_reference,
        path=path,
        arc_token=arc_token,
        ticket=ticket,
        # Both models in ONE graph rather than two: P, R, E and L are all differences
        # between these two columns, so scoring them in the same run on the same split is
        # what makes the subtraction meaningful. Two separate graphs would reintroduce the
        # between-run variance the whole exercise is trying to measure.
        eval_models=(base, tuned),
        eval_sets=("max_records=500",),
        title="-= VIBEQUEUE-2: parity at 500 records, base vs tuned %s%s =-"
        % (size, " (%s)" % recipe if recipe else ""),
        notes=(
            "models:    `%s` (base)" % base,
            "           `%s`%s" % (tuned, tuned_note),
            *caveat,
            "eval size: max_records=500 -- %s" % eval_size,
            _DATASETS_NOTE,
            *_FIXED,
            "override:  --set max_records=500 on both columns",
            "",
            *why,
        ),
    )


@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_parity_eval_500(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2: base AND tuned 4B at max_records=500, so parity has resolution."""
    _parity_500_row(
        arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket,
        base=_BASE,
        tuned=_ALL4_RERUN,
        size="4B",
        tuned_note=" (row 2 finetune)",
        eval_size="5x row 2's, which halves the CI",
        why=(
            "Why: at max_records=100 the binomial 95% CI half-width is ~0.08-0.10 on a",
            "proportion. Row 2's residual parity gap is 0.01 absolute em and the `blind`",
            "rise that trips the leakage gate is 0.05 absolute -- both sit inside that",
            "interval, so neither the parity verdict nor the leakage flag would survive a",
            "re-run. 500 records cuts the interval by ~2.2x.",
            "",
            "This does NOT by itself make a parity claim admissible: Δ still comes from the",
            "base model's three seeds only. The tuned side gets its noise floor from the",
            "three-seed rows queued alongside this one.",
        ),
    )


# --- the same pair, on a checkpoint whose Δ was measured on the SAME trainer ------------
#
# `ddi_parity_eval_500` scores `__ddi_qf__all4`, which was trained on `0ca69996b1` -- before
# gradient clipping and the non-finite guard existed. Δ for the tuned side comes from
# `__all4_s0/s1/s2`, which are the FIRST runs with them. Reading one against the other is the
# thing `_all4_seed_row`'s own note warns about: clipping changes the optimisation
# trajectory, so a noise floor built from two different trainers measures the trainer, not
# the seed. The parity number and its error bar have to come from the same code.
#
# `__all4_s0` is the seed-0 member of the Δ set, so this row puts the subject and its noise
# floor on the same trainer -- and at the same 500 records as the 4B's existing parity row,
# so the two are readable against each other and the only thing that moved is the trainer.
#
# Deliberately NOT a replacement for `ddi_parity_eval_500`: that row stays as the
# cross-check. If the two disagree by more than Δ, clipping changed the result and that is
# itself a finding worth having, since every finetune from here on runs with it.
_ALL4_S0 = constant.BASE_MODEL + _ALL4_SEED_SUFFIXES[0]


@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_parity_eval_500_all4_s0(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2: 500-record parity on `__all4_s0`, whose Δ came from the same trainer."""
    _parity_500_row(
        arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket,
        base=_BASE,
        tuned=_ALL4_S0,
        size="4B",
        recipe="clipped trainer",
        caveat=(
            "           ^ seed 0 of the three-seed Δ set, trained WITH gradient clipping",
        ),
        eval_size="the same as the existing 4B parity row",
        why=(
            "Why this and not `ddi_parity_eval_500`: that row scores `__ddi_qf__all4`, trained",
            "before clipping existed, while Δ for the tuned side comes from `__all4_s0/s1/s2`,",
            "the first runs with it. A noise floor from one trainer against a number from",
            "another measures the trainer. This row puts both on the same code, at the same",
            "eval size, with base and tuned in ONE graph so they share a split.",
            "",
            "The older row is kept as a cross-check, not replaced. A disagreement between the",
            "two larger than Δ means clipping moved the result, which matters on its own -- ",
            "every finetune from here on runs with it.",
        ),
    )


# --- the same 500-record base-vs-tuned column pair on the other two models --------------
#
# `ddi_parity_eval_500` gave the 4B a paired base-vs-tuned measurement at 500 records. The
# 7B and 27B have no equivalent: the 27B's own row scored at quality.yaml's default 100, and
# the 7B had no surviving checkpoint at all until the re-run. A cross-model comparison built
# from those would be reading a 500-record number against a 100-record one and calling the
# difference a size effect, when a large part of it is just the eval size changing.
#
# So these two rows exist to make the three models say the same sentence at the same
# resolution. They and `ddi_parity_eval_500` now go through one builder, `_parity_500_row`,
# and differ only in the model pair and their own prose -- same 500 records, same datasets,
# same fixed settings, both columns in ONE graph so base and tuned land on the same split
# and the subtraction between them is paired rather than two runs that happen to share a
# label. (They used to be three hand-copies of the same call, which is a slower way of
# saying the same thing and a faster way of stopping saying it.)
#
# What they still cannot do is make the three models a size *trend*. The 4B and 7B tuned
# checkpoints are full finetunes; the 27B's is a frozen trunk, top 8 of 64 layers. That is a
# different recipe, not a different scale, and no eval size fixes it -- see the frozen-trunk
# row below for the memory arithmetic that forces it. Every number this row produces has to
# be quoted with that attached, which is why the note says so twice.
@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_parity_eval_500_qwen25_7b(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2: base AND tuned 7B at max_records=500, matching the 4B's parity row."""
    _parity_500_row(
        arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket,
        base=_QWEN25_7B,
        tuned=_QWEN25_7B + _ALL4_SUFFIX,
        size="7B",
        # In the title, not only in the prose. This row and the 27B's are read side by
        # side and their tuned checkpoints are NOT the same recipe; the 27B's title has
        # said `frozen trunk` since it was written, so a reader scanning titles saw one
        # qualified row next to an unqualified one and had no reason to think the
        # difference was in the recipe rather than only in the size.
        recipe="full finetune",
        tuned_note=" (row 2 shape, FULL finetune)",
        eval_size="the same size as the 4B's parity row",
        why=(
            "Why: the 4B has a paired base-vs-tuned measurement at 500 records and this",
            "model did not. Comparing a 500-record number against a 100-record one reads",
            "the change in eval size as if it were a change in the model. This row removes",
            "that by scoring the 7B at the same size, with both columns in ONE graph so",
            "base and tuned share a split.",
            "",
            "The tuned checkpoint is a FULL finetune, so it is directly comparable to the",
            "4B's. The 27B's is not -- that one is a frozen trunk (top 8 of 64 layers) and",
            "its row says so.",
            "",
            "Single seed on the tuned side. Δ for this model comes from the 7B re-baseline",
            "seed rows, not from row 19, which measures the 4B's. Until those land the",
            "readable claims are the ordering and the direction of the effect within this",
            "model -- not parity.",
        ),
    )


@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_parity_eval_500_qwen36_27b(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2: base AND tuned 27B at max_records=500 -- tuned side is a frozen trunk."""
    _parity_500_row(
        arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket,
        base=_QWEN36_27B,
        tuned=_QWEN36_27B + _ALL4_FROZEN_SUFFIX,
        size="27B",
        recipe="frozen trunk",
        caveat=(
            "           ^ FROZEN TRUNK: only the top %d of 64 decoder layers were trained"
            % _ALL4_FROZEN_LAYERS,
        ),
        eval_size="the same size as the 4B's parity row",
        why=(
            "Why: this model's own row scored at quality.yaml's default 100 records. Read",
            "against the 4B's 500-record parity row, the eval size would be moving at the",
            "same time as the model. This row holds it fixed, with both columns in ONE",
            "graph so base and tuned share a split.",
            "",
            "READ THIS ROW NARROWLY -- and this is NOT fixed by the larger eval. The 4B and",
            "7B tuned checkpoints are full finetunes; this one trained %d of 64 layers"
            % _ALL4_FROZEN_LAYERS,
            "because a full 27B finetune needs ~222 GB per rank against a 141 GB card. A",
            "weaker number here is therefore ambiguous between `the mask does not transfer",
            "at 27B` and `%d layers was not enough capacity`, and only the first is about"
            % _ALL4_FROZEN_LAYERS,
            "DDI. This row shows whether the effect survives at this size; it does not put",
            "the 27B on a size trend with the other two, and no chart built from it may",
            "quote the number without the frozen-trunk caveat beside it.",
            "",
            "Single seed on the tuned side, and Δ for this model comes from the 27B",
            "re-baseline seed rows.",
        ),
    )


def _all4_seed_row(seed: int, *, arc_reference, path, arc_token, ticket):
    """One row-2 finetune at `seed`, writing its own checkpoint."""
    _all4_finetune_row(
        arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket,
        suffix=_ALL4_SEED_SUFFIXES[seed],
        seed=seed,
        label="(all datasets), seed %d of 0/1/2" % seed,
        title="-= VIBEQUEUE-2: row 2 finetune, seed %d (for the tuned-side Δ) =-" % seed,
        # The `row:` line above already names the seed; repeating it on the next line is
        # the only reason these rows' `train:` line differs from the other three.
        seed_in_train=False,
        # These three rows carry no `eval size:` line. See `_all4_finetune_row`.
        eval_size=False,
        why=(
            "Purpose: Δ for the TUNED model. Row 2 has one seed, so every parity number on",
            "it is a point estimate with no error bar on the tuned side. Three seeds of the",
            "same configuration give `full`'s seed-to-seed spread after finetuning, which is",
            "what `ddi.delta` turns into Δ (Δ = 2 · std across per-seed means, >= 3 seeds).",
            "",
            "These three are the FIRST finetunes to run with gradient clipping and the",
            "non-finite guards. Row 2's original seed-0 run predates them, so it is kept as",
            "a cross-check and deliberately NOT mixed into this Δ set: clipping changes the",
            "optimisation trajectory, and a noise floor built from two different trainers",
            "measures the trainer, not the seed.",
        ),
    )


@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_all4_seed0(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2: row 2 finetune, seed 0, clipped trainer (tuned-side Δ)."""
    _all4_seed_row(0, arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket)


@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_all4_seed1(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2: row 2 finetune, seed 1, clipped trainer (tuned-side Δ)."""
    _all4_seed_row(1, arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket)


@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_all4_seed2(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2: row 2 finetune, seed 2, clipped trainer (tuned-side Δ)."""
    _all4_seed_row(2, arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket)


# --- infrastructure probe: is DSMT_YT_WRAPPER_LOCAL_RPC load-bearing? -----------------
#
# Not a measurement. A 1-GPU eval whose only difference from `ddi_row19_rebaseline_seed0`
# is that it launches WITHOUT `DSMT_YT_WRAPPER_LOCAL_RPC=1`.
#
# The question: `constant.PYDL_JOB_VARIABLES` was lifted verbatim off the hand-built graph
# this project replaced, and `DSMT_*` configures the DeepSpeed-Megatron YT wrapper. This
# repo does not use DeepSpeed or Megatron -- `ddi/finetune.py` is transformers + torch DDP
# + bitsandbytes, and this is the only DSMT string in the tree. So the variable looks like
# dead inheritance. "Looks like" is the whole problem: the name suggests it configures a
# wrapper shipped inside the PyDL image rather than our code, in which case it is live
# regardless of which trainer we run, and `_PYDL_COMMON` carries an explicit warning that
# removing these is how an 8-GPU run breaks in ways that look like a model problem.
#
# So it is answered on the cheapest cell that exists rather than argued about: 1 GPU, no
# training, no checkpoint written, minutes to run. If this succeeds and its numbers match
# row 19's, the variable is inert on the eval path and can be dropped from the finetune
# path too -- on evidence. If it fails, that failure IS the documentation, and it cost one
# GPU-minute instead of an 8-GPU job.
#
# Deliberately keeps the other two variables. `UNITEDNN_MULTIPROCESSING_TRANSFORM_LD_PRELOAD=`
# unsets an LD_PRELOAD and is plainly not inert; changing more than one thing at a time
# would make a failure uninterpretable.
_JOB_VARS_NO_DSMT = tuple(
    v for v in constant.PYDL_JOB_VARIABLES if not v.startswith("DSMT_")
)


@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_probe_no_dsmt(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """Infra probe: row 19's 1-GPU eval, launched without DSMT_YT_WRAPPER_LOCAL_RPC."""
    build_ddi_pipeline(
        arc_reference=arc_reference,
        path=path,
        arc_token=arc_token,
        ticket=ticket,
        seeds=(0,),
        eval_models=(_BASE,),
        job_variables=_JOB_VARS_NO_DSMT,
        title="-= VIBEQUEUE-2 infra probe: eval without DSMT_YT_WRAPPER_LOCAL_RPC =-",
        notes=(
            "PROBE, NOT A MEASUREMENT -- do not read this into the grid.",
            "",
            "model:     `%s` (untrained)" % _BASE,
            "job vars:  %s" % ", ".join(_JOB_VARS_NO_DSMT),
            "dropped:   DSMT_YT_WRAPPER_LOCAL_RPC=1",
            _DATASETS_NOTE,
            *_FIXED,
            "",
            "Identical to row 19 seed 0 except for the missing variable. This repo runs no",
            "DeepSpeed and no Megatron -- the trainer is transformers + torch DDP -- and that",
            "variable is the only DSMT string in the tree, so it looks like inheritance from",
            "the hand-built graph this project replaced. It is tested rather than assumed",
            "because the name points at a wrapper inside the PyDL image, not at our code.",
            "",
            "Success here, with numbers matching row 19, means the variable is inert on the",
            "eval path and can be dropped from the finetune path on evidence. A failure is",
            "the answer too, and costs one GPU-minute rather than an 8-GPU job.",
        ),
    )


# --- row 2's shape on Qwen3.6-27B, under a frozen trunk -------------------------------
#
# The 27B cannot be full-finetuned by this code on a 141 GB card, and the arithmetic is not
# close enough to be worth attempting. Per rank a full finetune holds teacher (bf16) +
# student (bf16) + gradients (bf16) + AdamW8bit moments -- about 8 bytes per parameter --
# and DDP replicates every one of those terms on every rank, so the 8-GPU job does not
# divide it:
#
#     27.8B params x 8 B/param  =  ~222 GB per rank   vs   141 GB available
#
# Freezing the trunk removes the only optional half. The two model replicas have to be
# resident (111 GB), but gradients and optimizer state are paid only for what is trained:
# 8 of 64 layers is ~2.6B params, so ~5 GB of gradients + ~5 GB of AdamW8bit moments.
#
#     111 GB (teacher+student) + ~10 GB (grads+optimizer)  =  ~122 GB per rank
#
# leaving ~19 GB for activations, which `DDI_GRAD_CHECKPOINT=1` below keeps small.
#
# What this costs, stated plainly so the eventual write-up cannot forget it: this is NOT
# the same recipe as the 4B and 7B cells, which tune every layer. A weaker result here is
# therefore ambiguous between "the mask does not transfer to 27B" and "8 layers was not
# enough capacity", and only the first of those is about DDI. Read this row for whether the
# effect *survives* at 27B, not for a clean size trend against the other two.
@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_qwen36_27b_finetune_all4_frozen(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2: row 2's finetune on Qwen3.6-27B, top-8 layers only (frozen trunk)."""
    _all4_finetune_row(
        arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket,
        model=_QWEN36_27B,
        suffix=_ALL4_FROZEN_SUFFIX,
        ft_sets=("trainable_layers=%d" % _ALL4_FROZEN_LAYERS,),
        ft_env={"DDI_GRAD_CHECKPOINT": "1"},
        label="(all datasets), on the 27B base model, FROZEN TRUNK",
        title="-= VIBEQUEUE-2 row 2 shape: Qwen3.6-27B, all datasets, frozen trunk =-",
        # The two knobs above, restated in the note. This is the only row with any, which
        # is why they are a parameter rather than part of the shared header.
        train_extra=(
            "           trainable_layers=%d -- only the top %d of 64 decoder layers"
            % (_ALL4_FROZEN_LAYERS, _ALL4_FROZEN_LAYERS),
            "           + the final norm are updated; embeddings and lm_head stay frozen",
            "           DDI_GRAD_CHECKPOINT=1",
        ),
        why=(
            "Why the trunk is frozen, and it is not a tuning choice: a full finetune here",
            "needs ~222 GB per rank (teacher + student + grads + AdamW8bit at ~8 B/param on",
            "27.8B parameters). DDP replicates all of that per rank, so the 8-GPU job does",
            "not divide it, and the card has 141 GB. Freezing the trunk pays gradients and",
            "optimizer state only for the ~2.6B trained parameters: ~122 GB per rank.",
            "",
            "READ THIS ROW NARROWLY. The 4B and 7B cells tune every layer; this one does",
            "not. A weaker number here is ambiguous between `the mask does not transfer to",
            "27B` and `8 layers was not enough capacity`, and only the first is about DDI.",
            "It answers whether the effect survives at this size -- not where 27B sits on a",
            "size trend against the other two.",
            "",
            "The checkpoint name says `frozen%d` for the same reason: nothing downstream"
            % _ALL4_FROZEN_LAYERS,
            "should be able to pick these weights up believing they are a full finetune.",
        ),
    )

@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_qwen36_27b_rebaseline_seed0(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2: re-baseline Qwen3.6-27B-Instruct on seed 0 (row 19 shape, one seed)."""
    _rebaseline_row(
        0, arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket,
        model=_QWEN36_27B,
    )


# Seeds 1 and 2 exist so the 27B gets a Δ of its own. Measured 2026-08-07, its gaps on
# hotpot_qa are large (P = 63-65 % of headroom at 5 and 10 documents) -- but seed 0 alone
# cannot make them admissible: `ddi.delta` refuses below three seeds, by design, because a
# single seed has no spread to measure a noise floor from.
#
# The 4B's Δ is NOT a substitute. Δ is a per-model quantity, and this model's `full` cells
# sit higher (em 0.67 vs 0.56) on a different split partition, so its seed-to-seed spread
# has no reason to match. Borrowing one would be assuming the answer.
@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_qwen36_27b_rebaseline_seed1(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2: re-baseline Qwen3.6-27B-Instruct on seed 1 (row 19 shape)."""
    _rebaseline_row(
        1, arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket,
        model=_QWEN36_27B,
    )


@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_qwen36_27b_rebaseline_seed2(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2: re-baseline Qwen3.6-27B-Instruct on seed 2 (row 19 shape)."""
    _rebaseline_row(
        2, arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket,
        model=_QWEN36_27B,
    )


@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_qwen36_27b_gap_2docs(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2: Qwen3.6-27B-Instruct gap on hotpot_qa at 2 documents (row 20 shape)."""
    _gap_row(
        2, arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket,
        model=_QWEN36_27B,
    )

@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_qwen36_27b_gap_5docs(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2: Qwen3.6-27B-Instruct gap on hotpot_qa at 5 documents (row 20 shape)."""
    _gap_row(
        5, arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket,
        model=_QWEN36_27B,
    )

@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_qwen36_27b_gap_10docs(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
    ticket: vh3.String = constant.TICKET,
):
    """VIBEQUEUE-2: Qwen3.6-27B-Instruct gap on hotpot_qa at 10 documents (row 20 shape)."""
    _gap_row(
        10, arc_reference=arc_reference, path=path, arc_token=arc_token, ticket=ticket,
        model=_QWEN36_27B,
    )
