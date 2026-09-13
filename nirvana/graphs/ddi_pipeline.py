"""One builder for every DDI finetune-and-report graph. No graphs are declared here.

Every publishable row in this project is a call to `build_ddi_pipeline`: export, checkout,
merge, one finetune/eval cell per seed (or one eval per model), aggregate, comment. The
graphs themselves live in `ddi_rows.py`; this module is the chain with the differences
lifted into arguments, so a new variant is a call, not a new module.

    # one seed, one eval, a Python "Make DataFrame" report
    build_ddi_pipeline(arc_reference=..., path=..., arc_token=..., seeds=(0,),
                       ft_suffix=reserve_checkpoint("__ddi_qf__mine", model, owner="row N"))

    # eval-only: no finetune at all, one 1-GPU eval per listed model
    build_ddi_pipeline(arc_reference=..., path=..., arc_token=..., eval_models=(base, tuned))

    # another base model, another ticket
    build_ddi_pipeline(arc_reference=..., path=..., arc_token=..., seeds=(0,),
                       model="Qwen/Qwen3.5-7B-Instruct", ticket="VIBEQUEUE-7", ft_suffix=...)

Seed count is the only thing that changes between "single" and "grid": every seed gets its
own finetune + eval cell, and the report block is fed the whole list. The one shape detail
that follows from the operations rather than from taste: `Make DataFrame` accepts a single
connection on `inp`, so when there is more than one payload to report the builder puts a
one-line `return input` Starlark in front of it to collect them (`_collect`). With one
payload the eval is wired straight in and the graph is exactly seven blocks.

There is exactly one report backend. A second, Starlark-based one existed until
VIBEQUEUE-16 along with a cached-baseline eval block; both were unreachable — no caller ever
passed `aggregator=` or `with_baseline=` — and both are gone. CHANGELOG.md (2026-08-08)
keeps the reasoning, in particular why the baseline cache key was not safe to reuse.

**Checkpoint destinations are reserved, not merely named** — see `reserve_checkpoint`
below. The builder refuses to wire a finetune into a destination no module declared, and
refuses outright to write into the protected pre-review checkpoints. That is what makes
"a new training row forgot to register itself" a build failure rather than an 8-GPU job
that silently destroys another row's weights.

Build, dry-run, publish::

    cd nirvana && ya make -r
    make nv-list                              # every publishable entry point
    make nv-check   GRAPH=ddi_row2_all_datasets   # validates, publishes NOTHING
    make nv-publish GRAPH=ddi_row2_all_datasets   # the real thing; prints the instance URL

Not `./bin/bin publish -n <fqn>`, which older docstrings in this tree used to show: `-n`
is `--dry-run`, not `--name`, so that line validates and publishes nothing. See the header
of `Makefile.nirvana`.

`publish` returns a **new instance**, and that instance is pinned to the graph's DEFAULT
`arc_reference` — not to whatever revision you last ran, and not to your working copy. A
stale `constant.ARC_REFERENCE` therefore ships old code to every block of a fresh run
without saying anything. This has bitten twice. Check the instance's global before starting
it, and repin with `set_parameters(scope="global", ...)` if it is not the revision you mean.
"""

import json
import re
from collections.abc import Mapping
from typing import Sequence

import vh3
from vh3.lib.starlark.json_process import json_process
from vh3.models.graph import BlockCacheOptions

from junk.alexkay28.ddi.nirvana.common import constant
from junk.alexkay28.ddi.nirvana.graphs import scripts
from junk.alexkay28.ddi.nirvana.graphs.pydl import (
    _PYDL_COMMON,
    _PYDL_FINETUNE_COMMON,
    _PYDL_JOB,
    _PYDL_JOB_FINETUNE,
    _pydl_block_args,
)
from junk.alexkay28.ddi.nirvana.operations.external import (
    add_comment_to_ticket,
    arc_export,
    merge_tar_archives,
    python_3_deep_learning,
    python_run,
    zeliboba_arc_checkout_deterministic,
)

# The report the "Make DataFrame" block runs: real python3 with pandas available, so
# ordinary Python — no Starlark limitations here.
#
# It walks the payload instead of indexing into it. The eval block's json_output is a list
# of its logs/*.json documents; several evals collected together arrive as a list of those
# lists. Walking means one seed and N seeds render through the same code.
#
# The whole thing is sent as a Nirvana *expression*, so `${...}` is substituted before
# python ever sees it — which is the only way the block can learn the graph's URL and the
# `ticket` global. Keep `${` and `#{` out of the body.
_REPORT_TEMPLATE = r'''
import json

TICKET = 'TICKET_VALUE'
URL = 'WORKFLOW_URL'
METRICS = ("em", "f1", "presence", "loss")


def quality_docs(payload):
    """Every quality-summary document in the payload, whatever list nesting it arrived in."""
    if isinstance(payload, dict):
        if "tables" in payload:
            yield payload
        return
    if isinstance(payload, list):
        for item in payload:
            for doc in quality_docs(item):
                yield doc


def fmt(value):
    if isinstance(value, float):
        return "%.4f" % value
    return str(value)


def render(doc):
    lines = ["### " + str(doc.get("model_filter", "model"))]
    for table in doc.get("tables", []):
        if table.get("kind") != "quality" or table.get("metric") not in METRICS:
            continue
        rows = []
        for _model, masks in sorted(table.get("data", {}).items()):
            for mask, datasets in sorted(masks.items()):
                for dataset, value in sorted(datasets.items()):
                    rows.append("%s | %s | %s" % (mask, dataset, fmt(value)))
        if rows:
            lines.append("")
            lines.append("#### " + str(table.get("metric")))
            lines.append("mask | dataset | value")
            lines.append("--- | --- | ---")
            lines.extend(rows)
    return lines


payload = json.loads(inp.read_text())

lines = ["TITLE_VALUE", "", "graph: " + URL, ""]
EXTRA_SECTION
rendered = 0
for doc in quality_docs(payload):
    lines.extend(render(doc))
    lines.append("")
    rendered += 1

if rendered == 0:
    lines.append("No quality tables in the eval output; the raw payload follows.")
    lines.append("")
    lines.append(json.dumps(payload)[:4000])

message = "\n".join(lines).strip()
out.write_text(json.dumps([{"ticket": TICKET, "message": message}], indent=2))
'''


# Optional additions to the report, spliced in at `EXTRA_SECTION`. Both are ordinary
# top-level python that runs after `payload` and `lines` exist and before the per-model
# sections are appended, so a header note and a comparison land *above* the raw tables
# rather than trailing them.
#
# The substitution is on `"EXTRA_SECTION\n"`, so with nothing to inject the emitted script
# is byte-for-byte the one the already-published instances of the plain (title-less,
# note-less) shape run. That is the whole reason the placeholder sits alone on its line.
_NOTES_SECTION = '''\
for _note in NOTE_LINES:
    lines.append(_note)
lines.append("")
'''

# A `mask | dataset | <model> ...` table per metric, plus the `ddi_qf - full` gap the
# VIBEQUEUE-2 rows are actually read on. Only makes sense with more than one eval.
_COMPARE_SECTION = '''\
COLUMNS = COLUMN_MODELS


def signed(value):
    text = fmt(value)
    if not text.startswith("-"):
        text = "+" + text
    return text


def columns(payload):
    """One quality document per column, in COLUMNS order."""
    docs = list(quality_docs(payload))
    keyed = {}
    for doc in docs:
        keyed[str(doc.get("model_filter", ""))] = doc
    if all(name in keyed for name in COLUMNS):
        return [keyed[name] for name in COLUMNS]
    # `model_filter` is what `ddi.aggregate --model` recorded, so it normally names the
    # column outright. Wiring order carries the same information — the builder connects
    # one eval block per column, in COLUMNS order — so fall back to it rather than losing
    # the comparison because a name did not round-trip.
    return docs[:len(COLUMNS)]


def cells(doc, metric):
    """{mask: {dataset: value}} for one metric; {} when that metric is absent."""
    for table in doc.get("tables", []):
        if table.get("kind") == "quality" and table.get("metric") == metric:
            for _model, masks in sorted(table.get("data", {}).items()):
                return masks
    return {}


def side_by_side(payload):
    docs = columns(payload)
    if len(docs) < 2:
        return []
    out = []
    for metric in METRICS:
        per_column = [cells(doc, metric) for doc in docs]
        keys = []
        for mask in sorted(per_column[0]):
            for dataset in sorted(per_column[0][mask]):
                if all(mask in c and dataset in c[mask] for c in per_column):
                    keys.append((mask, dataset))
        if not keys:
            continue
        out.append("#### " + metric)
        out.append("mask | dataset | " + " | ".join(COLUMNS))
        out.append("--- | --- | " + " | ".join(["---"] * len(COLUMNS)))
        for mask, dataset in keys:
            values = [fmt(c[mask][dataset]) for c in per_column]
            out.append("%s | %s | %s" % (mask, dataset, " | ".join(values)))
        out.append("")
        # The table above shows the ddi_qf - full gap only as two rows to subtract by eye,
        # and that gap is the number every one of these rows is read on. Subtract it here.
        gap_datasets = sorted(
            d
            for d in per_column[0].get("ddi_qf", {})
            if all(
                "ddi_qf" in c and "full" in c and d in c["ddi_qf"] and d in c["full"]
                for c in per_column
            )
        )
        if not gap_datasets:
            continue
        out.append("#### " + metric + ": ddi_qf minus full")
        out.append("dataset | " + " | ".join(COLUMNS))
        out.append("--- | " + " | ".join(["---"] * len(COLUMNS)))
        for dataset in gap_datasets:
            gaps = [signed(c["ddi_qf"][dataset] - c["full"][dataset]) for c in per_column]
            out.append("%s | %s" % (dataset, " | ".join(gaps)))
        out.append("")
    return out


lines.extend(side_by_side(payload))
'''

_DEFAULT_TITLE = "-= DDI eval =-"


def _literal(text: str, what: str) -> str:
    """Reject anything that would not survive being pasted into the report script.

    The script is a Nirvana *expression* embedded in a double-quoted python list, so a
    `"` ends the string early and a `${` is eaten by expression substitution before python
    sees it. Both produce a block that fails at run time, hours downstream of here.
    """
    if '"' in text or "\\" in text or "${" in text or "#{" in text:
        raise ValueError(f'{what} must not contain a quote, a backslash, "${{" or "#{{": {text!r}')
    return text


def _report_code(
    ticket,
    title: str = _DEFAULT_TITLE,
    notes: Sequence[str] = (),
    columns: Sequence[str] = (),
) -> vh3.Expr:
    """The report script, with the ticket and the workflow URL wired in.

    `ticket` is either a plain string or the graph's `ticket` option; a graph option
    becomes `${global["ticket"]}` so the value is read at run time rather than frozen
    into the published graph.

    `title` is the comment's first line — the only thing that tells a reader which row of
    the grid they are looking at. `notes` are header lines under it (models, eval size).
    `columns` are the model ids to put side by side; fewer than two renders no comparison,
    which is why the default path is unaffected.
    """
    ticket_literal = ticket if isinstance(ticket, str) else f"{ticket:expr}"
    extra = ""
    if notes:
        extra += _NOTES_SECTION.replace(
            "NOTE_LINES", json.dumps([_literal(n, "note") for n in notes])
        )
    if len(tuple(columns)) > 1:
        extra += _COMPARE_SECTION.replace(
            "COLUMN_MODELS", json.dumps([_literal(c, "column") for c in columns])
        )
    return vh3.Expr(
        _REPORT_TEMPLATE.replace("EXTRA_SECTION\n", extra)
        .replace("TITLE_VALUE", _literal(title, "title"))
        .replace("TICKET_VALUE", ticket_literal)
        .replace("WORKFLOW_URL", f"{vh3.expr.meta.workflow_url:expr}")
    )


def _collect(payloads):
    """One JSON output carrying every payload.

    `Make DataFrame` takes a single connection on `inp`, so anything past the first payload
    has to be gathered first. One payload — the single-seed graph — needs no block at all.
    """
    if len(payloads) == 1:
        return payloads[0]
    return json_process(
        process="return input",
        input=list(payloads),
        **vh3.block_args(
            name="Collect eval summaries",
            code="merge_evals",
            cache_options=BlockCacheOptions(mr_policy="reuse_if_not_modified_strict"),
        ),
    )


def _block_code(prefix: str, model: str) -> str:
    """A block `code` derived from a model id: unique per model, stable across builds."""
    return prefix + "_" + re.sub(r"[^0-9a-z]+", "_", model.lower()).strip("_")


# --- the checkpoint-collision guard ----------------------------------------------------
#
# This used to be a hand-maintained table in `ddi_rows.py` (`_TRAINING_ROWS`) checked once
# at that module's import. It read that table and nothing else, so it could only see rows
# declared in `ddi_rows.py` -- and three publishable entry points in sibling modules
# (`ddi_qwen35_4b_finetune_eval`, `ddi_qwen35_4b_seed_grid`, `ddi_qwen35_4b_single`) wired
# finetunes straight into `__ddi_qf__seed{0,1,2}` without ever appearing in it. All three
# were deleted in VIBEQUEUE-16; this guard is what makes that class of bug impossible
# rather than merely absent.
#
# The guard is in two halves because of where the information lives:
#
#   * `reserve_checkpoint` runs at MODULE IMPORT. A graph module declares every destination
#     it will ever train into, once, at the point where the suffix constant is defined --
#     so the declaration and the name cannot drift. Two declarations of the same derived
#     path, or any declaration of a protected one, raise there, before anything in the
#     module can be published.
#   * `_claim_checkpoint` runs inside `build_ddi_pipeline`, on the `(model, suffix)` the
#     builder has actually derived for the cell it is about to wire, and refuses anything
#     that was not reserved.
#
# The second half is the structural part. `vh3.decorator.graph` does NOT execute a graph
# function at import -- `FunctionalWorkflowInstanceWrapper.build_executable` calls the body
# lazily, when that one graph is built, which is during `publish` / `publish --dry-run`
# (VERIFIED 2026-08-08). So a registry populated only by the builder would see exactly one
# graph per process and could never compare two rows. Requiring a reservation inverts that:
# the builder does not need to see the other rows, it only needs to see that THIS row was
# declared, and the declarations all happen at import where they can be compared.
#
# Net effect: forgetting to reserve makes a graph fail `nv-check`; reserving something
# another row already holds fails at import; and neither can be published either way.

# Checkpoints that already hold weights and must never be trained into again. The seed
# grid's three are on YT under `constant.YT_MODELS_ROOT`; `__seed0` in particular is
# `ddi_qf-all4`, the PRE-review reference finetune, kept deliberately as a fallback.
# `scripts.finetune_script` does `rm -rf "$DEST_CKPT"` before training and then uploads
# into an existing YT map_node with `create -i` plus per-file `write-file`, so a partial
# retrain leaves the node holding a MIXTURE of old and new shards -- worse than a clean
# clobber, and it passes every downstream loadability check.
_PROTECTED_SUFFIXES = frozenset(constant.ft_suffix(seed) for seed in constant.SEEDS)

# Every destination any imported graph module has reserved: derived path -> owner label,
# and (model, suffix) -> owner label. Two indexes because they answer different questions --
# the path one detects collisions, the pair one answers "did anybody declare this?".
_RESERVED_PATHS: dict[str, str] = {}
_RESERVED_CHECKPOINTS: dict[tuple[str, str], str] = {}


def _checkpoint_paths(model: str, suffix: str) -> tuple[str, ...]:
    """The two places a finetune of `model` into `suffix` writes: local disk, then YT."""
    safe_base = model.replace("/", "__")
    return (
        "results/checkpoints/" + safe_base + suffix,
        constant.YT_MODELS_ROOT + "/" + safe_base + suffix,
    )


def reserve_checkpoint(suffix: str, *models: str, owner: str) -> str:
    """Claim `<model><suffix>` for `owner` on each of `models`, and return `suffix`.

    Call this at module level, as the definition site of the suffix constant, so the name
    a row trains into and the name it registers are the same object and cannot drift.

    From the ticket: "The name **is** the checkpoint directory, so it must be unique -- a
    collision silently overwrites another run's weights." Silently is the operative word:
    `finetune_script` does `rm -rf "$DEST_CKPT"` before training and then uploads into an
    existing YT map_node with `create -i`, so a duplicate name destroys the earlier row's
    weights locally and overwrites them file-by-file on YT, with no error anywhere. By the
    time it is visible, both the 8-GPU job that produced them and the eval that scored the
    wrong ones have been paid for.

    Checked on the derived paths rather than on the suffixes, so this also catches a pair
    that differ only in a character the path does not preserve.

    Keyed on (base model, suffix), not on the suffix alone. The checkpoint directory is
    `<base model>__<suffix>`, so the *same* suffix on two different base models is not a
    collision -- which is what lets every base model reuse `__ddi_qf__all4`. Only the
    protected-suffix check is suffix-only, because those three names belong to the 4B and
    exist on YT under it.

    Re-reserving the same destination for the same `owner` is a no-op, so a row built from
    a loop may reserve idempotently.
    """
    if not models:
        raise ValueError(
            f"{owner}: reserve_checkpoint({suffix!r}) named no base model; the destination "
            "is <model><suffix>, so a suffix alone does not identify a checkpoint"
        )
    for model in models:
        if model == constant.BASE_MODEL and suffix in _PROTECTED_SUFFIXES:
            raise ValueError(
                f"{owner} checkpoint {suffix!r} is already the seed grid's; training it "
                "would overwrite the reference finetune on YT"
            )
        for path in _checkpoint_paths(model, suffix):
            held_by = _RESERVED_PATHS.get(path)
            if held_by is not None and held_by != owner:
                raise ValueError(
                    f"{owner} and {held_by} both write {path}; one would overwrite the other"
                )
            _RESERVED_PATHS[path] = owner
        _RESERVED_CHECKPOINTS[(model, suffix)] = owner
    return suffix


def _claim_checkpoint(model: str, suffix: str) -> None:
    """Refuse to wire a finetune whose destination is protected or was never reserved.

    Runs on the `(model, suffix)` `build_ddi_pipeline` derived, for every finetune cell it
    wires, from whatever module. That is the point: it does not consult a table of rows, it
    consults the cell in front of it, so a module cannot bypass it by not having an entry.
    """
    if model == constant.BASE_MODEL and suffix in _PROTECTED_SUFFIXES:
        raise ValueError(
            f"refusing to wire a finetune into {model + suffix!r}: that checkpoint already "
            "holds weights and is protected. `finetune_script` removes its destination "
            "before training and uploads file-by-file into the existing YT node, so this "
            "row would destroy the reference finetune and leave a mixture of old and new "
            "shards behind. Pick a new suffix and reserve_checkpoint() it"
        )
    if (model, suffix) not in _RESERVED_CHECKPOINTS:
        raise ValueError(
            f"finetune into {model + suffix!r} was never reserved. Every checkpoint this "
            "builder writes must be declared at module import with "
            f'reserve_checkpoint({suffix!r}, {model!r}, owner="<row>"), so two rows cannot '
            "silently claim one directory -- graph bodies run lazily, one per publish, so "
            "the builder alone cannot see the other rows"
        )


def build_ddi_pipeline(
    *,
    arc_reference,
    path,
    arc_token,
    model: str = constant.BASE_MODEL,
    seeds: Sequence[int] = (0,),
    ticket=constant.TICKET,
    eval_models: Sequence[str] | None = None,
    eval_sets: Sequence[str] = (),
    eval_env: Mapping[str, str] | None = None,
    ft_suffix: str | None = None,
    ft_sets: Sequence[str] = (),
    ft_env: Mapping[str, str] | None = None,
    job_variables: Sequence[str] | None = None,
    title: str = _DEFAULT_TITLE,
    notes: Sequence[str] = (),
):
    """Wire the whole DDI chain and return the report block's output.

    export + checkout -> merge -> one finetune/eval cell per seed -> report -> comment.

    * `seeds` — one cell per seed, differing in nothing else. `(0,)` is a single run,
      `(0, 1, 2)` is the seed-replication grid.
    * `ticket` — where the comment goes. Either a plain string or the graph's `ticket`
      option; an option becomes `${global["ticket"]}` and is read at run time, so the
      published graph is not frozen to one ticket.

    There is one report backend: the `Make DataFrame` block running `_REPORT_TEMPLATE`.
    A Starlark one existed alongside it and was deleted in VIBEQUEUE-16 — it hardcoded a
    base-vs-finetuned pair read as `input[0]`/`input[1]`, so it could not render eval-only
    mode's N columns, and it compiled the ticket into its own source, so it could not take
    a graph option. See CHANGELOG.md, 2026-08-08.

    **Eval-only mode.** Pass `eval_models` and the finetune cells disappear: the graph is
    export + checkout -> merge -> one 1-GPU eval per listed model -> report -> comment,
    and `seeds` is ignored. Most of the VIBEQUEUE-2 grid is this shape — rows that vary
    the *eval* against a checkpoint another row already trained. A model id here is
    whatever `yt-fetch` resolves: a base HF id, or the `__`-escaped name of a checkpoint
    already on YT.

    * `eval_sets` — `key=value` overrides of quality.yaml, e.g. `("max_records=500",)`.
      See `scripts.eval_script` for how they reach the config.
    * `eval_env` — extra environment for the eval jobs, e.g. `DDI_KILT_N_PASSAGES`.
    * `title` / `notes` — the comment's first line and its header. A row that does not
      name itself is unreadable next to the twelve others posted on the same ticket.

    Two or more `eval_models` are rendered side by side, one column each, with the
    `ddi_qf - full` gap subtracted out per model.

    **Train-and-eval mode** (the VIBEQUEUE-2 dataset-mix rows 8-11) is the finetune path
    with two knobs on it:

    * `ft_suffix` — the checkpoint name, replacing `constant.ft_suffix(seed)`. The eval
      that follows the finetune targets `model + ft_suffix`, so both blocks are named from
      this one value and cannot point at different weights. Only legal with a single seed:
      N seeds sharing one checkpoint name would train N times into the same directory and
      keep only the last, which is silent and expensive. Whatever it resolves to must have
      been reserved with `reserve_checkpoint` — including the `constant.ft_suffix(seed)`
      default, whose three values are protected and therefore always refused on the 4B.
    * `ft_sets` — `key=value` overrides on the *training* config, e.g.
      `("datasets=[synthetic_qa]",)`. Same `SETS=` mechanism as `eval_sets`, applied to
      experiments/finetune.yaml instead of quality.yaml.
    * `ft_env` — environment for the finetune job, for knobs `ddi.run` reads from the
      environment rather than the config (`DDI_GRAD_CHECKPOINT=1`). `ft_sets` is the right
      place for anything the YAML understands; this is for the rest.

    `eval_sets` / `eval_env` reach the post-finetune eval too, so a row can vary both
    sides; rows 8-11 vary only the train side and leave the eval at quality.yaml.
    """
    seeds = tuple(seeds)

    # `job_variables` overrides the environment every PyDL block is launched with.
    # Default None keeps `constant.PYDL_JOB_VARIABLES` verbatim; a graph passes an
    # explicit tuple only to test what one of them actually does. The leading "" is
    # part of the option's wire format, not a value — see `_PYDL_JOB`.
    _job = _PYDL_JOB
    _job_ft = _PYDL_JOB_FINETUNE
    if job_variables is not None:
        _vars = ("", *job_variables)
        _job = {**_PYDL_JOB, "job_variables": _vars}
        _job_ft = {**_PYDL_JOB_FINETUNE, "job_variables": _vars}
    eval_models = None if eval_models is None else tuple(eval_models)
    if eval_models is not None:
        if not eval_models:
            raise ValueError("eval_models must not be empty: there is nothing to evaluate")
        if len(set(eval_models)) != len(eval_models):
            raise ValueError(f"eval_models must be distinct, got {eval_models!r}")
        if ft_suffix is not None or tuple(ft_sets):
            raise ValueError(
                "ft_suffix/ft_sets configure the finetune, and eval-only mode has none; "
                "drop eval_models to train, or drop these to only evaluate"
            )
    elif not seeds:
        raise ValueError("seeds must not be empty: there is nothing to finetune")
    elif ft_suffix is not None and len(seeds) > 1:
        # The checkpoint name is the identity of the weights (VIBEQUEUE-2, "Models"): one
        # name for N seeds means N runs into one directory, N uploads to one YT node, and
        # only the last one survives — after paying for all N. The per-seed default exists
        # precisely so a grid cannot do this.
        raise ValueError(
            f"ft_suffix={ft_suffix!r} with seeds={seeds!r}: every seed would train into the "
            "same checkpoint and overwrite the previous one; use one seed per ft_suffix"
        )

    # --- code delivery: DL framework + this repo, merged into one tarball -------------
    dl_code = zeliboba_arc_checkout_deterministic(
        arc_token=arc_token,
        pulsar_token="alexkay28-nirvana-token-pulsar",
        yt_token="alexkay28-nirvana-token-yt",
        yt_cluster="hahn",
        arc_reference=arc_reference,
        yt_pool="nirvana-gena-align",
        mr_account="gena",
        use_baked_zdl=False,
    )
    ddi_code = arc_export(
        arc_token=arc_token,
        path=path,
        reference=arc_reference,
        **vh3.block_args(
            code="arc_export",
            options={"cpu-guarantee": 4, **constant.CPU_SCHEDULING},
        ),
    )
    # 16 GB: the defaults (100 MB RAM / 1 GB disk) cannot untar the framework plus this
    # repo and re-tar them, and that is what killed the first run of the sibling graph.
    tar_archive = merge_tar_archives(
        tar_archives=[dl_code, ddi_code],
        cpu_guarantee=100,
        **vh3.block_args(
            code="merge_tar_archives",
            options={"max-ram": 16000, "max-disk": 16000, **constant.CPU_SCHEDULING},
        ),
    )

    payloads = []

    # A cached base-model eval block used to sit here, served from YT on a hit. It was
    # deleted in VIBEQUEUE-16 with the rest of the unreachable path: its cache key covered
    # neither `eval_sets` nor `eval_env`, so it could only ever serve a row running
    # quality.yaml exactly as committed. A row that wants a base column lists the base model
    # in `eval_models` and pays for the GPU job, which also puts both columns in one graph on
    # one split. See CHANGELOG.md, 2026-08-08, before adding any eval cache back.

    # --- eval-only: no finetune, one 1-GPU eval per model -----------------------------
    for eval_model in eval_models or ():
        evaluation = python_3_deep_learning(
            script=tar_archive,
            pool_tree=tuple(constant.POOL_TREE),
            nodes_count=1,
            gpu_count=constant.EVAL_GPUS,
            run_command=scripts.eval_script(eval_model, sets=eval_sets, env=eval_env),
            **_PYDL_COMMON,
            **_job,
            **_pydl_block_args(_block_code("eval", eval_model)),
        )
        payloads.append(evaluation.json_output)

    # --- one independent cell per seed ------------------------------------------------
    for seed in () if eval_models is not None else seeds:
        # One value names the checkpoint for both blocks of the cell: the finetune writes
        # `model + suffix` and uploads it under that name, the eval below fetches exactly
        # that name back. Deriving them separately is how an eval ends up scoring another
        # row's weights.
        suffix = constant.ft_suffix(seed) if ft_suffix is None else ft_suffix
        # Structural half of the collision guard: this is the only place a finetune
        # destination is derived, so checking it here covers every row in every module.
        # See `_claim_checkpoint` and `reserve_checkpoint` above.
        _claim_checkpoint(model, suffix)
        # Block codes name the cell after whatever distinguishes it: the seed on a seed
        # grid, the checkpoint on a row that varies something else. The seed spelling is
        # kept verbatim so the published seed graphs keep their block codes.
        if ft_suffix is None:
            ft_code, eval_code = "finetune_seed%d" % seed, "eval_seed%d" % seed
        else:
            ft_code, eval_code = _block_code("finetune", suffix), _block_code("eval", suffix)
        finetune = python_3_deep_learning(
            script=tar_archive,
            pool_tree=tuple(constant.POOL_TREE),
            nodes_count=1,
            gpu_count=constant.FINETUNE_GPUS,
            run_command=scripts.finetune_script(
                seed, model=model, suffix=suffix, sets=ft_sets, env=ft_env
            ),
            **_PYDL_FINETUNE_COMMON,
            **_job_ft,
            **_pydl_block_args(ft_code),
        )
        # params= carries the ordering dependency as data: the eval must not start until
        # this cell's checkpoint is on YT. The eval script ignores the file itself. Doing
        # it this way makes the ordering a property of the dataflow, so it cannot be lost
        # by someone rewiring execution edges in the UI.
        evaluation = python_3_deep_learning(
            script=tar_archive,
            params=finetune.json_output,
            pool_tree=tuple(constant.POOL_TREE),
            nodes_count=1,
            gpu_count=constant.EVAL_GPUS,
            run_command=scripts.eval_script(model + suffix, sets=eval_sets, env=eval_env),
            **_PYDL_COMMON,
            **_job,
            **_pydl_block_args(eval_code),
        )
        payloads.append(evaluation.json_output)

    # --- render one comment -----------------------------------------------------------
    report = python_run(
        inp=_collect(payloads),
        code=_report_code(ticket, title=title, notes=notes, columns=eval_models or ()),
        requirements=("pandas",),
        max_ram=300,
        **vh3.block_args(name="Make DataFrame", code="python_run"),
    )

    add_comment_to_ticket(
        oauth_token="alexkay28-nirvana-token-tracker",
        field="ticket",
        field_comment="message",
        tickets=report,
        # The layer carrying commet_to_ticket.py. It happens to match the operation's
        # default, but a graph should state what it runs rather than inherit it invisibly.
        job_layer=("255602b3-9019-4ca5-ade0-0ab938fbfdbd",),
        **vh3.block_args(name="Add Comment To Ticket", code="post_grid"),
    )
    return report
