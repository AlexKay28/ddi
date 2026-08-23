"""Constants shared by the DDI Nirvana graphs.

Everything here is a value the graph needs and that is *not* a hyper-parameter of the
experiment itself — experiment settings live in ``ddi/experiments/*.yaml``.
"""

# Where the graph publishes to. Same workflow as the hand-built graph this replaces:
# https://nirvana.yandex-team.ru/flow/e0f1a670-0478-4b25-a176-0fb4b26d2a29
WORKFLOW_ID = "e0f1a670-0478-4b25-a176-0fb4b26d2a29"

# Arcadia subtree shipped to every job. `ddi/` currently lives only on the
# ALIGNMENT-2445 branch, so ARC_REFERENCE must be a commit hash from that branch —
# never "trunk", which the deterministic checkout op rejects outright.
ARC_PATH = "junk/alexkay28/ddi"

# The revision every graph exports — single source of truth, referenced by each graph's
# `arc_reference` default. Bump it whenever a fix has to reach a fresh instance.
#
# THE TRAP, and it has bitten twice: `publish` pins a new instance to the graph's DEFAULT,
# not to whatever you last ran, so a stale value here silently ships old code to every block
# of a fresh run. Once it was two values instead of one — ddi_qwen and ddi_grid each carried
# their own hash (83b2840a and 50a6c3b5), both predating the DDP barrier timeout fix in
# 2b2a8a891a, so publishing either would have deadlocked every GPU cell on the epoch-0
# holdout perplexities. Once it was one stale value: an instance published below 819bd015f9
# still ran interpreter shutdown, so the NCCL/UCX teardown segfault poisoned torchrun's exit
# code and a complete 4h18m 8-GPU run was discarded on 2026-08-06. Keeping ONE value here
# means a fix cannot reach one graph and miss the other.
#
# WHY THIS PIN IS A FLOOR, NOT A PREFERENCE. Almost everything under `ddi/` and
# `experiments/` is job-side: it reaches a run only through this value, whatever the graph
# files say. Published below the current pin, a run measures a harness the ticket has
# already declared superseded — a decode path that dies at the first step on a bf16
# alignment fault, a model that reasons through its whole `max_new_tokens` budget and scores
# em 0.0000, a job that starts with a 55,186-row metrics store and averages months-old rows
# into its table, a finetune that trains on NaN for ~3,230 steps and reports success, and a
# diagnostic block that dies on `No module named ddi.diagnose_grads`. Do not lower this to
# reproduce an old number; use the documented env knobs (`DDI_PROMPT_INSTRUCTION`,
# `DDI_KILT_TITLE_PLACEHOLDER`), which exist for precisely that.
#
# Per-revision archaeology — what each bump fixed and what it cost to find out, every entry
# a real incident — lives in CHANGELOG.md under "Pin history". Read it before lowering this.
ARC_REFERENCE = "be94ea06efd8dbc85eb74505502eebcb858cd8f1"

# Tracker ticket the result comment is posted to.
#
# It was `ALIGNMENT-2445` — the branch's ticket, not the grid's — while every live entry
# point passed the literal `"VIBEQUEUE-2"` as its `ticket` default, so this constant only
# ever reached the two legacy graphs that took no `ticket` option at all. Those two posted
# their tables to the wrong ticket for as long as they existed; they were deleted in
# VIBEQUEUE-16 and the value corrected here, with the entry points now defaulting to this
# constant instead of restating it 38 times.
TICKET = "VIBEQUEUE-2"

# YT
YT_PROXY = "watt"
YT_MODELS_ROOT = "//home/gena/alexkay28/tickets/ALIGNMENT-2445/models"

# `YT_BASELINES_ROOT` used to live here, pointing at cached base-model eval summaries keyed
# on model + a hash of quality.yaml. It went with the cached-baseline block in VIBEQUEUE-16:
# that key covered neither the `--set` overrides nor the environment, so it could only ever
# serve a row running the config exactly as committed. Every surviving row instead puts the
# base model in `eval_models` and scores both columns in one graph on one split, which is
# what makes the subtraction between them paired. See CHANGELOG.md, 2026-08-08, for the full
# argument -- read it before adding any eval cache back.

# --- the experiment ------------------------------------------------------------------
# The datasets the jobs actually run. Mirrors `datasets:` in BOTH experiments/quality.yaml
# and experiments/finetune.yaml, which are identical and must stay that way -- a train/eval
# dataset mismatch is exactly the confound this module exists to avoid.
#
# kilt_nq was dropped in d1e2f6662c because its loader now raises rather than fabricating
# title-only documents (R1). Four separate graph notes went on advertising it anyway, so
# every grid post claimed "hotpot_qa, musique, kilt_nq, synthetic_qa (all four, always)"
# while nine cells ran instead of twelve. Nothing failed; the report was simply describing
# a run that did not happen. Write the list HERE and nowhere else.
#
# `tests/test_graph_notes.py` asserts this tuple equals the YAML, so the next edit to
# either one fails a test instead of quietly producing a wrong report.
EVAL_DATASETS = ("hotpot_qa", "musique", "synthetic_qa")

# The rendered note lines themselves, not just the list they are built from.
#
# `EVAL_DATASETS` living here was only half the fix. The note string was still assembled
# inside `ddi_rows.py`, which needs `vh3` and therefore cannot be imported by a test — so
# `tests/test_graph_notes.py` re-derived the string locally and asserted its own copy
# equalled the constant, which can never fail. It duly missed a FIFTH hardcoded copy
# sitting in `_rebaseline_row`, correct at the time and one edit away from going stale
# exactly like the four before it.
#
# Rendering here makes the object the graphs use the same object the test imports, so the
# assertion is on the real thing rather than on a reproduction of it.
DATASETS_NOTE = "datasets:  " + ", ".join(EVAL_DATASETS)

# Settings every quality note states back, because the comments land next to a dozen
# others on one ticket and "which eval was this" is not recoverable from the numbers.
FIXED_NOTE_LINES = (
    "masks:     full, ddi_qf, blind",
    "split:     val (ddi/run.py `_quality_cell` loads `val`)",
    "metrics:   em, f1, presence, loss",
    "backends:  sdpa",
)

# Base model, exactly as HuggingFace names it.
BASE_MODEL = "Qwen/Qwen3.5-4B-Instruct"

# What `make finetune` produces and `upload_models_to_yt` mirrors. The suffix encodes
# the student mask + seed; `ddi.finetune` writes the checkpoint, the graph uploads it.
FT_SUFFIX = "__ddi_qf__seed0"
FT_MODEL = BASE_MODEL + FT_SUFFIX


def ft_suffix(seed: int) -> str:
    """Checkpoint suffix for one grid cell: student mask + seed."""
    return f"__ddi_qf__seed{seed}"


def ft_model(seed: int) -> str:
    """HF-style id of the fine-tuned checkpoint for `seed`, as yt-fetch resolves it."""
    return BASE_MODEL + ft_suffix(seed)


# Grid axis A from VIBEQUEUE-2: seed replication. Runs first because a single seed cannot
# separate a base-vs-finetuned delta from run-to-run noise, which makes every other axis
# uninterpretable until the spread is known.
SEEDS = (0, 1, 2)

# 8-GPU DDP for the finetune; a 4B eval fits comfortably on one card.
FINETUNE_GPUS = 8
EVAL_GPUS = 1

POOL_TREE = ["gpu_hutang_141g"]

# The infiniband module every 8-GPU finetune is pinned to, via `scheduling_segment_modules`
# in the block's YT custom spec.
#
# WHY THIS EXISTS. An 8-GPU gang lands in YT's `large_gpu` scheduling segment, and that
# segment is *module-aware*: the tree's `module_type` is `infiniband_cluster`, and the
# scheduler may reassign an operation to a different module every
# `module_reconsideration_timeout` (1200 s on this tree). A reassignment ABORTS the running
# gang -- abort reason `wrong_scheduling_segment_module` -- and the job restarts from
# scratch. `ddi.finetune` has no mid-run checkpointing, so every such abort throws away all
# progress. VERIFIED 2026-08-07: row 2's 4B finetune had been "running" for over three
# hours across four incarnations (2 x wrong_scheduling_segment_module + 1 lost job) and its
# best incarnation reached only step 700 of ~3300 before being moved. Nothing looks wrong
# from the outside: Nirvana reports `running`, progress climbs, no block fails.
#
# WHY YATI11 SPECIFICALLY. The tree's `module_assignment_heuristic` is
# `min_remaining_feasible_capacity` -- it packs an operation into the SMALLEST module that
# still fits, which is also the one most likely to need rebalancing later. That is how a
# 3-hour 8-GPU job ended up in YATI13-YP, a 24-GPU module with zero remaining capacity.
# Module capacity on 2026-08-07 (total / remaining GPUs): YATI11 4736/1264,
# YATI11-DEGRADED 448/16, YATI12 432/0, YATI11-YP 96/16, YATI12-YP 72/16, YATI13 72/0,
# YATI13-YP 24/0. YATI11 is two orders of magnitude larger than what one cell needs, so the
# scheduler has no reason to move a job out of it.
#
# THE TRADE-OFF, stated plainly: a pinned operation WAITS when its module is full instead
# of being placed elsewhere. That is the intended bargain -- for a job with no checkpoint
# and a multi-hour runtime, queueing is recoverable and an abort is not.
#
# An unknown module name is rejected by the scheduler ("Segment modules %v are not found in
# the tree", scheduling_segment_manager.cpp), so a typo here fails loudly rather than
# silently dropping the pin. Verify a candidate against
# //sys/scheduler/orchid/scheduler/pool_trees/gpu_hutang_141g/scheduling_segments before
# changing it.
LARGE_GPU_MODULE = "YATI11"

# --- CPU scheduling for the code-delivery blocks ---------------------------------------
# Nirvana derives a CPU job's pool from the quota name plus a *priority class*, so a block
# that names no pool lands in `nirvana-gena-align-normal` — not a misconfiguration, just
# the default. On 2026-08-06 that pool was jammed: 0 of 376 operation slots running against
# 418 queued, demand 583 CPU against a 150 guarantee, while the parent pool ran 4/75
# uncontended. Code delivery sat behind that queue for over two hours per graph, and the
# four live operations had to be moved by hand with `yt_pools.py move`.
#
# Naming the parent explicitly keeps `arc_export` and `merge_tar_archives` out of that
# queue. `job-scheduler-yt-token` is NOT optional decoration: Nirvana requires a YT token
# whenever a custom pool is set, and omitting it makes the pool option silently ineffective.
# The pool must exist in the tree the job actually runs in — `nirvana-gena-align` is real on
# hahn/physical (where these CPU blocks run) and absent from the watt GPU tree.
CPU_YT_POOL = "nirvana-gena-align"
YT_TOKEN_SECRET = "alexkay28-nirvana-token-yt"
CPU_SCHEDULING = {
    "job-scheduler-yt-pool": CPU_YT_POOL,
    "job-scheduler-yt-token": YT_TOKEN_SECRET,
}

# PyDL image + pins. Pinned deliberately: the graph this replaces ran
# `pip install --upgrade transformers` at job start, which made two runs a week apart
# genuinely different experiments.
PYDL_IMAGE = ["45a4ab44-cd51-4400-9861-3c3626c75595"]
PYDL_PIP = [
    "yandex-pydl-event-logger",
    "yandex-save-interval-selector==0.0.6",
    "sacrebleu==2.5.1",
    "numpy==2.2.6",
    # Pinned, not ">=": the PyDL image ships a transformers that already satisfies
    # ">=5.11.0", so pip reports "Requirement already satisfied" and never upgrades —
    # and that preinstalled copy does not know the `qwen3_5` architecture, so loading
    # Qwen3.5-4B dies with `KeyError: 'qwen3_5'`. The graph this replaces worked around
    # it with `pip install --upgrade transformers` at job start; removing that line
    # without pinning here is what broke run 2. 5.14.1 = newest on the internal index.
    "transformers==5.14.1",
    "bitsandbytes>=0.50.0",
]
PYDL_JOB_VARIABLES = [
    "DSMT_YT_WRAPPER_LOCAL_RPC=1",
    "MLPROXY_YT_WRAPPER_LOCAL_RPC=1",
    "UNITEDNN_MULTIPROCESSING_TRANSFORM_LD_PRELOAD=",
]
PYDL_ENVIRONMENT = ["unset CUDA_INJECTION64_PATH"]
PYDL_TTL_MIN = 10080

# A finetune job gets its own, much tighter TTL. The default above is 7 days, which is a
# safety net for evals but a disaster for an 8-GPU training cell: on 2026-08-06 a job whose
# user process had already exited (`FATAL: finetune failed` written, log finalised with
# __END_OF_LOG__) stayed alive in YT for over three hours holding all 8 GPUs, and with the
# 7-day TTL nothing would have reclaimed them for a week. The wedge is in the PyDL/YT layer,
# not in our script, so the script cannot free the slot itself — only a TTL can. A finetune
# cell runs ~4.5h, so 24h leaves generous headroom for a bigger mix while bounding the cost
# of a hang to one day instead of seven.
FINETUNE_TTL_MIN = 1440
