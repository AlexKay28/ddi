"""Shell bodies for the DDI PyDL blocks.

These are the `run_command` of each job. They live here, in Arc, instead of in an
`Echo` block wired to the job's option: the graph they replace kept its real script in
an Echo and left a *stale* `run-command` on the block itself, so the Nirvana UI showed
a script that trained `Qwen/Qwen2.5-7B-Instruct` while the graph actually trained a 4B.

Two behavioural changes from those Echo scripts, both deliberate:

1. **No `pip install --upgrade` at job start.** Package pins live in the block's `pip`
   option (``constant.PYDL_PIP``). Upgrading at runtime meant two runs a week apart were
   not the same experiment.
2. **Failures fail the block.** The originals ran `set -uo pipefail` (no `-e`) and
   `continue`d past a failed fetch/finetune, then still wrote `json_output` and exited 0
   — so a broken finetune reported success and the eval downstream silently scored a
   stale checkpoint. These exit non-zero instead.
"""

from collections.abc import Mapping, Sequence

from junk.alexkay28.ddi.nirvana.common import constant

# Common prologue: offline HF, writable cache, dataset record cache.
_PREAMBLE = """
set -uo pipefail

export YT_PROXY={yt_proxy}
export HF_HOME="${{HF_HOME:-$HOME/hf_home}}"
mkdir -p "$HF_HOME"          # /mnt/models is not writable on these hosts
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$SOURCE_CODE_PATH"

# hotpot_qa / musique / kilt_nq loaders need results/splits/*_records.jsonl to resolve
# offline (ddi/data/records_cache.py). Without it those datasets silently drop out of
# the eval, which looks like a quality change rather than a missing input.
make yt-fetch-records || {{ echo "FATAL: yt-fetch-records failed"; exit 1; }}
""".format(yt_proxy=constant.YT_PROXY)

# Emit every logs/*.json as one JSON array on the block's json_output endpoint.
_EMIT_JSON_OUTPUT = """
echo "=== logs ==="
ls -la logs || true
if ls logs/*.json >/dev/null 2>&1; then
    python3 -c '
import glob, json, sys
out = []
for p in sorted(glob.glob("logs/*.json")):
    with open(p) as fh:
        out.append(json.load(fh))
json.dump(out, open(sys.argv[1], "w"))
' "$JSON_OUTPUT_FILE"
else
    echo "FATAL: no logs/*.json produced"
    exit 1
fi
"""


def finetune_script(
    seed: int = 0,
    model: str = constant.BASE_MODEL,
    suffix: str | None = None,
    sets: Sequence[str] = (),
    env: Mapping[str, str] | None = None,
) -> str:
    """Self-distillation finetune for one grid cell, then mirror the checkpoint to YT.

    `seed` selects the cell. Both the seed and the checkpoint directory go through
    `--set` (as `SETS=` on the Makefile), which is what makes cells independent: before
    that flag existed every cell trained into the single `output_dir:` hardcoded in
    experiments/finetune.yaml and clobbered its predecessor. The overrides are folded
    into ddi.run's config hash, so cells stay distinguishable in results/runs.jsonl.

    `model` defaults to the ticket's base model, so `finetune_script(seed)` is byte-for-byte
    what it always was; it exists so a graph can be spawned against another base model.

    `suffix` overrides the checkpoint directory's name, which is otherwise
    `constant.ft_suffix(seed)` -> `__ddi_qf__seed{N}`. VIBEQUEUE-2's Models table names
    checkpoints after what the row *varies* (`__ddi_qf__synth`), not after its seed, and
    every row of that grid holds seed=0 — so without this every mix row would train into
    `__ddi_qf__seed0` and overwrite the reference finetune, locally and on YT. The name is
    the identity of the weights: it feeds both `DEST_CKPT` and the YT upload path, from
    this one value, so the two cannot drift apart.

    `sets` are extra `key=value` overrides appended to the same `SETS=` variable, for the
    train-side knobs that have no flag — `datasets=[synthetic_qa]`, `max_records=500`.
    They are validated exactly like the eval side's (see `_sets_arg`): Make splits `SETS`
    on whitespace, so an entry containing a space would arrive at `ddi.run` as two broken
    overrides.

    `env` is extra environment for the job, for the knobs read from the environment rather
    than from the config — `DDI_GRAD_CHECKPOINT=1` on the cells big enough to need it.
    """
    extra_sets = _validate_sets(sets)
    return _PREAMBLE + """
{exports}MODEL="{base}"
SEED={seed}
SAFE_NAME="$(echo "$MODEL" | sed 's#/#__#g')"
DEST_CKPT="results/checkpoints/${{SAFE_NAME}}{suffix}"

make yt-fetch MODEL="$MODEL" || {{ echo "FATAL: yt-fetch failed for $MODEL"; exit 1; }}

rm -rf "$DEST_CKPT"
# Deliberately NOT `|| exit 1`. Every rank segfaults in NCCL/UCX teardown *after* training
# prints "finetune: done - 1 passed, 0 failed", which poisons torchrun's exit code. On
# 2026-08-06 that discarded three separate successful 4-hour 8-GPU runs at the final step:
# the guard exited before the upload below, so the checkpoints died with their jobs.
# The checkpoint test on the next line is the real gate — trust the artifact, not the exit
# status. A genuine training failure still fails here, because it leaves no checkpoint.
make finetune MODEL="$MODEL" NUM_GPUS={gpus} \\
     SETS="output_dir=$DEST_CKPT seed=$SEED{extra_sets}" \\
     FT_SUMMARY="logs/finetune_seed${{SEED}}_summary.json" \\
  || echo "WARN: make finetune exited non-zero (teardown segfault?) - verifying checkpoint"

# `-d` alone is too weak a gate. The directory is created before the shards are written, so
# a run that dies *during* the save leaves it behind holding a partial shard set — which
# would sail through this check and upload a checkpoint nothing can load. Require the files
# a loadable checkpoint must actually have, so "broke halfway" fails here rather than three
# blocks downstream when the eval cannot load the model.
[ -d "$DEST_CKPT" ] || {{ echo "FATAL: no checkpoint at $DEST_CKPT"; exit 1; }}
[ -s "$DEST_CKPT/config.json" ] \\
  || {{ echo "FATAL: checkpoint at $DEST_CKPT has no config.json"; exit 1; }}
ls "$DEST_CKPT"/*.safetensors >/dev/null 2>&1 \\
  || {{ echo "FATAL: checkpoint at $DEST_CKPT has no safetensors shards"; exit 1; }}
# A multi-shard save is only loadable with its index; without it transformers cannot
# assemble the model, and the failure would surface as a confusing eval-side error.
if ls "$DEST_CKPT"/*-of-*.safetensors >/dev/null 2>&1; then
    [ -s "$DEST_CKPT/model.safetensors.index.json" ] \\
      || {{ echo "FATAL: sharded checkpoint at $DEST_CKPT has no index json"; exit 1; }}
fi
echo "checkpoint -> $DEST_CKPT"

echo "=== upload checkpoint to YT ==="
DEST="{models_root}/${{SAFE_NAME}}{suffix}"
yt --proxy={yt_proxy} create map_node "$DEST" -r -i || exit 1
shopt -s globstar
for f in "$DEST_CKPT"/**; do
    [ -f "$f" ] || continue
    echo "uploading $(basename "$f") ..."
    yt --proxy={yt_proxy} write-file "$DEST/$(basename "$f")" < "$f" || exit 1
done
""".format(
        base=model,
        seed=seed,
        suffix=constant.ft_suffix(seed) if suffix is None else suffix,
        extra_sets="".join(" " + entry for entry in extra_sets),
        exports=_exports(env),
        gpus=constant.FINETUNE_GPUS,
        models_root=constant.YT_MODELS_ROOT,
        yt_proxy=constant.YT_PROXY,
    ) + _EMIT_JSON_OUTPUT


def _validate_sets(sets: Sequence[str]) -> tuple[str, ...]:
    """The `key=value` entries, checked for the two ways they silently break.

    `SETS` is a space-separated Make variable (`SET_ARGS = $(foreach s,$(SETS),--set '$(s)')`),
    so an entry containing a space would silently become two half-parsed overrides. Reject
    that here rather than at `ddi.run`'s argument parser, four layers down inside a GPU job.
    Practical consequence for the VIBEQUEUE-2 mix rows: a two-dataset override has to be
    written `datasets=[hotpot_qa,musique]`, with no space after the comma. YAML parses that
    flow sequence identically, so nothing is lost.

    Shared by the train side (`finetune_script`) and the eval side (`eval_script`) — the
    Makefile expands `SETS` the same way for both, so one rule covers them.
    """
    entries = tuple(sets)
    for entry in entries:
        if any(c.isspace() for c in entry):
            raise ValueError(
                f"--set entry {entry!r} contains whitespace; Make splits SETS on spaces, "
                "so it would arrive at ddi.run as two broken overrides"
            )
        if "=" not in entry:
            raise ValueError(f"--set entry {entry!r} is not 'key=value'")
        if '"' in entry:
            # The entry is interpolated into a double-quoted `SETS="..."` word in the
            # emitted shell; a quote there ends it early and the rest becomes stray argv.
            raise ValueError(f"--set entry {entry!r} contains a double quote")
    return entries


def _sets_arg(sets: Sequence[str]) -> str:
    """` SETS="a=1 b=2"`, or the empty string when there is nothing to override."""
    entries = _validate_sets(sets)
    if not entries:
        return ""
    return ' SETS="%s"' % " ".join(entries)


def _exports(env: Mapping[str, str] | None) -> str:
    """`export K="V"` lines, sorted, each newline-terminated; empty string for no env.

    These are knobs read from the environment rather than from the experiment YAML —
    `DDI_KILT_N_PASSAGES` (ddi/data/kilt_nq.py) is the only one so far. Sorted so the same
    mapping always produces the same script, which keeps the block's cache key stable.
    """
    if not env:
        return ""
    return "".join('export %s="%s"\n' % (key, value) for key, value in sorted(env.items()))


def eval_script(
    model: str,
    sets: Sequence[str] = (),
    env: Mapping[str, str] | None = None,
) -> str:
    """Run experiments/quality.yaml against one model id.

    `model` is whatever `yt-fetch` can resolve: a base HF id, or a fine-tuned name whose
    YT node is the `__`-escaped form (the Makefile tries the escaped name first, then
    falls back to `basename` for base models).

    `sets` are `key=value` overrides of quality.yaml's top-level keys — `max_records=500`,
    `max_records=null`, `datasets=[kilt_nq]`. They ride the Makefile's existing `SETS=`
    variable: `exp-%` already expands it into `--set` arguments on the `ddi.run` step, the
    same way `finetune` does, so no Makefile change was needed to reach `quality.yaml`.
    The `ddi.aggregate` step of `exp-%` deliberately does not receive them — it reads
    `results/runs.jsonl` and the YAML's `report:` block, neither of which any of these keys
    touches, so forwarding them there would change nothing. `ddi.run` folds the overrides
    into the run's config hash, so two evals differing only by `sets` stay distinguishable.

    `env` is extra environment for the job, for the knobs that are read from the
    environment instead of the config.

    With neither argument the emitted script is byte-for-byte what it always was.
    """
    return _PREAMBLE + """
MODEL="{model}"
{exports}make yt-fetch MODEL="$MODEL" || {{ echo "FATAL: yt-fetch failed for $MODEL"; exit 1; }}
make exp-quality MODEL="$MODEL" BACKENDS=sdpa{sets} || {{ echo "FATAL: exp-quality failed"; exit 1; }}
""".format(model=model, exports=_exports(env), sets=_sets_arg(sets)) + _EMIT_JSON_OUTPUT


def diagnose_script(
    model: str,
    args: Sequence[str] = (),
    env: Mapping[str, str] | None = None,
) -> str:
    """Run `ddi.diagnose_grads` against one model id — the non-finite-gradient bisect.

    A diagnostic, not a measurement: it writes no checkpoint, touches no store, and its
    product is the table in `logs/diagnose_grads.json`. It exists because Qwen2.5-7B
    produces non-finite gradients on step 1 at `lr=1e-7` with a finite loss, which is a
    backward-pass fault rather than divergence, and the finetune path is far too expensive
    a place to bisect one.

    `args` are appended verbatim to the module's argv (`--truncate-layers 4`, `--fills
    -inf`), so the same block can be re-pointed at a narrower sweep without a code change.
    They are checked for whitespace-free-ness only where it matters — each entry becomes
    one shell word — so pass a flag and its value as two entries.
    """
    for entry in args:
        if any(c.isspace() for c in entry) or '"' in entry:
            raise ValueError(
                f"diagnose arg {entry!r} contains whitespace or a quote; each entry becomes "
                "one shell word, so pass a flag and its value as separate entries"
            )
    return _PREAMBLE + """
MODEL="{model}"
{exports}make yt-fetch MODEL="$MODEL" || {{ echo "FATAL: yt-fetch failed for $MODEL"; exit 1; }}
python3 -m ddi.diagnose_grads --model "$MODEL"{args} \\
  || {{ echo "FATAL: ddi.diagnose_grads failed"; exit 1; }}
""".format(
        model=model,
        exports=_exports(env),
        args="".join(" " + entry for entry in args),
    ) + _EMIT_JSON_OUTPUT
