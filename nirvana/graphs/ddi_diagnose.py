"""VIBEQUEUE-2: the 1-GPU bisect for Qwen2.5-7B's non-finite gradients.

Row 2's 7B finetune has failed twice. The first run (`391bfce7`) had no guards, applied the
gradients anyway, and spent eight GPU-hours training on NaN before uploading a NaN
checkpoint and posting a table of zeros. The second (`59da6b26`, on `acda80b6f5`) hit the
new clipping + non-finite guard and aborted after 2.7 minutes — the guards work — but its
message says "training is diverging", and that reading does not survive the log:

    [ddi.finetune] epoch 0 baseline: student_ddi_qf=3.742 teacher_full=2.022   <- healthy
    [ddi.finetune] step 1: non-finite gradients, step SKIPPED (1/20 allowed)
    [ddi.finetune] step 1 loss=0.0006 lr=1.00e-07 (epoch 1, rec 4/13115)

Step 1, at `lr=1e-7`, with a finite loss, before the optimiser has moved a single weight,
on all 8 ranks, and then on every step after it. Nothing has diverged; the backward pass
itself is producing non-finite gradients. Lowering the LR cannot help and clipping a NaN
returns a NaN, so there is nothing to tune — it has to be bisected.

**Why this model and not the other two.** They are not the same kind of network. Read from
`transformers==5.14.1` (the version in the job's `pip.freeze.log`):
`Qwen3_5Model.forward` builds one mask per layer *type* --

    causal_mask_mapping = {
        "full_attention":   create_causal_mask(**mask_kwargs),
        "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
    }

-- and `masking_utils.create_recurrent_attention_mask` opens with `if attention_mask is
None or attention_mask.ndim != 2: return None`. The DDI mask is 4D, so every
`linear_attention` layer runs unmasked. With `full_attention_interval: 4`:

    Qwen3.5-4B-Instruct    Qwen3_5 hybrid    mask reaches  8 of 32 layers
    Qwen3.6-27B-Instruct   Qwen3_5 hybrid    mask reaches 16 of 64 (frozen8 trained 2)
    Qwen2.5-7B-Instruct    Qwen2ForCausalLM  mask reaches 28 of 28

(The 7B really is all 28: `use_sliding_window: false` sets `sliding_window = None`, and
`Qwen2Config.__post_init__` then labels every layer `full_attention`.)

So the 7B does roughly 3.5x the masked-softmax backward of the 4B and 14x that of the 27B
run that succeeded — which is also the order the three runs failed in. The suspect that
fits the evidence is `ddi/masks.py:124`, which fills blocked entries with `float("-inf")`
where HF's own builders use `torch.finfo(dtype).min` and never `-inf`; a `-inf` entry is
the documented way to get NaN out of a fused attention backward when a whole key tile is
masked, and `ddi_qf`'s doc_i x doc_j blocks are exactly that shape.

That is a hypothesis with three plausible carriers -- the fill value, the kernel the head
dim selects (128 on the 7B, 256 on both hybrids), and sheer depth of masked layers. This
graph varies them one at a time on 1 GPU instead of arguing about them on 8.

Publish through Makefile.nirvana, which stamps the instance's identity comment — every row
in this repo publishes into ONE workflow, so instances are otherwise told apart only by
uuid::

    make nv-pin                                  # what the next publish would ship
    make nv-check   GRAPH=ddi_diagnose_7b_grads  # validate, publish nothing
    make nv-publish GRAPH=ddi_diagnose_7b_grads NOTE='DIAGNOSTIC, not a measurement'

Deliberately not the `./bin/bin publish -n ...` form the older module docstrings here used
to show: `-n` is `--dry-run`, not `--name`, so that line validates and publishes nothing.
(The other two graph modules were pointed at `make nv-check` / `nv-publish` in
VIBEQUEUE-16, so this is now the form the whole tree documents.)

The pin is `constant.ARC_REFERENCE`, not your working copy, and `ddi/diagnose_grads.py`
exists only from `be94ea06ef` onward — an instance published against an earlier pin queues
for a GPU and then dies on `No module named ddi.diagnose_grads`. `nv-pin` shows this before
you spend the wait.
"""

import vh3

from junk.alexkay28.ddi.nirvana.common import constant
from junk.alexkay28.ddi.nirvana.graphs import scripts
from junk.alexkay28.ddi.nirvana.graphs.pydl import (
    _PYDL_COMMON,
    _PYDL_JOB,
    _pydl_block_args,
)
from junk.alexkay28.ddi.nirvana.operations.external import (
    arc_export,
    merge_tar_archives,
    python_3_deep_learning,
    zeliboba_arc_checkout_deterministic,
)

_QWEN25_7B = "Qwen/Qwen2.5-7B-Instruct"

# No aggregate and no Tracker comment, unlike every other graph here. Those two blocks
# render `tables[kind=quality]` into a metrics table, and this run produces no metrics —
# forcing its output through them would either fail the report or, worse, post something
# that reads like a result. The product is the block's own log plus
# `logs/diagnose_grads.json`, and it is meant to be read by a human once.
_DIAGNOSE_ARGS = ("--records", "3", "--dataset", "hotpot_qa")


def _diagnose_graph(model: str, arc_reference, path, arc_token, args=_DIAGNOSE_ARGS):
    """export + checkout -> merge -> one 1-GPU diagnostic block. Nothing downstream."""
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
    tar_archive = merge_tar_archives(
        tar_archives=[dl_code, ddi_code],
        cpu_guarantee=100,
        **vh3.block_args(
            code="merge_tar_archives",
            options={"max-ram": 16000, "max-disk": 16000, **constant.CPU_SCHEDULING},
        ),
    )
    return python_3_deep_learning(
        script=tar_archive,
        pool_tree=tuple(constant.POOL_TREE),
        nodes_count=1,
        gpu_count=constant.EVAL_GPUS,
        run_command=scripts.diagnose_script(model, args=args),
        **_PYDL_COMMON,
        **_PYDL_JOB,
        **_pydl_block_args("diagnose_grads"),
    )


@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_diagnose_7b_grads(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
):
    """VIBEQUEUE-2 diagnostic: bisect Qwen2.5-7B's step-1 non-finite gradients (1 GPU)."""
    _diagnose_graph(_QWEN25_7B, arc_reference, path, arc_token)


# --- follow-up: which of the two `default` kernels is the broken one --------------------
#
# The first sweep answered the question it was built for, and eliminated everything it was
# built to eliminate. On Qwen2.5-7B, 24 of 24 `default`-backend conditions produced NaN
# gradients and 24 of 24 `math` conditions were finite; the mask fill value (`-inf` vs
# `finfo.min`) changed nothing, gradient checkpointing changed nothing, and `full` failed
# exactly like `ddi_qf` -- so this was never about the DDI mask's geometry, and the `-inf`
# suspicion was wrong. The 4B control was finite in all 48. Forwards were finite everywhere.
#
# What is left is kernel selection. `default` lets SDPA choose, and the plausible reason the
# 4B escapes is that it cannot choose the same thing: head_dim is 128 on the 7B and 256 on
# both hybrids, and flash does not serve 256.
#
# This matters for cost, not just for the write-up. `math` materialises the full T x T score
# matrix per layer -- at seq_len ~1700 over 28 layers that is a real throughput and memory
# tax on an 8-GPU finetune. If only FLASH is broken, pinning EFFICIENT_ATTENTION costs
# approximately nothing and row 2 on the 7B can run at its original price. So the narrowing
# is worth its few GPU-minutes before committing the expensive job to a backend.
#
# A backend that refuses these shapes outright is a result too, not a failure: `_one_step`
# catches and reports it, and "flash never ran" would move the blame to EFFICIENT.
_BACKEND_ARGS = ("--records", "1", "--dataset", "hotpot_qa",
                 "--fills", "finfo_min", "--grad-checkpoint", "0",
                 "--backends", "flash,efficient,math")


@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_diagnose_7b_backends(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
):
    """VIBEQUEUE-2 diagnostic: split the 7B's failing `default` kernel into flash vs efficient."""
    _diagnose_graph(_QWEN25_7B, arc_reference, path, arc_token, args=_BACKEND_ARGS)


@vh3.decorator.graph("https://nirvana.yandex-team.ru/flow/" + constant.WORKFLOW_ID + "/graph")
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_names(arc_reference="arc_reference")
def ddi_diagnose_4b_grads(
    *,
    arc_reference: vh3.String = constant.ARC_REFERENCE,
    path: vh3.String = constant.ARC_PATH,
    arc_token: vh3.Secret = "alexkay28-nirvana-token-arcadia",
):
    """VIBEQUEUE-2 diagnostic: the same sweep on Qwen3.5-4B, the model that trains (1 GPU).

    The control. Without it a clean 7B condition proves nothing — the sweep has to show
    the 4B staying finite everywhere the 7B does not, on the same code and the same
    records. It also prints the `layer_types` census, which is the direct check that the
    DDI mask reaches only 8 of this model's 32 layers.
    """
    _diagnose_graph(constant.BASE_MODEL, arc_reference, path, arc_token)
