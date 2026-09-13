"""The `Python 3 Deep Learning` block settings every DDI graph shares. No graphs here.

This file declares nothing publishable. It is the settings layer under
`ddi_pipeline.build_ddi_pipeline` and `ddi_diagnose`: the job options, the YT custom spec,
the pip pins and the block identity that every PyDL cell in this project is built from.

It was `ddi_qwen.py` and it did own a graph — `ddi_qwen35_4b_finetune_eval`, the first
vh3 graph in the project. That graph was deleted in VIBEQUEUE-16: it predated `title=` /
`notes=`, so its Tracker comment carried the default header and could not be told apart
from the dozen others on the ticket; it posted to `constant.TICKET` rather than the
`VIBEQUEUE-2` every live entry point passes; it spread `_PYDL_JOB` on its finetune block
instead of `_PYDL_JOB_FINETUNE`, so its 8-GPU cell ran without the infiniband module pin;
and it trained into `__ddi_qf__seed0`, the pre-review reference checkpoint, which
`scripts.finetune_script` would have `rm -rf`'d before training. The module was renamed so
nothing is named after a graph it no longer contains.

Everything below — image, pip pins, ports, NCCL flags, YT scheduler settings, disk, the
`user-requested-secret` extra option — is carried over verbatim from the hand-built graph
this project replaced (workflow `e0f1a670-0478-4b25-a176-0fb4b26d2a29`). The one thing
deliberately dropped is `mlproxy_config`: those blocks carried a TOML pointing at
ALIGNMENT-2662 / ALIGNMENT-2727 checkpoint dirs, inherited from whatever graph they were
cloned from. It is inert (`enable_mlproxy=False`) and belongs to another ticket.

There is nothing to publish from this module. The publishable entry points live in
`ddi_rows.py` and `ddi_diagnose.py`; `make nv-list` enumerates them.
"""

import json

import vh3
from vh3.models.graph import BlockCacheOptions
from vh3.models.operation.operation_option import OperationOption

from junk.alexkay28.ddi.nirvana.common import constant

# Job settings shared by all three PyDL blocks, lifted verbatim off the live graph.
# Anything removed from here changes scheduling or networking behaviour, which is exactly
# where an 8-GPU DDP run breaks in ways that look like a model problem.
_PYDL_COMMON = dict(
    pulsar_token="alexkay28-nirvana-token-pulsar",
    image=tuple(constant.PYDL_IMAGE),
    pip=tuple(constant.PYDL_PIP),
    ttl=constant.PYDL_TTL_MIN,
    environment=tuple(constant.PYDL_ENVIRONMENT),
    mlproxy_custom_version="0.3.5",
)

# Same settings, tighter deadline, for the 8-GPU training cells only. `_PYDL_COMMON` already
# supplies `ttl`, so this has to override it rather than be passed alongside — passing both
# is a duplicate-keyword TypeError, not a merge. See constant.FINETUNE_TTL_MIN for why a
# training cell must not inherit the 7-day eval TTL.
_PYDL_FINETUNE_COMMON = {**_PYDL_COMMON, "ttl": constant.FINETUNE_TTL_MIN}

# The custom YT spec shared by every block, and the finetune-only variant that adds the
# infiniband module pin. Built with `json.dumps` rather than written as a literal so the
# two cannot drift: the eval blocks and the training blocks must agree on everything
# except the pin, and a hand-maintained second JSON string is exactly how that stops being
# true. See `constant.LARGE_GPU_MODULE` for why the pin exists at all.
_YT_CUSTOM_SPEC = {"job_cpu_monitor": {"enable_cpu_reclaim": False}, "weight": 0.95}
# Only the 8-GPU cells belong to the module-aware `large_gpu` segment; a 1-GPU eval sits in
# `default`, where modules are not assigned and the pin would be meaningless.
_YT_CUSTOM_SPEC_FINETUNE = {
    **_YT_CUSTOM_SPEC,
    "scheduling_segment_modules": [constant.LARGE_GPU_MODULE],
}

# The private job options the live blocks set. These are ordinary keyword arguments —
# but only because `operations/external.py` was generated with `--import-private-options`.
# See the note at the top of that file: without it they cannot be passed at all.
_PYDL_JOB = dict(
    mpi_processes_count=0,
    job_variables=("", *constant.PYDL_JOB_VARIABLES),
    install_pydl_package=False,
    job_scheduler_yt_custom_spec=json.dumps(_YT_CUSTOM_SPEC),
    nccl_debug_file_enable=True,
    disable_nccl_telemetry=True,
    ports=(
        "unitednn_distributed_run_handshake_barrier_port=any tcp master,nocheck",
        "unitednn_distributed_run_main_barrier_port=any tcp master,nocheck",
        "btl_port = any tcp both,nocheck",
        "oob_port = any tcp both,nocheck",
        "sshd_port = any tcp both,nocheck",
    ),
    allow_partial_node_multihost=True,
    enable_yt_job_rpc_proxy=True,
    job_scheduler_yt_pool="nirvana-gena-align-projects-extinfo",
    mr_account="gena",
    max_disk=1600000,
    job_scheduler_instance="watt",
    underdeep_token="alexkay28-nirvana-token-underdeep",
    yt_job_rpc_proxy_threads=8,
)

# Same job options, with the 8-GPU cells pinned to one infiniband module. Spread this
# INSTEAD of `_PYDL_JOB` on a finetune block — spreading both is a duplicate-keyword
# TypeError, not a merge.
_PYDL_JOB_FINETUNE = {
    **_PYDL_JOB,
    "job_scheduler_yt_custom_spec": json.dumps(_YT_CUSTOM_SPEC_FINETUNE),
}


def _pydl_block_args(code: str):
    """Block identity plus the two options that genuinely are not on the operation.

    `user-requested-secret` and `rn_token` are the only real extras — everything else the
    blocks set is a private option of the operation itself and is passed as a kwarg.
    """
    return vh3.block_args(
        name="Python 3 Deep Learning",
        code=code,
        extra_options=[
            OperationOption(name="user-requested-secret", type="secret", required=True),
            OperationOption(name="rn_token", type="secret"),
        ],
        options={"user-requested-secret": "robot-transformer-universal-token"},
        cache_options=BlockCacheOptions(mr_policy="reuse_always"),
    )
