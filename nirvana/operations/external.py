"""External Nirvana operations used by the DDI graphs.

GENERATED, do not hand-edit. Produced by:

    ./bin/bin import --full --import-private-options \\
        "https://nirvana.yandex-team.ru/flow/e0f1a670-0478-4b25-a176-0fb4b26d2a29/020c68e3-e3ab-4e17-90c8-b810e089c429/graph"

and lifting the operation declarations out of the result.

`--import-private-options` is required, not optional. Without it the PyDL wrapper
comes out with 27 keyword arguments instead of 93, missing every private option the
live blocks actually set (ports, max-disk, the NCCL flags, the YT scheduler settings,
underdeep-token...). Those cannot be worked around by declaring them in
`block_args(extra_options=...)`: vh3 refuses options it does not know, and Nirvana
refuses to "override option metadata that is already defined in operation". The only
way through is a wrapper that exposes them.

One exception to "generated": `python_run` ("Python Run", labelled "Make DataFrame" on
the hand-built graph) was written by hand in the same shape the importer produces —
only the options the graphs actually set are exposed.
"""

from typing import Literal, NamedTuple, Optional, Sequence

import vh3


@vh3.decorator.external_operation(
    "https://nirvana.yandex-team.ru/operation/e8f3de88-f98c-4511-8b49-f1145c99a2dc/overview"
)
def zeliboba_arc_checkout_deterministic(
    *,
    arc_token: vh3.Secret,
    pulsar_token: vh3.Secret,
    yt_token: vh3.Secret,
    yt_cluster: vh3.Enum[
        Literal[
            "hahn",
            "freud",
            "marx",
            "hume",
            "arnold",
            "markov",
            "bohr",
            "landau",
            "seneca-vla",
            "seneca-sas",
            "seneca-man",
            "watt",
            "kolmogorov",
        ]
    ],
    use_baked_zdl: vh3.Boolean = False,
    arc_reference: Optional[vh3.String] = None,
    sandbox_token: Optional[vh3.Secret] = None,
    sandbox_owner: Optional[vh3.String] = None,
    yt_pool: Optional[vh3.String] = None,
    mr_account: Optional[vh3.String] = None,
) -> vh3.Binary:
    raise NotImplementedError()


@vh3.decorator.external_operation(
    "https://nirvana.yandex-team.ru/operation/9ff37ad1-f024-4079-9e90-a7b62743449e/overview", vh3.job_run_base
)
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_output_names("code")
def arc_export(
    *,
    arc_token: vh3.Secret,
    path: vh3.String,
    job_command: vh3.String = vh3.dedent("""\
    bash -c '
    if [ ${param.reference} = "trunk" ]; then
        >&2 echo "Using Reference=trunk in a deterministic operation. Most likely this is not what you want"
        exit 1
    fi
    mkdir ~/.arc
    chmod 700 ~/.arc
    echo "${param["arc-token"]}" >> ~/.arc/token
    chmod 400 ~/.arc/token
    mkdir bare_repo
    cd bare_repo
    ${resource.arc} init --bare || exit 1;
    ${resource.arc} fetch ${param.reference};
    ${resource.arc} export ${param.reference} ${param.path} --to ../export || exit 1;
    cd ../export/${param.path}
    tar -cf ${output.code} * || exit 1;
    '
    """),
    job_environments: vh3.MultipleStrings = (),
    job_variables: vh3.MultipleStrings = (),
    ports: vh3.MultipleStrings = (),
    reference: vh3.String = "trunk",
    job_layer: vh3.MultipleStrings = ("4d3ced44-39c3-48a6-9dbe-1f9f852ad081",),
) -> vh3.Binary:
    raise NotImplementedError()


@vh3.decorator.external_operation(
    "https://nirvana.yandex-team.ru/operation/7941af94-7c81-4ab5-8458-423c732ee6c8/overview", vh3.job_run_base
)
@vh3.decorator.nirvana_output_names("tar_archive")
def merge_tar_archives(
    *,
    tar_archives: Sequence[vh3.Binary],
    job_command: vh3.String = 'bash -c \'mkdir tmp; cd tmp; [#list input["tar_archives"] as a] tar xvf ${a}; [/#list] tar cvf ${output["tar_archive"]} *\'',
    cpu_guarantee: vh3.Integer = 100,
    job_is_vanilla: vh3.Boolean = True,
    job_layer: vh3.MultipleStrings = ("4d3ced44-39c3-48a6-9dbe-1f9f852ad081",),
) -> vh3.Binary:
    raise NotImplementedError()


class Python3DeepLearningOutputs(NamedTuple):
    data: vh3.Binary
    state: vh3.Binary
    logs: vh3.Binary
    json_output: vh3.JSON
    pulsar_link: vh3.OptionalOutput[vh3.HTML]
    pulsar_info: vh3.OptionalOutput[vh3.JSON]
    telemetry_link: vh3.OptionalOutput[vh3.HTML]


@vh3.decorator.external_operation(
    "https://nirvana.yandex-team.ru/operation/b5442913-8ff3-46fd-ac69-ad2c57d4a078/overview"
)
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
def python_3_deep_learning(
    *,
    pulsar_token: vh3.Secret,
    script: Sequence[vh3.Binary] = (),
    data: Sequence[vh3.Binary] = (),
    data_files: Sequence[vh3.Binary] = (),
    state: Optional[vh3.Binary] = None,
    params: Optional[vh3.JSON] = None,
    pulsar_params: Optional[vh3.JSON] = None,
    image: vh3.MultipleEnums[
        Literal["pydl/nvidia-pytorch:24.10.0", "pydl/nvidia-tensorflow:24.10.0", "pydl/python:3.12.11.0"]
    ] = ("pydl/python:3.12.11.0",),
    enable_legacy_docker_to_layers_conversion: vh3.Boolean = True,
    pool_tree: vh3.MultipleEnums[
        Literal[
            "gpu_beijing_180g",
            "gpu_hutang_141g",
            "gpu_hainan_80g",
            "gpu_a800_80g",
            "gpu_anshan_80g",
            "gpu_tesla_a100_80g",
            "gpu_tesla_a100",
            "gpu_tesla_v100",
            "physical",
        ]
    ] = ("physical",),
    target_platform: Optional[vh3.String] = None,
    nodes_count: vh3.Integer = 1,
    gpu_count: vh3.Integer = 0,
    pip: vh3.MultipleStrings = (),
    run_command: vh3.String = "python -c \"print('Hello, World!')\"",
    openmpi_runner: vh3.Boolean = False,
    openmpi_params: Optional[vh3.String] = None,
    mpi_emulate_torchrun_env: vh3.Boolean = False,
    mpi_processes_count: vh3.Integer = -1,
    pulsar: vh3.Boolean = False,
    job_variables: vh3.MultipleStrings = (),
    ttl: vh3.Integer = 1440,
    timestamp: vh3.Date = "2017-08-25T16:40:26+0300",
    python_version: vh3.String = "3",
    install_pydl_package: vh3.Boolean = True,
    auto_snapshot: vh3.Integer = 0,
    environment: vh3.MultipleStrings = (),
    hosts_to_env: vh3.MultipleEnums[Literal["master", "slave"]] = (),
    enable_interruption_file: vh3.Boolean = False,
    extra_node_run_command: vh3.MultipleStrings = (),
    reserved_job_count: vh3.Integer = 0,
    retries_on_job_failure: vh3.Integer = 0,
    allow_infra_retries: vh3.Integer = 1,
    disable_infra_retries: vh3.Boolean = False,
    job_scheduler_yt_pool: Optional[vh3.String] = None,
    mr_account: Optional[vh3.String] = None,
    gpu_max_ram: Optional[vh3.Integer] = None,
    cpu_cores_usage: Optional[vh3.Integer] = None,
    max_ram: Optional[vh3.Integer] = None,
    max_disk: Optional[vh3.Integer] = None,
    gpu_type: Optional[
        vh3.Enum[
            Literal[
                "NONE",
                "ANY",
                "CUDA_ANY",
                "CUDA_3_5",
                "CUDA_5_2",
                "CUDA_6_1",
                "CUDA_7_0",
                "CUDA_8_0",
                "CUDA_9_0",
                "CUDA_10_0",
            ]
        ]
    ] = None,
    strict_gpu_type: vh3.Boolean = False,
    yt_disk_medium: Optional[vh3.String] = None,
    yt_disk_account: Optional[vh3.String] = None,
    force_tmpfs_disk: vh3.Boolean = False,
    max_tmpfs_disk: Optional[vh3.Integer] = None,
    job_host_tags: vh3.MultipleStrings = (),
    job_scheduler_instance: Optional[vh3.String] = None,
    job_scheduler_yt_custom_spec: vh3.String = "{}",
    job_scheduler_yt_custom_task_spec: vh3.String = "{}",
    network: Optional[vh3.Enum[Literal["none"]]] = None,
    enable_mlproxy: vh3.Boolean = False,
    mlproxy_config: Optional[vh3.String] = None,
    mlproxy_config_format: vh3.Enum[Literal["toml", "yaml", "json"]] = "toml",
    mlproxy_custom_version: Optional[vh3.String] = None,
    mlproxy_with_sync: vh3.Boolean = True,
    unn_logging_config_yt_path: vh3.String = "//home/yt-ml/unitednn/pydl_loggers_config.yaml",
    debug_timeout: vh3.Integer = 0,
    nccl_debug_file_enable: vh3.Boolean = False,
    ucx_debug_file_enable: vh3.Boolean = False,
    ucx_debug_log_level: vh3.Enum[
        Literal["FATAL", "ERROR", "WARN", "INFO", "DEBUG", "TRACE", "REQ", "DATA", "ASYNC", "FUNC", "POLL"]
    ] = "DEBUG",
    yt_debug_file_enable: vh3.Boolean = True,
    job_core_yt_dir_path: vh3.String = "//home/pydl/core_dumps/operation",
    job_core_yt_table_ttl: vh3.Integer = 336,
    collect_telemetry: vh3.Boolean = True,
    disable_nccl_telemetry: vh3.Boolean = False,
    inner_user_params: vh3.String = "{}",
    mpi_debug_logs_enable: vh3.Boolean = True,
    split_mpi_logs: vh3.Boolean = True,
    base_mpi_mca_debug_verbose_level: vh3.Integer = 100,
    extra_mpi_mca_debug_params: vh3.String = "        --debug-daemons         -mca btl_base_verbose 100         -mca pmix_base_verbose 0         -mca ess_base_verbose 0         -mca oob_base_verbose 100         -mca plm_base_verbose 100         -mca rml_base_verbose 100         -mca orte_base_verbose 100         -mca ras_base_verbose 0",
    resource_affinity_binding_enable: vh3.Boolean = False,
    per_rank_resource_dump_enable: vh3.Boolean = False,
    cpu_affinity_binding_type: vh3.Enum[Literal["None", "NUMA Local", "NUMA Split"]] = "None",
    memory_affinity_binding_type: vh3.Enum[Literal["None", "NUMA Local"]] = "None",
    rdma_affinity_binding_type: vh3.Enum[Literal["None", "NUMA Local", "NUMA Split"]] = "None",
    underdeep_token: Optional[vh3.Secret] = None,
    underdeep_run_params: Optional[vh3.String] = None,
    ports: vh3.MultipleStrings = (
        "sshd_port = any tcp both,nocheck",
        "oob_port = any tcp both,nocheck",
        "btl_port = any tcp both,nocheck",
        "unitednn_distributed_run_handshake_barrier_port = any tcp master,nocheck",
        "unitednn_distributed_run_main_barrier_port = any tcp master,nocheck",
        "torch_rdzv_port = 29500 tcp master,nocheck",
    ),
    custom_infiniband_sl_setting_enable: vh3.Boolean = True,
    custom_infiniband_sl_value: vh3.Integer = 1,
    sshd_force_enable: vh3.Boolean = False,
    before_run_command: Optional[vh3.String] = None,
    after_run_command: Optional[vh3.String] = None,
    slave_run_command: Optional[vh3.String] = None,
    propagate_run_command_exit_code: vh3.Boolean = False,
    host_names_mapping_collection_enable: vh3.Boolean = False,
    preferred_layer_format: Optional[vh3.String] = None,
    allow_partial_node_multihost: vh3.Boolean = False,
    enable_yt_job_rpc_proxy: vh3.Boolean = False,
    force_disable_yt_job_rpc_proxy: vh3.Boolean = False,
    yt_job_rpc_proxy_threads: Optional[vh3.Integer] = None,
    macro_command: vh3.String = "./bin macro-context-run ml.nirvana.python_deep_learning.operations.macro_operation.python_3_deep_learning",
) -> Python3DeepLearningOutputs:
    raise NotImplementedError()


@vh3.decorator.external_operation(
    "https://nirvana.yandex-team.ru/operation/b8c37185-25df-41fe-806b-905a7d2fe14f/overview", vh3.job_run_base
)
@vh3.decorator.nirvana_names_transformer(vh3.name_transformers.snake_to_dash, options=True)
@vh3.decorator.nirvana_output_names("out")
def python_run(
    *,
    code: vh3.String,
    inp: Optional[vh3.JSON] = None,
    interpreter: vh3.String = "python3",
    requirements: vh3.MultipleStrings = (),
    max_ram: vh3.Integer = 100,
    job_layer: vh3.MultipleStrings = ("cccebdf5-79f5-4db3-8e04-db6cd6975581",),
) -> vh3.JSON:
    raise NotImplementedError()


@vh3.decorator.external_operation(
    "https://nirvana.yandex-team.ru/operation/6113e78d-4f1a-4454-a05e-47e24db0b30c/overview", vh3.job_run_base
)
@vh3.decorator.nirvana_names(oauth_token="oauthToken")
def add_comment_to_ticket(
    *,
    oauth_token: vh3.Secret,
    field: vh3.String,
    field_comment: vh3.String,
    tickets: vh3.JSON,
    job_command: vh3.String = 'python3 commet_to_ticket.py ${input["tickets"]} ${param["oauthToken"]} ${param["field"]} ${param["field_comment"]} ',
    job_layer: vh3.MultipleStrings = ("255602b3-9019-4ca5-ade0-0ab938fbfdbd",),
):
    raise NotImplementedError()

