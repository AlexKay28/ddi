"""Runnable V70 fixed-marker training, evaluation, and real-Qwen smoke.

The five request markers are fixed deterministic conditioning vectors. They are
not learned link tokens. Documents are always retained and tokenized as separate
segments; the stable schedule selects either monolithic execution or genuinely
independent, differentiable document K/V encoding.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

UTC = getattr(datetime, "UTC", datetime.timezone.utc)  # noqa: UP017

ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / ".v70_cache"
for variable, suffix in {
    "HF_HOME": "huggingface",
    "TRANSFORMERS_CACHE": "huggingface/transformers",
    "TORCH_HOME": "torch",
    "TMPDIR": "tmp",
    "TRITON_CACHE_DIR": "triton",
    "CUDA_CACHE_PATH": "cuda",
    "TORCH_EXTENSIONS_DIR": "torch_extensions",
}.items():
    os.environ.setdefault(variable, str(CACHE_ROOT / suffix))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from ddi.canonical_cache import (  # noqa: E402
    ExecutionContract,
    model_config_identity,
    runtime_identity,
)
from ddi.data import load as load_dataset  # noqa: E402
from ddi.finetune import _assert_finite_weights, _freeze_trunk  # noqa: E402
from ddi.link_token_contract import DocumentPayload  # noqa: E402
from ddi.prompt import build_query_text  # noqa: E402
from scripts.v70_harness import (  # noqa: E402
    EMBEDDING_POLICY,
    EncodedRequest,
    evaluate_abcd,
    independent_forward,
    install_fixed_link_tokens,
    monolithic_forward,
    require_adamw8bit,
    reset_rngs,
    teacher_answer_logits,
    training_loss,
)
from scripts.v70_schedule import RecordSchedule, stable_schedule  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
BASE_CHECKPOINT = ROOT / "results/checkpoints/Qwen__Qwen2.5-7B-Instruct__ddi_qf__8L_restart"
SEED = 47
TRAINABLE_LAYERS = 8
LEARNING_RATE = 1e-5
EFFECTIVE_BATCH_SIZE = 4
WARMUP_STEPS = 20
V70_METADATA = "v70_metadata.json"
PRIMARY_EVAL_SPLITS = ("dev", "confirmation100", "confirmation300")
EVAL_SPLITS = (*PRIMARY_EVAL_SPLITS, "train-smoke")
SOURCE_PATHS = (
    "scripts/v70_harness.py",
    "scripts/v70_schedule.py",
    "scripts/v70_run.py",
)


def _json_dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_id_list(record_ids: list[str], *, sorted_ids: bool) -> str:
    values = sorted(record_ids) if sorted_ids else record_ids
    return _sha256_bytes("\n".join(values).encode())


def _checkpoint_hashes(path: Path) -> dict[str, Any]:
    files = {
        str(file.relative_to(path)): _sha256_file(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }
    if not files:
        raise ValueError(f"checkpoint has no files: {path}")
    return {
        "sha256": _sha256_bytes(_json_dump(files).encode()),
        "files": files,
    }


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return _sha256_bytes(tensor.detach().float().cpu().numpy().tobytes())


def _log(path: Path | None, message: str) -> None:
    line = f"{datetime.datetime.now(UTC).isoformat()} {message}"
    print(f"[v70.run] {message}", file=sys.stderr, flush=True)
    if path is not None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def configure_cache_dirs() -> None:
    for variable in (
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "TORCH_HOME",
        "TMPDIR",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
        "TORCH_EXTENSIONS_DIR",
    ):
        Path(os.environ[variable]).mkdir(parents=True, exist_ok=True)


def encode_real_record(tokenizer: Any, record: dict[str, Any]) -> EncodedRequest:
    """Tokenize original document/query/answer segments without a text round trip."""
    documents = tuple(
        DocumentPayload(
            tuple(
                tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
            )
        )
        for text in record["info_contexts"]
    )
    if not documents:
        raise ValueError("V70 training requires at least one original document")
    query_ids = tuple(
        tokenizer(
            build_query_text(tokenizer, record),
            add_special_tokens=False,
            return_tensors="pt",
        )
        .input_ids[0]
        .tolist()
    )
    answer_ids = tuple(
        tokenizer(
            record["dialog"][-1]["content"],
            add_special_tokens=False,
            return_tensors="pt",
        )
        .input_ids[0]
        .tolist()
    )
    if not query_ids or not answer_ids:
        raise ValueError("V70 query and answer token segments must be non-empty")
    return EncodedRequest(documents, query_ids, answer_ids)


def _generation_request(request: EncodedRequest) -> EncodedRequest:
    return EncodedRequest(request.documents, request.query_token_ids)


def _load_common_records(seed: int, limit: int | None = None) -> list[dict[str, Any]]:
    manifest = json.loads((ROOT / "experiments/v62_split_manifest.json").read_text())
    ids = manifest["splits"]["train"]["record_ids"]
    by_id = {record["id"]: record for record in load_dataset("hotpot_qa", "train", seed)}
    records = [by_id[record_id] for record_id in ids if record_id in by_id]
    records = [
        record
        for record in records
        if record.get("possible_answers") and record.get("info_contexts")
    ]
    if limit is not None:
        records = records[:limit]
    if not records:
        raise RuntimeError("no V70 training records matched the fixed V62 train manifest")
    return records


def load_frozen_eval_ids(split: str, records: int | None = None) -> dict[str, Any]:
    """Load and validate one V70 evaluation ID list without reading record labels."""
    if split not in EVAL_SPLITS:
        raise ValueError(f"unknown V70 evaluation split: {split!r}")
    if split in PRIMARY_EVAL_SPLITS and records is not None:
        raise ValueError(f"--records is forbidden for primary split {split!r}")
    if split == "train-smoke" and (records is None or records < 1):
        raise ValueError("train-smoke requires --records >= 1")

    dev_manifest_path = ROOT / "experiments/v67_fresh_dev_split.json"
    confirmation_manifest_path = ROOT / "experiments/v70_confirmation300.json"
    v62_manifest_path = ROOT / "experiments/v62_split_manifest.json"
    dev_manifest = json.loads(dev_manifest_path.read_text())
    confirmation_manifest = json.loads(confirmation_manifest_path.read_text())
    v62_manifest = json.loads(v62_manifest_path.read_text())
    dev_ids = dev_manifest["development"]["record_ids"]
    confirmation_ids = confirmation_manifest["confirmation"]["record_ids"]
    original_confirmation_ids = v62_manifest["splits"]["confirmation"]["record_ids"]
    train_ids = v62_manifest["splits"]["train"]["record_ids"]

    expected = ((dev_ids, 100, "development"), (confirmation_ids, 300, "confirmation"))
    for ids, count, name in expected:
        if len(ids) != count or len(set(ids)) != count:
            raise ValueError(f"frozen {name} IDs must contain exactly {count} unique values")
    if len(original_confirmation_ids) != 100 or len(set(original_confirmation_ids)) != 100:
        raise ValueError("frozen original confirmation must contain exactly 100 unique IDs")
    if confirmation_ids[:100] != original_confirmation_ids:
        raise ValueError("confirmation100 is not the frozen prefix of confirmation300")
    if set(dev_ids) & set(confirmation_ids):
        raise ValueError("frozen development and confirmation IDs overlap")
    if set(train_ids) & (set(dev_ids) | set(confirmation_ids)):
        raise ValueError("frozen evaluation IDs overlap the V62 training IDs")
    if _sha256_id_list(dev_ids, sorted_ids=True) != dev_manifest["development"]["sha256"]:
        raise ValueError("frozen development ID hash mismatch")
    if (
        _sha256_id_list(confirmation_ids, sorted_ids=True)
        != confirmation_manifest["confirmation"]["sha256"]
    ):
        raise ValueError("frozen confirmation ID hash mismatch")

    if split == "dev":
        ids = dev_ids
        manifest_path = dev_manifest_path
        data_role = "development_selection"
    elif split == "confirmation100":
        ids = confirmation_ids[:100]
        manifest_path = confirmation_manifest_path
        data_role = "confirmation_primary_frozen_prefix"
    elif split == "confirmation300":
        ids = confirmation_ids
        manifest_path = confirmation_manifest_path
        data_role = "confirmation_primary"
    else:
        ids = train_ids[:records]
        manifest_path = v62_manifest_path
        data_role = "infrastructure_reproduction_only"

    return {
        "split": split,
        "data_role": data_role,
        "record_ids": list(ids),
        "count": len(ids),
        "id_list_sha256": _sha256_id_list(ids, sorted_ids=False),
        "id_set_sha256": _sha256_id_list(ids, sorted_ids=True),
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": _sha256_file(manifest_path),
    }


def load_frozen_eval_records(split_info: dict[str, Any]) -> list[dict[str, Any]]:
    """Select frozen records in manifest order and fail if the source lacks any ID."""
    stream = load_dataset("hotpot_qa", "train", 0)
    by_id: dict[str, dict[str, Any]] = {}
    for record in stream:
        record_id = record["id"]
        if record_id in by_id:
            raise ValueError(f"duplicate ID in hotpot_qa train stream: {record_id}")
        by_id[record_id] = record
    missing = [record_id for record_id in split_info["record_ids"] if record_id not in by_id]
    if missing:
        raise ValueError(f"hotpot_qa train stream is missing frozen IDs: {missing}")
    return [by_id[record_id] for record_id in split_info["record_ids"]]


def _load_model(checkpoint: Path) -> tuple[Any, Any]:
    from ddi import inference

    return inference.load(MODEL_ID, backend="eager", checkpoint_dir=str(checkpoint))


def _prepare_teacher(checkpoint: Path) -> tuple[Any, Any, Any]:
    teacher, tokenizer = _load_model(checkpoint)
    state = install_fixed_link_tokens(teacher, tokenizer)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher.to("cuda"), tokenizer, state


def _prepare_student(
    checkpoint: Path, seed: int
) -> tuple[Any, Any, Any, Any, list[torch.nn.Parameter]]:
    reset_rngs(seed)
    student, tokenizer = _load_model(checkpoint)
    state = install_fixed_link_tokens(student, tokenizer)
    _freeze_trunk(student, TRAINABLE_LAYERS)
    student = student.to("cuda")
    if student.dtype != torch.bfloat16:
        raise RuntimeError(f"V70 independent K/V training requires bf16, got {student.dtype}")
    student.train()
    parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = require_adamw8bit(parameters, lr=LEARNING_RATE, weight_decay=0.0)
    return student, tokenizer, state, optimizer, parameters


def _parameter_probe(model: Any) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        flat = parameter.detach().reshape(-1)
        indices = sorted({0, flat.numel() // 2, flat.numel() - 1})
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(flat[indices].float().cpu().numpy().tobytes())
    return digest.hexdigest()


def _gradient_metrics(parameters: list[torch.nn.Parameter]) -> dict[str, Any]:
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    finite = bool(gradients) and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    nonzero = sum(int(torch.count_nonzero(gradient).item()) for gradient in gradients)
    norm = (
        torch.linalg.vector_norm(
            torch.stack(
                [torch.linalg.vector_norm(gradient.detach().float()) for gradient in gradients]
            )
        ).item()
        if gradients
        else 0.0
    )
    return {
        "tensor_count": len(gradients),
        "finite": finite,
        "nonzero_elements": nonzero,
        "l2_norm": norm,
    }


@contextmanager
def _forward_probe(model: Any):
    calls: list[dict[str, Any]] = []
    original = model.forward

    def observed(*args: Any, **kwargs: Any) -> Any:
        calls.append(
            {
                "input_shape": list(kwargs["input_ids"].shape),
                "attention_mask_shape": list(kwargs["attention_mask"].shape),
                "position_ids_shape": list(kwargs["position_ids"].shape),
                "position_min": int(kwargs["position_ids"].min().item()),
                "position_max": int(kwargs["position_ids"].max().item()),
                "has_past_key_values": kwargs.get("past_key_values") is not None,
            }
        )
        return original(*args, **kwargs)

    model.forward = observed
    try:
        yield calls
    finally:
        model.forward = original


def _scheduled_forward(
    model: Any,
    contract: Any,
    request: EncodedRequest,
    k: int,
    schedule: RecordSchedule,
) -> Any:
    if schedule.encoding == "monolithic":
        return monolithic_forward(model, contract, request, k=k)
    return independent_forward(model, contract, request, k=k)


def learning_rate_for_step(step: int) -> float:
    """Return the preregistered LR for a one-based optimizer step."""
    if step < 1:
        raise ValueError("optimizer step must be one-based")
    return LEARNING_RATE * min(step / WARMUP_STEPS, 1.0)


def training_plan(
    records: list[dict[str, Any]], steps: int, seed: int
) -> list[list[tuple[dict[str, Any], RecordSchedule]]]:
    """Build the fixed, arm-independent record and encoding schedule."""
    if not records or steps < 1:
        raise ValueError("training plan requires records and at least one optimizer step")
    plan = []
    for optimizer_step in range(steps):
        presentations = []
        for batch_index in range(EFFECTIVE_BATCH_SIZE):
            presentation_index = optimizer_step * EFFECTIVE_BATCH_SIZE + batch_index
            record = records[presentation_index % len(records)]
            presentations.append(
                (record, stable_schedule(record["id"], presentation_index, seed))
            )
        plan.append(presentations)
    return plan


def save_v70_checkpoint(
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    *,
    arm: str,
    step: int,
    seed: int,
) -> None:
    if output_dir.name != arm and not output_dir.name.endswith(f"_{arm}"):
        raise ValueError("V70 checkpoint path must be arm-specific (end in the arm name)")
    output_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metadata = {
        "schema": "v70-checkpoint-v1",
        "arm": arm,
        "k": int(arm[1:]),
        "step": step,
        "seed": seed,
        "fixed_marker_conditioning": True,
        "learned_link_tokens": False,
        "embedding_policy": EMBEDDING_POLICY,
        "schedule": "sha256-v70",
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "warmup_optimizer_steps": WARMUP_STEPS,
    }
    (output_dir / V70_METADATA).write_text(_json_dump(metadata) + "\n")


def load_v70_checkpoint(
    path: Path, expected_arm: str | None = None
) -> tuple[Any, Any, Any, dict[str, Any]]:
    metadata = json.loads((path / V70_METADATA).read_text())
    if metadata.get("schema") != "v70-checkpoint-v1" or not metadata.get(
        "fixed_marker_conditioning"
    ):
        raise ValueError("not a V70 fixed-marker checkpoint")
    if expected_arm is not None and metadata.get("arm") != expected_arm:
        raise ValueError(f"checkpoint arm {metadata.get('arm')!r} != expected {expected_arm!r}")
    model, tokenizer = _load_model(path)
    state = install_fixed_link_tokens(model, tokenizer)
    return model, tokenizer, state, metadata


def run_evaluation(
    model: Any,
    tokenizer: Any,
    token_state: Any,
    records: list[tuple[str, EncodedRequest, list[str]]],
    cache_dir: Path,
    *,
    weights_identity: str,
    k: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    contract = ExecutionContract(
        weights_identity=weights_identity,
        adapters_identity="none",
        tokenizer_identity=_sha256_file(Path(tokenizer.name_or_path) / "tokenizer.json"),
        model_config_identity=model_config_identity(model),
        runtime_identity=runtime_identity(model),
        backend="eager",
        dtype=str(model.dtype),
    )
    return evaluate_abcd(
        model,
        tokenizer,
        token_state,
        records,
        contract,
        cache_dir,
        k=k,
        max_new_tokens=max_new_tokens,
    )


def build_evaluation_output(
    result: dict[str, Any],
    *,
    checkpoint: Path,
    checkpoint_hashes: dict[str, Any],
    split_info: dict[str, Any],
    k: int,
    max_new_tokens: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Attach sufficient immutable provenance to raw A/B/C/D evaluation output."""
    config_path = checkpoint / "config.json"
    tokenizer_path = checkpoint / "tokenizer.json"
    return {
        **result,
        "schema": "v70-evaluation-v2",
        "timestamp_utc": datetime.datetime.now(UTC).isoformat(),
        "split": split_info["split"],
        "data_role": split_info["data_role"],
        "record_count": split_info["count"],
        "k": k,
        "max_new_tokens": max_new_tokens,
        "elapsed_seconds": elapsed_seconds,
        "provenance": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hashes["sha256"],
            "checkpoint_file_sha256": checkpoint_hashes["files"],
            "config_sha256": _sha256_file(config_path),
            "tokenizer_sha256": _sha256_file(tokenizer_path),
            "id_list_sha256": split_info["id_list_sha256"],
            "id_set_sha256": split_info["id_set_sha256"],
            "record_ids": split_info["record_ids"],
            "split_manifest": split_info["manifest"],
            "split_manifest_sha256": split_info["manifest_sha256"],
            "dataset_source": "ddi.data.load('hotpot_qa', 'train', 0)",
            "source_sha256": {
                path: _sha256_file(ROOT / path)
                for path in SOURCE_PATHS
            },
        },
    }


def evaluation_cache_dir(
    output: Path, checkpoint: Path, split: str, checkpoint_sha256: str
) -> Path:
    """Return a cache namespace that cannot alias another checkpoint or split."""
    return output.parent / (
        f".{output.stem}.cache.{checkpoint.name}.{split}.{checkpoint_sha256[:16]}"
    )


def _gpu_idle_snapshot() -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("V70 smoke requires CUDA_VISIBLE_DEVICES=0")
    processes = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    state = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-gpu=name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if processes:
        raise RuntimeError(f"GPU 0 is busy; refusing model load: {processes}")
    return {"compute_processes": [], "state": state}


def _smoke_arm(
    teacher: Any,
    request: EncodedRequest,
    record_id: str,
    checkpoint: Path,
    *,
    k: int,
    seed: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    arm = f"k{k}"
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    student, _, state, optimizer, parameters = _prepare_student(checkpoint, seed)
    schedule = stable_schedule(record_id, 0, seed, independent_rate=1.0)
    initial_probe = _parameter_probe(student)

    student.eval()
    reset_rngs(seed)
    with torch.no_grad():
        with _forward_probe(student) as calls:
            first_result = _scheduled_forward(student, state.contract, request, k, schedule)
        first = first_result.answer_logits.clone()
        layout = first_result.layout
        del first_result
        reset_rngs(seed)
        repeat = _scheduled_forward(
            student, state.contract, request, k, schedule
        ).answer_logits.clone()
        deterministic = torch.equal(first, repeat)
        monolithic = monolithic_forward(student, state.contract, request, k=k).answer_logits
        independent = independent_forward(student, state.contract, request, k=k).answer_logits
        encoding_delta = float((monolithic.float() - independent.float()).abs().max().item())
    del repeat, monolithic, independent

    student.train()
    reset_rngs(seed)
    optimizer.zero_grad(set_to_none=True)
    loss = training_loss(
        teacher,
        student,
        state.contract,
        request,
        k=k,
        encoding=schedule.encoding,
    )
    has_grad_fn = loss.grad_fn is not None
    loss.backward()
    gradients = _gradient_metrics(parameters)
    if not gradients["finite"] or gradients["nonzero_elements"] == 0:
        raise RuntimeError(f"{arm}: no finite nonzero decoder gradient")
    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    _assert_finite_weights(student, f"V70 smoke {arm}")

    link_ids = set(state.contract.link_token_ids)
    link_count = sum(token in link_ids for token in layout.physical_token_ids)
    result = {
        "arm": arm,
        "k": k,
        "schedule": asdict(schedule),
        "initial_trainable_probe_sha256": initial_probe,
        "input_length": len(layout.physical_token_ids),
        "link_count": link_count,
        "link_ids": list(state.contract.link_token_ids),
        "actual_link_ids": [token for token in layout.physical_token_ids if token in link_ids],
        "forward_calls": calls,
        "k0_rerun_deterministic": deterministic if k == 0 else None,
        "independent_monolithic_answer_logit_max_abs_delta": encoding_delta,
        "loss": float(loss.detach().item()),
        "components": {"teacher_kl_loss": float(loss.detach().item())},
        "loss_has_grad_fn": has_grad_fn,
        "gradients": gradients,
        "optimizer_class": f"{type(optimizer).__module__}.{type(optimizer).__name__}",
        "finite_loss": bool(torch.isfinite(loss)),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    logits = first.detach().float().cpu()
    del first, loss, optimizer, student
    torch.cuda.empty_cache()
    return result, logits


def run_smoke(output_parent: Path, checkpoint: Path = BASE_CHECKPOINT, seed: int = SEED) -> Path:
    configure_cache_dirs()
    stamp = datetime.datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_parent / f"v70_harness_smoke_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "smoke.log"
    started = time.perf_counter()
    try:
        gpu_before = _gpu_idle_snapshot()
        _log(log_path, f"nvidia-smi immediately before model load: {gpu_before['state']}")
        teacher, tokenizer, _teacher_state = _prepare_teacher(checkpoint)
        record = _load_common_records(seed, limit=1)[0]
        request = encode_real_record(tokenizer, record)
        teacher_target = teacher_answer_logits(teacher, request)
        teacher_target_repeat = teacher_answer_logits(teacher, request)
        teacher_deterministic = torch.equal(teacher_target, teacher_target_repeat)
        teacher_sha = _tensor_sha256(teacher_target)
        del teacher_target, teacher_target_repeat
        _log(log_path, f"record={record['id']} documents={len(request.documents)}")

        arms: dict[str, Any] = {}
        logits: dict[str, torch.Tensor] = {}
        for k in (0, 5):
            arms[f"k{k}"], logits[f"k{k}"] = _smoke_arm(
                teacher, request, record["id"], checkpoint, k=k, seed=seed
            )
            _log(log_path, f"completed k{k}: loss={arms[f'k{k}']['loss']:.8g}")

        n_documents = len(request.documents)
        arm_delta = float((logits["k0"] - logits["k5"]).abs().max().item())
        assertions = {
            "document_count_positive": n_documents > 0,
            "length_delta_is_5N": arms["k5"]["input_length"] - arms["k0"]["input_length"]
            == 5 * n_documents,
            "links_only_in_k5": arms["k0"]["link_count"] == 0
            and arms["k5"]["link_count"] == 5 * n_documents,
            "masks_reached_model": all(
                call["attention_mask_shape"]
                for arm in arms.values()
                for call in arm["forward_calls"]
            ),
            "positions_reached_model": all(
                call["position_ids_shape"] for arm in arms.values() for call in arm["forward_calls"]
            ),
            "k0_rerun_deterministic": arms["k0"]["k0_rerun_deterministic"] is True,
            "k0_k5_answer_logits_differ": arm_delta > 0.0,
            "independent_monolithic_delta_nonzero": all(
                arm["independent_monolithic_answer_logit_max_abs_delta"] > 0.0
                for arm in arms.values()
            ),
            "both_losses_have_grad_fn": all(arm["loss_has_grad_fn"] for arm in arms.values()),
            "finite_nonzero_decoder_gradients": all(
                arm["gradients"]["finite"] and arm["gradients"]["nonzero_elements"] > 0
                for arm in arms.values()
            ),
            "adamw8bit_constructed": all(
                arm["optimizer_class"].endswith("AdamW8bit") for arm in arms.values()
            ),
            "no_nan": all(
                arm["finite_loss"] and arm["gradients"]["finite"] for arm in arms.values()
            ),
            "identical_initial_student_probe": arms["k0"]["initial_trainable_probe_sha256"]
            == arms["k5"]["initial_trainable_probe_sha256"],
            "teacher_target_deterministic": teacher_deterministic,
            "teacher_student_vocab_match": teacher.config.vocab_size
            == len(tokenizer)
            == max(arms["k5"]["link_ids"]) + 1,
        }
        failed = [name for name, passed in assertions.items() if not passed]
        result = {
            "schema": "v70-real-qwen-connectivity-smoke-v1",
            "timestamp_utc": datetime.datetime.now(UTC).isoformat(),
            "model_id": MODEL_ID,
            "checkpoint": str(checkpoint),
            "record_id": record["id"],
            "n_documents": n_documents,
            "document_lengths": [len(document.token_ids) for document in request.documents],
            "query_length": len(request.query_token_ids),
            "answer_length": len(request.answer_token_ids),
            "fixed_marker_conditioning": True,
            "learned_link_tokens": False,
            "embedding_policy": EMBEDDING_POLICY,
            "teacher_target_sha256": teacher_sha,
            "teacher_target_identical_across_arms": teacher_deterministic,
            "k0_k5_answer_logit_max_abs_delta": arm_delta,
            "arms": arms,
            "assertions": assertions,
            "all_assertions_passed": not failed,
            "failed_assertions": failed,
            "peak_gpu_memory_bytes": max(arm["peak_gpu_memory_bytes"] for arm in arms.values()),
            "elapsed_seconds": time.perf_counter() - started,
        }
        with (output_dir / "result.jsonl").open("x", encoding="utf-8") as stream:
            stream.write(_json_dump(result) + "\n")
        manifest = {
            "schema": "v70-smoke-manifest-v1",
            "command": "CUDA_VISIBLE_DEVICES=0 python scripts/v70_run.py smoke",
            "python": sys.executable,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_before": gpu_before,
            "cache_environment": {
                key: os.environ[key]
                for key in (
                    "HF_HOME",
                    "TRANSFORMERS_CACHE",
                    "TORCH_HOME",
                    "TMPDIR",
                    "TRITON_CACHE_DIR",
                    "CUDA_CACHE_PATH",
                    "TORCH_EXTENSIONS_DIR",
                )
            },
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_config_sha256": _sha256_file(checkpoint / "config.json"),
            "no_model_checkpoint_copy": True,
        }
        with (output_dir / "manifest.json").open("x", encoding="utf-8") as stream:
            stream.write(_json_dump(manifest) + "\n")
        _log(log_path, f"assertions passed={not failed}; failed={failed}")
        sha_paths = [
            ROOT / "scripts/v70_harness.py",
            ROOT / "scripts/v70_schedule.py",
            ROOT / "scripts/v70_run.py",
            ROOT / "tests/test_v70_harness.py",
            output_dir / "result.jsonl",
            output_dir / "manifest.json",
            log_path,
        ]
        with (output_dir / "SHA256SUMS").open("x", encoding="utf-8") as stream:
            for path in sha_paths:
                stream.write(f"{_sha256_file(path)}  {path.relative_to(ROOT)}\n")
        if failed:
            raise RuntimeError(f"V70 smoke assertions failed: {failed}")
        return output_dir
    except Exception as error:
        _log(log_path, f"FAILED: {type(error).__name__}: {error}")
        raise


def train_arm(arm: str, steps: int, checkpoint_dir: Path, seed: int) -> None:
    if arm not in {"k0", "k5"} or steps < 1:
        raise ValueError("V70 training requires arm k0/k5 and steps >= 1")
    configure_cache_dirs()
    gpu_before = _gpu_idle_snapshot()
    print(f"nvidia-smi immediately before model load: {gpu_before['state']}", file=sys.stderr)
    teacher, tokenizer, _teacher_state = _prepare_teacher(BASE_CHECKPOINT)
    student, student_tokenizer, student_state, optimizer, parameters = _prepare_student(
        BASE_CHECKPOINT, seed
    )
    if teacher.config.vocab_size != student.config.vocab_size:
        raise RuntimeError("teacher/student KL vocabulary dimensions differ")
    records = _load_common_records(seed)
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    losses_path = checkpoint_dir.parent / f"{checkpoint_dir.name}.losses.jsonl"
    if losses_path.exists():
        raise FileExistsError(losses_path)
    k = int(arm[1:])
    for step, presentations in enumerate(training_plan(records, steps, seed), start=1):
        step_started = time.perf_counter()
        lr = learning_rate_for_step(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        presentation_losses = []
        record_ids = []
        schedules = []
        for record, schedule in presentations:
            request = encode_real_record(tokenizer, record)
            reset_rngs(int(schedule.shuffle_key[:16], 16) % 2**32)
            loss = training_loss(
                teacher,
                student,
                student_state.contract,
                request,
                k=k,
                encoding=schedule.encoding,
            )
            presentation_losses.append(float(loss.detach().item()))
            record_ids.append(record["id"])
            schedules.append(asdict(schedule))
            (loss / EFFECTIVE_BATCH_SIZE).backward()
        gradients = _gradient_metrics(parameters)
        if not gradients["finite"] or gradients["nonzero_elements"] == 0:
            raise RuntimeError(f"step {step}: invalid gradients")
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        row = {
            "step": step,
            "arm": arm,
            "k": k,
            "record_ids": record_ids,
            "schedules": schedules,
            "presentation_losses": presentation_losses,
            "mean_loss": sum(presentation_losses) / EFFECTIVE_BATCH_SIZE,
            "gradients": gradients,
            "lr": lr,
            "elapsed_seconds": time.perf_counter() - step_started,
        }
        with losses_path.open("a", encoding="utf-8") as stream:
            stream.write(_json_dump(row) + "\n")
    _assert_finite_weights(student, f"V70 {arm} before checkpoint")
    save_v70_checkpoint(student, student_tokenizer, checkpoint_dir, arm=arm, step=steps, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke", help="one-step real-Qwen k0/k5 connectivity smoke")
    smoke.add_argument("--output-parent", type=Path, default=ROOT / "results")
    smoke.add_argument("--checkpoint", type=Path, default=BASE_CHECKPOINT)
    smoke.add_argument("--seed", type=int, default=SEED)
    train = commands.add_parser("train", help="run a V70 k0 or k5 training arm")
    train.add_argument("--arm", choices=("k0", "k5"), required=True)
    train.add_argument("--steps", type=int, required=True, help="number of optimizer steps")
    train.add_argument("--checkpoint-dir", type=Path, required=True)
    train.add_argument("--seed", type=int, default=SEED)
    evaluate = commands.add_parser("evaluate", help="run A/B/C/D through evaluate_abcd")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--arm", choices=("k0", "k5"), required=True)
    evaluate.add_argument("--split", choices=EVAL_SPLITS, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument(
        "--records", type=int, help="record limit; required and valid only for train-smoke"
    )
    evaluate.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.command == "smoke":
        print(run_smoke(args.output_parent, args.checkpoint, args.seed))
    elif args.command == "train":
        train_arm(args.arm, args.steps, args.checkpoint_dir, args.seed)
    else:
        split_info = load_frozen_eval_ids(args.split, args.records)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite evaluation output: {args.output}")
        checkpoint_hashes = _checkpoint_hashes(args.checkpoint)
        cache_dir = evaluation_cache_dir(
            args.output, args.checkpoint, args.split, checkpoint_hashes["sha256"]
        )
        if cache_dir.exists():
            raise FileExistsError(f"refusing to reuse evaluation cache: {cache_dir}")
        configure_cache_dirs()
        gpu_before = _gpu_idle_snapshot()
        print(f"nvidia-smi immediately before model load: {gpu_before['state']}", file=sys.stderr)
        model, tokenizer, state, metadata = load_v70_checkpoint(args.checkpoint, args.arm)
        model = model.to("cuda").eval()
        records = load_frozen_eval_records(split_info)
        encoded = [
            (
                record["id"],
                _generation_request(encode_real_record(tokenizer, record)),
                record["possible_answers"],
            )
            for record in records
        ]
        started = time.perf_counter()
        result = run_evaluation(
            model,
            tokenizer,
            state,
            encoded,
            cache_dir,
            weights_identity=checkpoint_hashes["sha256"],
            k=metadata["k"],
            max_new_tokens=args.max_new_tokens,
        )
        output = build_evaluation_output(
            result,
            checkpoint=args.checkpoint,
            checkpoint_hashes=checkpoint_hashes,
            split_info=split_info,
            k=metadata["k"],
            max_new_tokens=args.max_new_tokens,
            elapsed_seconds=time.perf_counter() - started,
        )
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(_json_dump(output) + "\n")


if __name__ == "__main__":
    main()
