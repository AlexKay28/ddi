"""Frozen TODO5 bf16 reduced-precision-reduction ablation; diagnostic only.

The two precision arms execute in fresh subprocesses.  It intentionally tests
only PyTorch's ``allow_bf16_reduced_precision_reduction`` switch: it is a
hypothesis about bf16 GEMM accumulation, not an established explanation for
the V60 pre-assembly projection divergence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import DynamicCache

ROOT = Path(__file__).resolve().parents[1]
V58_INPUT = ROOT / "results/v58_diag_bf16_20260908.json"
GIB = 1024**3
LIMITS = {"gpus": 1, "seconds": 600, "gpu_gib": 45, "artifacts_gib": 2, "min_free_disk_gib": 20}
UTC = timezone.utc  # noqa: UP017  # Python 3.10 test environment lacks datetime.UTC.


def runtime_dependencies() -> None:
    """Defer the full transformers model surface so CPU flag-helper tests need no model stack."""
    global CacheEntry, DocCache, assemble_dynamic_cache, forward_query_with_doc_cache
    global build_mask, to_additive, build_positions, load
    if "CacheEntry" in globals():
        return
    from ddi.cache import CacheEntry, DocCache, assemble_dynamic_cache, forward_query_with_doc_cache
    from ddi.inference import load
    from ddi.masks import build_mask, to_additive
    from ddi.positions import build_positions


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def comparison(a: torch.Tensor, b: torch.Tensor, *, logits: bool = False) -> dict:
    a, b = a.detach().cpu(), b.detach().cpu()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    value = {
        "shape": list(a.shape),
        "finite": finite,
        "equal": torch.equal(a, b),
        "max_abs": float((a.float() - b.float()).abs().max()) if finite else None,
        "allclose_1e4": torch.allclose(a.float(), b.float(), atol=1e-4, rtol=1e-4),
    }
    if logits:
        value["argmax_equal"] = torch.equal(a.argmax(-1), b.argmax(-1))
        value["a_argmax"] = a.argmax(-1).tolist()
        value["b_argmax"] = b.argmax(-1).tolist()
    return value


def bf16_reduction_flag(torch_module=torch) -> dict:
    """Return availability and the observed global flag without assuming a CUDA build."""
    matmul = getattr(getattr(getattr(torch_module, "backends", None), "cuda", None), "matmul", None)
    if matmul is None or not hasattr(matmul, "allow_bf16_reduced_precision_reduction"):
        return {"available": False, "value": None}
    return {"available": True, "value": bool(matmul.allow_bf16_reduced_precision_reduction)}


def set_bf16_reduction(enabled: bool, torch_module=torch) -> dict:
    """Set the sole experimental switch, recording unavailable builds explicitly."""
    before = bf16_reduction_flag(torch_module)
    if not before["available"]:
        return {"requested": enabled, "before": before, "after": before, "unavailable": True}
    torch_module.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = enabled
    after = bf16_reduction_flag(torch_module)
    if after["value"] is not enabled:
        raise RuntimeError("bf16 reduced-precision-reduction flag did not take requested value")
    return {"requested": enabled, "before": before, "after": after, "unavailable": False}


class TraceCache(DynamicCache):
    def __init__(self, trace: dict[str, torch.Tensor]):
        super().__init__()
        self.trace = trace

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx == 0:
            self.trace["k_post_rope"] = key_states.transpose(1, 2).detach().cpu().clone()
            self.trace["v_post_rope"] = value_states.transpose(1, 2).detach().cpu().clone()
        return super().update(key_states, value_states, layer_idx, cache_kwargs)


def capture_projection(trace: dict[str, torch.Tensor], name: str):
    """Copy a projection output for comparison without replacing the live output."""

    def hook(_module, _args, output):
        trace.setdefault(name, output.detach().cpu().clone())
        return None

    return hook


@torch.no_grad()
def traced_prefill(model, ids: list[int], lengths: list[int], query_length: int):
    runtime_dependencies()
    trace: dict[str, torch.Tensor] = {}
    handles = []
    # Qwen2 builds RoPE cos/sin on position_ids' device. Model.device can be
    # misleading for a wrapped model, so derive the execution device from its
    # actual parameters and use it for every RoPE-relevant input.
    device = next(model.parameters()).device
    attn = model.model.layers[0].self_attn
    for name in ("q_proj", "k_proj", "v_proj"):
        handles.append(getattr(attn, name).register_forward_hook(capture_projection(trace, name)))
    positions = build_positions(lengths, query_length, 0, "restart")
    mask = to_additive(build_mask(lengths, query_length, 0, "ddi_qf"), model.dtype)
    try:
        out = model(
            input_ids=torch.tensor([ids], device=device),
            position_ids=positions[None].to(device),
            attention_mask=mask[None, None].to(device),
            past_key_values=TraceCache(trace),
            use_cache=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    return out, trace


def entries_from(out, lengths: list[int]) -> list[CacheEntry]:
    runtime_dependencies()
    entries, offset = [], 0
    for doc, length in enumerate(lengths):
        entries.append(
            CacheEntry(
                key=f"doc-{doc}",
                doc_len=length,
                layout="restart",
                tensor_dtype=str(out.past_key_values.key_cache[0].dtype),
                k_per_layer=[
                    x[..., offset : offset + length, :].detach().clone()
                    for x in out.past_key_values.key_cache
                ],
                v_per_layer=[
                    x[..., offset : offset + length, :].detach().clone()
                    for x in out.past_key_values.value_cache
                ],
            )
        )
        offset += length
    return entries


def assembled_exact(entries: list[CacheEntry], model) -> list[dict]:
    runtime_dependencies()
    cache = assemble_dynamic_cache(entries, model.device)
    rows = []
    for kind, tensors in (("k", cache.key_cache), ("v", cache.value_cache)):
        field = f"{kind}_per_layer"
        for layer, tensor in enumerate(tensors):
            expected = torch.cat(
                [getattr(entry, field)[layer].to(model.device) for entry in entries], -2
            )
            rows.append({"layer": layer, "kind": kind, **comparison(expected, tensor)})
    if not all(row["finite"] and row["equal"] for row in rows):
        raise AssertionError("cache assembly is not exact")
    return rows


def disk_roundtrip(entries: list[CacheEntry], output: Path) -> tuple[list[CacheEntry], list[dict]]:
    runtime_dependencies()
    store, loaded, rows = DocCache(output / "roundtrip"), [], []
    for entry in entries:
        store.put(entry)
        restored = store.get(entry.key)
        if restored is None:
            raise AssertionError("disk payload disappeared")
        checks = [
            comparison(x, y)
            for field in ("k_per_layer", "v_per_layer")
            for x, y in zip(getattr(entry, field), getattr(restored, field), strict=True)
        ]
        if not all(check["finite"] and check["equal"] for check in checks):
            raise AssertionError("disk roundtrip is not exact")
        rows.append(
            {"key": entry.key, "file_sha256": sha(store._path(entry.key)), "tensors": checks}
        )
        loaded.append(restored)
    return loaded, rows


@torch.no_grad()
def diagnose(model, docs: list[list[int]], query: list[int], output: Path) -> dict:
    runtime_dependencies()
    lengths = [len(doc) for doc in docs]
    a, a_trace = traced_prefill(model, sum(docs, []) + query, lengths, len(query))
    a_entries = entries_from(a, lengths)
    b_entries, b_traces = [], []
    for index, doc in enumerate(docs):
        out, trace = traced_prefill(model, doc, [len(doc)], 0)
        entry = entries_from(out, [len(doc)])[0]
        b_entries.append(CacheEntry(**{**entry.__dict__, "key": f"doc-{index}"}))
        b_traces.append(trace)
    projection = []
    offset = 0
    for document, (length, trace) in enumerate(zip(lengths, b_traces, strict=True)):
        for operation in ("q_proj", "k_proj", "v_proj", "k_post_rope", "v_post_rope"):
            projection.append(
                {
                    "layer": 0,
                    "document": document,
                    "operation": operation,
                    **comparison(a_trace[operation][:, offset : offset + length], trace[operation]),
                }
            )
        offset += length
    c_entries, roundtrip = disk_roundtrip(b_entries, output)
    assembly = {
        "A_slices": assembled_exact(a_entries, model),
        "B": assembled_exact(b_entries, model),
        "C": assembled_exact(c_entries, model),
    }
    paths = {
        "A": a,
        "B": forward_query_with_doc_cache(model, torch.tensor([query]), b_entries),
        "C": forward_query_with_doc_cache(model, torch.tensor([query]), c_entries),
    }
    logits = {}
    for left, right in (("A", "B"), ("B", "C")):
        x, y = paths[left].logits[:, -len(query) :], paths[right].logits[:, -len(query) :]
        logits[f"{left}_vs_{right}"] = {
            "all": comparison(x, y, logits=True),
            "last": comparison(x[:, -1], y[:, -1], logits=True),
        }
    # The original bounded greedy diagnostic: all arms consume A's selected token.
    generated, greedy, first_disagreement = [], [], None
    total, max_steps = sum(lengths) + len(query), 16
    mask = to_additive(build_mask(lengths, len(query), max_steps, "ddi_qf"), model.dtype)
    for step in range(max_steps):
        selected = int(paths["A"].logits[0, -1].argmax())
        checks = {
            f"{left}_vs_{right}": comparison(
                paths[left].logits[:, -1], paths[right].logits[:, -1], logits=True
            )
            for left, right in (("A", "B"), ("B", "C"))
        }
        if not checks["A_vs_B"]["argmax_equal"] and first_disagreement is None:
            first_disagreement = step
        greedy.append(
            {
                "generated_position": step,
                "shared_prefix": list(generated),
                "a_selected_token": selected,
                "comparisons": checks,
            }
        )
        generated.append(selected)
        for name, old in list(paths.items()):
            paths[name] = model(
                input_ids=torch.tensor([[selected]], device=model.device),
                attention_mask=mask[total + step : total + step + 1, : total + step + 1][
                    None, None, None
                ].to(model.device),
                position_ids=torch.tensor(
                    [[max(lengths) + len(query) + step]], device=model.device
                ),
                cache_position=torch.tensor([total + step], device=model.device),
                past_key_values=old.past_key_values,
                use_cache=True,
            )
    return {
        "layer0_preassembly": projection,
        "first_divergent_operation": next((row for row in projection if not row["equal"]), None),
        "roundtrip": roundtrip,
        "assembly": assembly,
        "query_logits": logits,
        "teacher_forced_greedy": {"steps": greedy, "first_argmax_disagreement": first_disagreement},
        "exact_gate": {"atol": 1e-4, "rtol": 1e-4, "relaxed_gate_set": False},
    }


def runtime_flags() -> dict:
    return {
        "bf16_reduced_reduction": bf16_reduction_flag(),
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "sdpa_flash": torch.backends.cuda.flash_sdp_enabled(),
        "sdpa_math": torch.backends.cuda.math_sdp_enabled(),
        "sdpa_mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
    }


def input_ids() -> dict:
    value = json.loads(V58_INPUT.read_text())
    checkpoint = Path(value["checkpoint"]).resolve()
    if (
        value.get("seed") != 58
        or [len(x) for x in value.get("docs", [])] != [32, 32]
        or len(value.get("query_ids", [])) != 8
    ):
        raise ValueError("V58 observed IDs are not the frozen N=2,L=32,Q=8 input")
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    return value


def freeze(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"unique output already exists: {output}")
    if shutil.disk_usage(output.parent).free < LIMITS["min_free_disk_gib"] * GIB:
        raise RuntimeError("less than 20 GiB disk free")
    observed = input_ids()
    output.mkdir(parents=True)
    shutil.copyfile(V58_INPUT, output / "observed_v58_ids.json")
    sources = {}
    for path in (
        Path(__file__),
        ROOT / "ddi/cache.py",
        ROOT / "ddi/inference.py",
        ROOT / "ddi/masks.py",
        ROOT / "ddi/model.py",
        ROOT / "ddi/positions.py",
        ROOT / "tests/test_ablate_bf16_reduction.py",
    ):
        target = output / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        sources[str(path.relative_to(ROOT))] = sha(target)
    patch = subprocess.check_output(
        ["git", "diff", "HEAD", "--", "ddi", "scripts", "tests"], cwd=ROOT
    )
    (output / "working.patch").write_bytes(patch)
    checkpoint = Path(observed["checkpoint"]).resolve()
    tree = {
        str(path.relative_to(checkpoint)): {
            "sha256": sha(path),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(checkpoint.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "data_role": "development_observed_V58_ids_only",
        "observed_input_sha256": sha(output / "observed_v58_ids.json"),
        "checkpoint": str(checkpoint),
        "checkpoint_and_tokenizer_tree": tree,
        "sources": sources,
        "patch_sha256": sha(output / "working.patch"),
        "base_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dirty_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ),
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "cuda": torch.version.cuda,
        "torch_build": torch.__config__.show(),
        "flags_at_freeze": runtime_flags(),
        "limits": LIMITS,
        "arms": {
            "default": "PyTorch default",
            "no_bf16_reduced_reduction": "allow_bf16_reduced_precision_reduction=False",
        },
        "diagnosis": (
            "V60 observed layer-0 q_proj divergence before assembly; B/C were exact. "
            "The flag is a hypothesis, not an established cause."
        ),
        "no_fp32_arm": True,
        "no_relaxed_success_gate": True,
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "manifest.sha256.json", {"sha256": sha(output / "manifest.json")})
    if (
        sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        >= LIMITS["artifacts_gib"] * GIB
    ):
        raise RuntimeError("freeze exceeds artifact budget")


def verify_manifest(output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())
    if (
        sha(output / "manifest.json")
        != json.loads((output / "manifest.sha256.json").read_text())["sha256"]
    ):
        raise RuntimeError("manifest hash changed")
    if sha(output / "observed_v58_ids.json") != manifest["observed_input_sha256"]:
        raise RuntimeError("observed V58 IDs changed")
    for relative, expected in manifest["sources"].items():
        if sha(ROOT / relative) != expected or sha(output / "source" / relative) != expected:
            raise RuntimeError(f"source identity changed: {relative}")
    for name, expected in manifest["checkpoint_and_tokenizer_tree"].items():
        path = Path(manifest["checkpoint"]) / name
        if (
            not path.is_file()
            or sha(path) != expected["sha256"]
            or path.stat().st_size != expected["size"]
            or path.stat().st_mtime_ns != expected["mtime_ns"]
        ):
            raise RuntimeError(f"checkpoint/tokenizer identity changed: {name}")
    return manifest


def preflight(output: Path) -> dict:
    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not selected or "," in selected:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to exactly one A100 index")
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    processes = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    row = next(
        (line.split(", ") for line in gpu.splitlines() if line.split(", ")[0] == selected), None
    )
    if row is None or "A100" not in row[2] or int(row[4]) >= 512 or row[1] in processes:
        raise RuntimeError("selected A100 is not idle")
    value = {
        "gpu": gpu,
        "processes": processes,
        "visible": selected,
        "disk": shutil.disk_usage(output).free,
    }
    if value["disk"] < LIMITS["min_free_disk_gib"] * GIB:
        raise RuntimeError("less than 20 GiB disk free")
    return value


def arm(output: Path, mode: str) -> None:
    runtime_dependencies()
    arm_output = output / mode
    arm_output.mkdir()
    started = time.monotonic()
    result = {
        "mode": mode,
        "started_utc": datetime.now(UTC).isoformat(),
        "flags_before": runtime_flags(),
    }
    try:
        # Each arm is a new interpreter; no CUDA/global-backend state can cross arms.
        result["bf16_flag_transition"] = (
            set_bf16_reduction(mode == "default")
            if mode == "no_bf16_reduced_reduction"
            else {
                "requested": "default",
                "before": bf16_reduction_flag(),
                "after": bf16_reduction_flag(),
                "unavailable": not bf16_reduction_flag()["available"],
            }
        )
        result["flags_after_setting"] = runtime_flags()
        manifest = verify_manifest(output)
        observed = json.loads((output / "observed_v58_ids.json").read_text())
        torch.manual_seed(observed["seed"])
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, 44 * GIB / torch.cuda.get_device_properties(0).total_memory)
        )
        torch.cuda.reset_peak_memory_stats()
        model, _ = load(manifest["checkpoint"], dtype="bf16", backend="sdpa")
        model = model.to("cuda").eval()
        result["diagnostic"] = diagnose(model, observed["docs"], observed["query_ids"], arm_output)
        torch.cuda.synchronize()
        result["resources"] = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "allocator_cap_gib": 44,
        }
        if result["resources"]["peak_reserved_bytes"] > LIMITS["gpu_gib"] * GIB:
            raise RuntimeError("45 GiB peak limit exceeded")
        result["flags_post_run"] = runtime_flags()
        result["status"] = "completed_diagnostic_not_success_gate"
    except BaseException:
        result["status"] = "failed"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        result["artifact_bytes"] = sum(
            path.stat().st_size for path in output.rglob("*") if path.is_file()
        )
        write_json(arm_output / "result.json", result)


def run(output: Path) -> None:
    write_json(output / "run_claim.json", {"started_utc": datetime.now(UTC).isoformat()})
    started = time.monotonic()
    result = {"subprocess_arms": [], "limits": LIMITS}
    try:
        result["preflight"] = preflight(output)
        verify_manifest(output)
        for mode in ("default", "no_bf16_reduced_reduction"):
            remaining = int(LIMITS["seconds"] - (time.monotonic() - started))
            if remaining <= 0:
                raise TimeoutError("parent 600 second budget exhausted before arm")
            command = [
                sys.executable,
                "-B",
                "-m",
                "scripts.ablate_bf16_reduction",
                "arm",
                "--output",
                str(output),
                "--mode",
                mode,
            ]
            completed = subprocess.run(
                command, cwd=ROOT, text=True, capture_output=True, timeout=remaining, check=True
            )
            result["subprocess_arms"].append(
                {
                    "mode": mode,
                    "command": command,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            )
        result["status"] = "completed_diagnostic_not_success_gate"
    except BaseException:
        result["status"] = "failed"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        result["artifact_bytes"] = sum(
            path.stat().st_size for path in output.rglob("*") if path.is_file()
        )
        if (
            result["wall_seconds"] > LIMITS["seconds"]
            or result["artifact_bytes"] >= LIMITS["artifacts_gib"] * GIB
        ):
            result["status"] = "failed_limits"
        write_json(output / "result.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "arm"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("default", "no_bf16_reduced_reduction"))
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == "freeze":
        freeze(output)
    elif args.action == "run":
        run(output)
    else:
        if args.mode is None:
            parser.error("arm requires --mode")
        arm(output, args.mode)


if __name__ == "__main__":
    main()
