"""Bounded, untimed V58 development localization; never a confirmation or speed gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import torch
from transformers import DynamicCache

from ddi.cache import CacheEntry, DocCache, assemble_dynamic_cache, forward_query_with_doc_cache
from ddi.inference import load
from ddi.masks import build_mask, to_additive
from ddi.positions import build_positions

ROOT = Path(__file__).resolve().parents[1]
UTC = UTC
GIB = 1024**3


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def compare(a, b, logits=False):
    a, b = a.detach().cpu(), b.detach().cpu()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    result = dict(
        shape=list(a.shape),
        finite=finite,
        equal=torch.equal(a, b),
        max_abs=float((a.float() - b.float()).abs().max()) if finite else None,
        allclose_1e4=torch.allclose(a.float(), b.float(), atol=1e-4, rtol=1e-4),
    )
    if logits:
        result.update(
            argmax_equal=torch.equal(a.argmax(-1), b.argmax(-1)),
            a_argmax=a.argmax(-1).tolist(),
            b_argmax=b.argmax(-1).tolist(),
        )
    return result


class TraceCache(DynamicCache):
    def __init__(self, trace):
        super().__init__()
        self.trace = trace

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        for name, tensor in (("k_post_rope", key_states), ("v_cache", value_states)):
            self.trace.append((layer_idx, name, tensor.transpose(1, 2).detach().cpu().clone()))
        return super().update(key_states, value_states, layer_idx, cache_kwargs)


def traced_forward(model, ids, lengths, query_length):
    trace, handles = [], []

    def install(module, layer, name, before=False):
        def save(tensor):
            if isinstance(tensor, tuple):
                tensor = tensor[0]
            trace.append((layer, name, tensor.detach().cpu().clone()))

        if before:
            handles.append(module.register_forward_pre_hook(lambda mod, args: save(args[0])))
        else:
            handles.append(module.register_forward_hook(lambda mod, args, out: save(out)))

    install(model.model.embed_tokens, -1, "embedding")

    def save_rotary(module, args, out):
        for name, tensor in zip(("rope_cos", "rope_sin"), out, strict=True):
            trace.append((-1, name, tensor.detach().cpu().clone()))

    handles.append(model.model.rotary_emb.register_forward_hook(save_rotary))
    for i, layer in enumerate(model.model.layers):
        install(layer.input_layernorm, i, "input_hidden", True)
        install(layer.input_layernorm, i, "input_norm")
        for name in ("q_proj", "k_proj", "v_proj"):
            install(getattr(layer.self_attn, name), i, name)
        install(layer.self_attn.o_proj, i, "attention_before_o_proj", True)
        install(layer.self_attn, i, "attention_output")
        install(layer.post_attention_layernorm, i, "post_attention_residual", True)
        install(layer.post_attention_layernorm, i, "post_attention_norm")
        install(layer.mlp, i, "mlp_output")
        install(layer, i, "layer_output")
    positions = build_positions(lengths, query_length, 0, "restart")
    mask = to_additive(build_mask(lengths, query_length, 0, "ddi_qf"), model.dtype)
    try:
        out = model(
            input_ids=torch.tensor([ids], device=model.device),
            position_ids=positions[None].to(model.device),
            attention_mask=mask[None, None].to(model.device),
            past_key_values=TraceCache(trace),
            use_cache=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    return out, trace, positions.tolist()


def trace_comparisons(combined, independent, lengths):
    """Operation-major, then document-major: never report a later layer of doc 0 first."""
    rows = []
    for event, (layer, operation, tensor) in enumerate(combined):
        offset = 0
        for doc, (trace, length) in enumerate(zip(independent, lengths, strict=True)):
            other_layer, other_operation, other = trace[event]
            assert (layer, operation) == (other_layer, other_operation)
            rows.append(
                dict(
                    event=event,
                    layer=layer,
                    operation=operation,
                    document=doc,
                    **compare(tensor[:, offset : offset + length], other),
                )
            )
            offset += length
    return rows


def entries_from(out, lengths):
    entries, offset = [], 0
    for doc, length in enumerate(lengths):
        entries.append(
            CacheEntry(
                key=f"doc-{doc}",
                doc_len=length,
                layout="restart",
                tensor_dtype=str(out.past_key_values.key_cache[0].dtype),
                k_per_layer=[
                    k[..., offset : offset + length, :].detach().clone()
                    for k in out.past_key_values.key_cache
                ],
                v_per_layer=[
                    v[..., offset : offset + length, :].detach().clone()
                    for v in out.past_key_values.value_cache
                ],
            )
        )
        offset += length
    return entries


def cache_comparisons(a, b):
    return [
        dict(layer=i, kind=kind, **compare(x, y))
        for kind, xs, ys in (("k", a.key_cache, b.key_cache), ("v", a.value_cache, b.value_cache))
        for i, (x, y) in enumerate(zip(xs, ys, strict=True))
    ]


@torch.no_grad()
def diagnose(model, docs, query, output):
    lengths = list(map(len, docs))
    report = dict(
        data_role="development_observed_V58",
        untimed=True,
        speed_claim=False,
        backend=model.config._attn_implementation,
        dtype=str(model.dtype),
    )

    def record(name, value):
        report[name] = value
        with (output / "events.jsonl").open("a") as handle:
            handle.write(json.dumps({name: value}, allow_nan=False) + "\n")

    a, combined, positions = traced_forward(model, sum(docs, []) + query, lengths, len(query))
    a_entries = entries_from(a, lengths)
    b_entries, independent, doc_positions = [], [], []
    for doc_index, doc in enumerate(docs):
        b, trace, pos = traced_forward(model, doc, [len(doc)], 0)
        entry = entries_from(b, [len(doc)])[0]
        entry = CacheEntry(
            **{**entry.__dict__, "key": f"doc-{doc_index}", "tensor_dtype": str(model.dtype)}
        )
        b_entries.append(entry)
        independent.append(trace)
        doc_positions.append(pos)
    rows = trace_comparisons(combined, independent, lengths)
    record("operations", rows)
    record("first_divergent_operation", next((r for r in rows if not r["equal"]), None))
    record("first_failed_tolerance", next((r for r in rows if not r["allclose_1e4"]), None))
    record(
        "positions",
        dict(
            combined=positions,
            independent=doc_positions,
            v_rope="V is not rotated; v_proj is pre-cache V",
        ),
    )
    del combined, independent, b

    store = DocCache(output / "roundtrip")
    c_entries, payload = [], []
    for entry in b_entries:
        store.put(entry)
        loaded = store.get(entry.key)
        assert loaded is not None
        blob = torch.load(store._path(entry.key), map_location="cpu", weights_only=True)
        meta = {k: v for k, v in entry.__dict__.items() if k not in {"k_per_layer", "v_per_layer"}}
        assert set(blob) == {"k", "v", "meta"} and blob["meta"] == meta
        checks = [
            compare(x, y)
            for field in ("k_per_layer", "v_per_layer")
            for x, y in zip(getattr(entry, field), getattr(loaded, field), strict=True)
        ]
        assert all(r["equal"] and r["finite"] for r in checks)
        payload.append(
            dict(
                key=entry.key,
                metadata_exact=True,
                tensors=checks,
                file_sha256=sha(store._path(entry.key)),
            )
        )
        c_entries.append(loaded)
    record("roundtrip_payload", payload)
    assemblies = {}
    for name, entries in (("A_slices", a_entries), ("B", b_entries), ("C", c_entries)):
        cache = assemble_dynamic_cache(entries, model.device)
        checks = [
            compare(torch.cat([getattr(e, field)[i].to(model.device) for e in entries], -2), t)
            for field, tensors in (
                ("k_per_layer", cache.key_cache),
                ("v_per_layer", cache.value_cache),
            )
            for i, t in enumerate(tensors)
        ]
        assert all(r["equal"] and r["finite"] for r in checks)
        assemblies[name] = checks
    record("assembly", assemblies)
    paths = {"A": a}
    for name, entries in (("A_replay", a_entries), ("B", b_entries), ("C", c_entries)):
        paths[name] = forward_query_with_doc_cache(model, torch.tensor([query]), entries)
    query_results = {}
    for left, right in (("A", "A_replay"), ("A", "B"), ("B", "C")):
        x, y = paths[left].logits[:, -len(query) :], paths[right].logits[:, -len(query) :]
        query_results[f"{left}_vs_{right}"] = dict(
            all=compare(x, y, True),
            last=compare(x[:, -1], y[:, -1], True),
            positions=[compare(x[:, i], y[:, i], True) for i in range(len(query))],
        )
    record("query_logits", query_results)
    record(
        "prefill_kv",
        {
            f"{x}_vs_{y}": cache_comparisons(paths[x].past_key_values, paths[y].past_key_values)
            for x, y in (("A", "A_replay"), ("A", "B"), ("B", "C"))
        },
    )

    # All paths ALWAYS consume A's token. No numeric comparison on mismatched prefixes.
    generated, steps, first_disagreement = [], [], None
    max_steps = 19
    total = sum(lengths) + len(query)
    mask = to_additive(build_mask(lengths, len(query), max_steps, "ddi_qf"), model.dtype)
    for step in range(max_steps):
        selected = int(paths["A"].logits[0, -1].argmax())
        checks = {
            f"{x}_vs_{y}": compare(paths[x].logits[:, -1], paths[y].logits[:, -1], True)
            for x, y in (("A", "A_replay"), ("A", "B"), ("B", "C"))
        }
        if not checks["A_vs_B"]["argmax_equal"] and first_disagreement is None:
            first_disagreement = step
        steps.append(
            dict(
                generated_position=step,
                shared_prefix=list(generated),
                a_selected_token=selected,
                comparisons=checks,
                kv_finite={
                    name: [
                        bool(torch.isfinite(k).all() and torch.isfinite(v).all())
                        for k, v in zip(
                            out.past_key_values.key_cache,
                            out.past_key_values.value_cache,
                            strict=True,
                        )
                    ]
                    for name, out in paths.items()
                },
            )
        )
        generated.append(selected)
        if step >= 15 and (first_disagreement is None or step >= first_disagreement + 3):
            break
        for name, previous in list(paths.items()):
            paths[name] = model(
                input_ids=torch.tensor([[selected]], device=model.device),
                attention_mask=mask[total + step, : total + step + 1][None, None, None].to(
                    model.device
                ),
                position_ids=torch.tensor(
                    [[max(lengths) + len(query) + step]], device=model.device
                ),
                cache_position=torch.tensor([total + step], device=model.device),
                past_key_values=previous.past_key_values,
                use_cache=True,
            )
    record(
        "teacher_forced",
        dict(
            first_argmax_disagreement=first_disagreement,
            steps=steps,
            eos_policy="ignore, bounded localization",
        ),
    )
    record(
        "final_kv",
        {
            f"{x}_vs_{y}": cache_comparisons(paths[x].past_key_values, paths[y].past_key_values)
            for x, y in (("A", "A_replay"), ("A", "B"), ("B", "C"))
        },
    )
    return report


def precision():
    return dict(
        tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic=torch.are_deterministic_algorithms_enabled(),
        matmul_precision=torch.get_float32_matmul_precision(),
        bf16_reduced_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        fp16_reduced_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        sdpa_flash=torch.backends.cuda.flash_sdp_enabled(),
        sdpa_math=torch.backends.cuda.math_sdp_enabled(),
        sdpa_mem_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
        threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
        autocast=False,
        backend="sdpa",
        dtype="bf16",
    )


def freeze(output, checkpoint, inputs):
    output.mkdir(exist_ok=False)
    if shutil.disk_usage(output).free < 20 * GIB:
        raise RuntimeError("less than 20 GiB disk free")
    original = json.loads(inputs.read_text())
    assert original["seed"] == 58 and list(map(len, original["docs"])) == [32, 32]
    assert len(original["query_ids"]) == 8
    assert checkpoint.resolve() == Path(original["checkpoint"]).resolve()
    sources, external, packages = {}, {}, {}
    for path in sorted(
        {
            *ROOT.joinpath("ddi").rglob("*.py"),
            *ROOT.joinpath("scripts").rglob("*.py"),
            ROOT / "tests/test_diagnose_canonical_kv.py",
        }
    ):
        target = output / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        sources[str(path.relative_to(ROOT))] = sha(target)
    # Hash actual package files, not only version strings or RECORD assertions.
    for name in (
        "torch",
        "transformers",
        "tokenizers",
        "safetensors",
        "numpy",
        "huggingface-hub",
    ):
        dist = importlib.metadata.distribution(name)
        packages[name] = dist.version
        for relative in dist.files or []:
            path = Path(dist.locate_file(relative)).resolve()
            if not path.is_file() or path.suffix == ".pyc":
                continue
            external[str(path)] = sha(path)
            if path.suffix == ".py":
                target = output / "external_source" / name / str(relative)
                # Distribution executable paths may contain '..'; only snapshot source modules.
                if ".." not in relative.parts:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
    runtime_metadata = {}
    for dist in importlib.metadata.distributions():
        runtime_metadata[dist.metadata["Name"]] = dict(
            version=dist.version,
            record_sha256=hashlib.sha256((dist.read_text("RECORD") or "").encode()).hexdigest(),
        )
    # Imported stdlib/dependency source and native modules complement the package-wide hashes.
    generated_modules = {}
    for module_name, module in list(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).is_file():
            path = Path(filename).resolve()
            if not path.is_relative_to(ROOT):
                if (
                    str(path) not in external
                    and path.parent.parent == Path(tempfile.gettempdir()).resolve()
                ):
                    target = output / "generated_source" / f"{module_name}.py"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
                    generated_modules[module_name] = sha(target)
                    continue
                external[str(path)] = sha(path)
                if path.suffix == ".py":
                    target = output / "imported_source" / str(path).lstrip("/")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
    patch = subprocess.check_output(
        ["git", "diff", "HEAD", "--", "ddi", "scripts", "tests"], cwd=ROOT
    )
    (output / "working.patch").write_bytes(patch)
    shutil.copyfile(inputs, output / "observed_input.json")
    manifest = dict(
        created_utc=datetime.now(UTC).isoformat(),
        data_role="development_observed_V58",
        checkpoint=str(checkpoint.resolve()),
        checkpoint_files={
            str(p.relative_to(checkpoint)): dict(
                sha256=sha(p), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns
            )
            for p in sorted(checkpoint.rglob("*"))
            if p.is_file()
        },
        checkpoint_hash_policy="all files freshly hashed; no reused hashes and no shard copies",
        sources=sources,
        external_files=external,
        packages=packages,
        runtime_distribution_metadata=runtime_metadata,
        generated_modules=generated_modules,
        python=sys.version,
        executable=str(Path(sys.executable).resolve()),
        executable_sha256=sha(sys.executable),
        torch_build=torch.__config__.show(),
        cuda=torch.version.cuda,
        flags=precision(),
        base_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        dirty_status=subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).decode(),
        patch_sha256=sha(output / "working.patch"),
        input_sha256=sha(output / "observed_input.json"),
        docs=original["docs"],
        query=original["query_ids"],
        seed=58,
        limits=dict(gpus=1, seconds=600, gpu_gib=45, artifacts_gib=2, min_free_disk_gib=20),
        omitted_gates=[
            "untouched confirmation",
            "FP32",
            "timing/speed",
            "training",
            "workload/QA generalization",
            "canonical eager API validation",
        ],
        executing_command=(
            f"timeout 600s {sys.executable} -B -m scripts.diagnose_canonical_kv "
            f"run --output {output}"
        ),
    )
    write_json(output / "manifest.json", manifest)
    write_json(output / "manifest.sha256.json", dict(sha256=sha(output / "manifest.json")))
    for path in output.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    assert sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) < GIB
    print(
        json.dumps(
            dict(frozen=str(output), source_files=len(sources), external_files=len(external))
        )
    )


def verify_sources(output, manifest):
    failures = []
    for name, expected in manifest["sources"].items():
        for path in (ROOT / name, output / "source" / name):
            if sha(path) != expected:
                failures.append(str(path))
    for name, expected in manifest["external_files"].items():
        if sha(name) != expected:
            failures.append(name)
    for name, expected in manifest["generated_modules"].items():
        module = sys.modules.get(name)
        if (
            module is None
            or sha(module.__file__) != expected
            or sha(output / "generated_source" / f"{name}.py") != expected
        ):
            failures.append(name)
    return failures


def run(output):
    # Exclusive claim prevents rerunning a frozen manifest or overwriting a failed attempt.
    write_json(output / "run_claim.json", dict(started_utc=datetime.now(UTC).isoformat()))
    start = time.monotonic()
    result = {}

    def terminated(signum, frame):
        raise TimeoutError("external timeout terminated the bounded run")

    signal.signal(signal.SIGTERM, terminated)
    try:
        manifest = json.loads((output / "manifest.json").read_text())
        assert (
            sha(output / "manifest.json")
            == json.loads((output / "manifest.sha256.json").read_text())["sha256"]
        )
        assert not verify_sources(output, manifest), "sources changed after freeze"
        assert precision() == manifest["flags"]
        assert sha(output / "observed_input.json") == manifest["input_sha256"]
        for name, identity in manifest["checkpoint_files"].items():
            path = Path(manifest["checkpoint"]) / name
            assert path.stat().st_size == identity["size"]
            assert path.stat().st_mtime_ns == identity["mtime_ns"]
        # Immediately before CUDA initialization/model load, retain full device/disk state.
        smi = subprocess.check_output(["nvidia-smi"], text=True)
        df = subprocess.check_output(["df", "-B1", str(output)], text=True)
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
        write_json(
            output / "preflight.json",
            dict(
                nvidia_smi=smi,
                df=df,
                gpu=gpu,
                processes=processes,
                visible=os.getenv("CUDA_VISIBLE_DEVICES"),
            ),
        )
        assert shutil.disk_usage(output).free >= 20 * GIB
        selected = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        assert selected and "," not in selected, "explicit single GPU required"
        row = next(r.split(", ") for r in gpu.splitlines() if r.split(", ")[0] == selected)
        assert "A100" in row[2] and int(row[4]) < 512, "selected A100 is not idle"
        assert row[1] not in processes, "selected GPU has active compute work"
        assert torch.cuda.device_count() == 1
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, 44 * GIB / torch.cuda.get_device_properties(0).total_memory)
        )
        torch.cuda.reset_peak_memory_stats()
        gpu_start = time.monotonic()
        torch.manual_seed(manifest["seed"])
        model, _ = load(manifest["checkpoint"], dtype="bf16", backend="sdpa")
        model = model.to("cuda").eval()
        result = diagnose(model, manifest["docs"], manifest["query"], output)
        torch.cuda.synchronize()
        result["resources"] = dict(
            gpu_phase_wall_seconds=time.monotonic() - gpu_start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            allocator_cap_gib=44,
            hard_cap_gib=45,
        )
        assert torch.cuda.max_memory_reserved() <= 45 * GIB
        result["postflight_nvidia_smi"] = subprocess.check_output(["nvidia-smi"], text=True)
        result["source_changes_after"] = verify_sources(output, manifest)
        assert not result["source_changes_after"]
        result["status"] = "completed_localization_not_gate_pass"
    except BaseException:
        result.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        if torch.cuda.is_initialized():
            result["allocator_peak_bytes"] = dict(
                allocated=torch.cuda.max_memory_allocated(),
                reserved=torch.cuda.max_memory_reserved(),
            )
        result["wall_seconds"] = time.monotonic() - start
        result["artifact_bytes"] = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        if result["artifact_bytes"] >= 2 * GIB:
            result["status"] = "failed_artifact_budget"
        write_json(output / "result.json", result)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k
                    in {
                        "status",
                        "resources",
                        "artifact_bytes",
                        "wall_seconds",
                        "first_divergent_operation",
                    }
                }
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--inputs", type=Path, default=ROOT / "results/v58_diag_bf16_20260908.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.action == "freeze":
        freeze(args.output.resolve(), args.checkpoint.resolve(), args.inputs)
    else:
        run(args.output.resolve())


if __name__ == "__main__":
    main()
