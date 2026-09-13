"""Frozen, untimed diagnostic of the real eager ``CanonicalCache`` B/C path.

This is deliberately a narrow development-observed V58 check.  A is the retained
SDPA monolithic DDI-QF prefill baseline; B and C are *separate* eager instances of
the actual ``ddi.canonical_cache.CanonicalCache`` API (fresh and disk-backed).
It records A/B 1e-4 diagnostics, but only exact B/C checks are gates.  It makes no
timing, throughput, QA, or order-invariance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from ddi.cache import DocCache, assemble_dynamic_cache
from ddi.canonical_cache import (
    CanonicalCache,
    ExecutionContract,
    model_config_identity,
    runtime_identity,
)
from ddi.inference import load
from ddi.masks import build_mask, to_additive
from ddi.positions import build_positions
from scripts.diagnose_canonical_kv import GIB, sha, write_json

ROOT = Path(__file__).resolve().parents[1]
UTC = UTC


def compare(left: torch.Tensor, right: torch.Tensor, *, logits: bool = False) -> dict[str, Any]:
    """Small JSON-safe comparison; it never writes logits or KV tensors as artifacts."""
    left, right = left.detach().cpu(), right.detach().cpu()
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    result: dict[str, Any] = {
        "shape": list(left.shape),
        "finite": finite,
        "equal": torch.equal(left, right),
        "max_abs": float((left.float() - right.float()).abs().max()) if finite else None,
        "allclose_1e4": torch.allclose(left.float(), right.float(), atol=1e-4, rtol=1e-4),
    }
    if logits:
        result["argmax_equal"] = torch.equal(left.argmax(-1), right.argmax(-1))
        result["left_argmax"] = left.argmax(-1).tolist()
        result["right_argmax"] = right.argmax(-1).tolist()
    return result


def _ids(values: list[int]) -> torch.Tensor:
    return torch.tensor([values], dtype=torch.long)


def _a_baseline(model: Any, docs: list[torch.Tensor], query: torch.Tensor) -> Any:
    """The original V58 monolithic SDPA DDI-QF baseline, intentionally not altered."""
    lengths = [int(doc.shape[1]) for doc in docs]
    query_length = int(query.shape[1])
    tokens = torch.cat([*docs, query], dim=1).to(model.device)
    positions = build_positions(lengths, query_length, 0, "restart").unsqueeze(0)
    mask = to_additive(build_mask(lengths, query_length, 0, "ddi_qf"), model.dtype)
    with torch.no_grad():
        return model(
            input_ids=tokens,
            position_ids=positions.to(model.device),
            attention_mask=mask[None, None].to(model.device),
            use_cache=True,
        )


def _entry_checks(source, disk, device: torch.device) -> dict[str, Any]:
    tensors = [
        compare(a, b.to(device))
        for field in ("k_per_layer", "v_per_layer")
        for a, b in zip(getattr(source, field), getattr(disk, field), strict=True)
    ]
    metadata = (
        "key",
        "doc_len",
        "layout",
        "cache_identity",
        "token_ids_sha256",
        "tensor_dtype",
        "schema_version",
    )
    return {
        "metadata_equal": all(getattr(source, name) == getattr(disk, name) for name in metadata),
        "tensors": tensors,
    }


def _assembly_checks(entries, device: torch.device) -> list[dict[str, Any]]:
    assembled = assemble_dynamic_cache(entries, device)
    return [
        compare(
            torch.cat([getattr(entry, field)[layer].to(device) for entry in entries], -2), value
        )
        for field, values in (
            ("k_per_layer", assembled.key_cache),
            ("v_per_layer", assembled.value_cache),
        )
        for layer, value in enumerate(values)
    ]


def fixed_cases(docs: list[list[int]], query: list[int]) -> list[dict[str, Any]]:
    """All inputs are token-native and fixed before any model is loaded."""
    # Use ordinary low token IDs rather than arithmetic modulo (which could create pad ID 0).
    first_changed = docs[0].copy()
    first_changed[0] = 2 if first_changed[0] != 2 else 3
    query_changed = query.copy()
    query_changed[0] = 2 if query_changed[0] != 2 else 3
    return [
        {"name": "base_cold", "documents": docs, "query": query},
        {"name": "base_warm", "documents": docs, "query": query},
        {"name": "reordered", "documents": [docs[1], docs[0]], "query": query},
        {"name": "duplicate", "documents": [docs[0], docs[0], docs[1]], "query": query},
        {
            "name": "partial_changed_doc_one_token",
            "documents": [first_changed, docs[1]],
            "query": query,
        },
        {"name": "changed_query_one_token", "documents": docs, "query": query_changed},
        {"name": "neighbor_shortened31", "documents": [docs[0], docs[1][:-1]], "query": query},
    ]


def _contract(model: Any, identities: dict[str, str]) -> ExecutionContract:
    return ExecutionContract(
        weights_identity=identities["weights_identity"],
        adapters_identity="none",
        tokenizer_identity=identities["tokenizer_identity"],
        model_config_identity=model_config_identity(model),
        runtime_identity=runtime_identity(model),
        backend="eager",
        dtype=str(model.dtype),
    )


def diagnose(
    model_a: Any | None,
    model_bc: Any,
    docs: list[list[int]],
    query: list[int],
    cache_dir: Path,
    identities: dict[str, str],
    *,
    natural_generation: bool = True,
    frozen_a_logits: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Run fixed cases.  ``model_bc`` must be an eager Qwen model used by both B and C."""
    if model_a is not None and model_a.config._attn_implementation != "sdpa":
        raise ValueError("A must remain the SDPA baseline")
    if model_a is None and frozen_a_logits is None:
        raise ValueError("requires the SDPA A baseline or its frozen per-case logits")
    if model_bc.config._attn_implementation != "eager":
        raise ValueError("B/C must share the eager canonical backend")
    contract = _contract(model_bc, identities)
    fresh = CanonicalCache(model_bc, contract)
    cached = CanonicalCache(model_bc, contract, DocCache(cache_dir))
    report: dict[str, Any] = {
        "data_role": "development_observed_V58",
        "untimed": True,
        "a_backend": "sdpa",
        "bc_backend": "eager",
        "canonical_api": "CanonicalCache",
        "cases": [],
        "d_reference": "not_run_optional",
        "no_qa_or_speed_claim": True,
    }
    for spec in fixed_cases(docs, query):
        documents, query_ids = [_ids(x) for x in spec["documents"]], _ids(spec["query"])
        a_logits = (
            frozen_a_logits[spec["name"]]
            if frozen_a_logits is not None
            else _a_baseline(model_a, documents, query_ids).logits[:, -len(spec["query"]) :]
        )
        b = fresh.request(documents, query_ids)
        c = cached.request(documents, query_ids)
        b_entries = fresh.documents(documents)
        c_entries = cached.documents(documents)
        repeated = [fresh.encode_document(document) for document in documents]
        repeat_checks = [
            compare(x, y)
            for entry, again in zip(b_entries, repeated, strict=True)
            for x, y in zip(
                entry.k_per_layer + entry.v_per_layer,
                again.k_per_layer + again.v_per_layer,
                strict=True,
            )
        ]
        disk_checks = [
            _entry_checks(x, y, model_bc.device) for x, y in zip(b_entries, c_entries, strict=True)
        ]
        b_query, c_query = b.logits, c.logits
        teacher = []
        prefix: list[int] = []
        for _ in range(3):
            b_step = fresh.request(documents, _ids(spec["query"] + prefix))
            c_step = cached.request(documents, _ids(spec["query"] + prefix))
            b_last, c_last = b_step.logits[:, -1], c_step.logits[:, -1]
            chosen = int(b_last.argmax(-1).item())
            teacher.append(
                {
                    "shared_prefix": prefix.copy(),
                    "b_vs_c_logits": compare(b_last, c_last, logits=True),
                    "b_greedy_id": chosen,
                    "c_greedy_id": int(c_last.argmax(-1).item()),
                }
            )
            prefix.append(chosen)
        row: dict[str, Any] = {
            "name": spec["name"],
            "documents": spec["documents"],
            "query": spec["query"],
            "a_vs_b_diagnostic_1e4": compare(a_logits, b_query, logits=True),
            "b_vs_c_query": compare(b_query, c_query, logits=True),
            "repeat_encoding_exact": repeat_checks,
            "source_disk_exact": disk_checks,
            "assembly_b_exact": _assembly_checks(b_entries, model_bc.device),
            "assembly_c_exact": _assembly_checks(c_entries, model_bc.device),
            "teacher_forced": teacher,
        }
        if natural_generation:
            b_generated = fresh.request(documents, query_ids, max_new_tokens=4)
            c_generated = cached.request(documents, query_ids, max_new_tokens=4)
            row["natural_generation_up_to4"] = {
                "b_ids": b_generated,
                "c_ids": c_generated,
                "exact": b_generated == c_generated,
            }
        report["cases"].append(row)
    exact_rows = []
    for row in report["cases"]:
        exact_rows.extend(
            [
                row["b_vs_c_query"],
                *row["repeat_encoding_exact"],
                *row["assembly_b_exact"],
                *row["assembly_c_exact"],
            ]
        )
        exact_rows.extend(check for disk in row["source_disk_exact"] for check in disk["tensors"])
        exact_rows.extend(step["b_vs_c_logits"] for step in row["teacher_forced"])
    report["exact_bc_gate"] = bool(
        all(check["equal"] and check["finite"] for check in exact_rows)
        and all(
            disk["metadata_equal"] for row in report["cases"] for disk in row["source_disk_exact"]
        )
        and all(row["natural_generation_up_to4"]["exact"] for row in report["cases"])
    )
    report["status"] = (
        "completed_development_diagnostic" if report["exact_bc_gate"] else "failed_exact_bc_gate"
    )
    return report


def _identity_for_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode() + b"\0" + sha(path).encode() + b"\0")
    return digest.hexdigest()


def _source_files() -> dict[str, str]:
    paths = [
        *ROOT.joinpath("ddi").rglob("*.py"),
        *ROOT.joinpath("scripts").rglob("*.py"),
        ROOT / "tests/test_validate_canonical_smoke.py",
    ]
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(paths)}


def freeze(output: Path, checkpoint: Path, inputs: Path) -> None:
    """Create a unique immutable input/source snapshot before GPU work."""
    output.mkdir(exist_ok=False)
    if shutil.disk_usage(output).free < 20 * GIB:
        raise RuntimeError("less than 20 GiB disk free")
    original = json.loads(inputs.read_text())
    docs, query = original["docs"], original["query_ids"]
    if original.get("seed") != 58 or list(map(len, docs)) != [32, 32] or len(query) != 8:
        raise ValueError("requires saved V58 seed-58 [32,32], query-8 IDs")
    if checkpoint.resolve() != Path(original["checkpoint"]).resolve():
        raise ValueError("checkpoint must match saved V58 input")
    for relative in ("ddi", "scripts", "tests/test_validate_canonical_smoke.py"):
        source = ROOT / relative
        targets = source.rglob("*.py") if source.is_dir() else [source]
        for path in targets:
            target = output / "source" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    shutil.copyfile(inputs, output / "observed_input.json")
    patch = subprocess.check_output(
        ["git", "diff", "HEAD", "--", "ddi", "scripts", "tests"], cwd=ROOT
    )
    (output / "working.patch").write_bytes(patch)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "data_role": "development_observed_V58",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_identity": _identity_for_tree(checkpoint),
        "weights_identity": _identity_for_tree(checkpoint),
        "tokenizer_identity": _identity_for_tree(checkpoint),
        "input_sha256": sha(output / "observed_input.json"),
        "source_tree_identity": _identity_for_tree(output / "source"),
        "source_files": _source_files(),
        "patch_sha256": sha(output / "working.patch"),
        "seed": 58,
        "docs": docs,
        "query": query,
        "fixed_cases": fixed_cases(docs, query),
        "limits": {
            "gpus": 1,
            "seconds": 600,
            "gpu_gib": 45,
            "artifacts_gib": 2,
            "min_free_disk_gib": 20,
        },
        "a_backend": "sdpa",
        "bc_backend": "eager",
        "executing_command": (
            f"timeout 600s {sys.executable} -B -m scripts.validate_canonical_smoke "
            f"run --output {output}"
        ),
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "manifest.sha256.json", {"sha256": sha(output / "manifest.json")})
    for path in output.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    print(
        json.dumps(
            {"frozen": str(output), "source_tree_identity": manifest["source_tree_identity"]}
        )
    )


def run(output: Path) -> None:
    write_json(output / "run_claim.json", {"started_utc": datetime.now(UTC).isoformat()})
    result: dict[str, Any] = {}
    started = time.monotonic()
    try:
        manifest = json.loads((output / "manifest.json").read_text())
        assert (
            sha(output / "manifest.json")
            == json.loads((output / "manifest.sha256.json").read_text())["sha256"]
        )
        assert sha(output / "observed_input.json") == manifest["input_sha256"]
        assert _identity_for_tree(output / "source") == manifest["source_tree_identity"]
        assert all(
            sha(ROOT / name) == expected for name, expected in manifest["source_files"].items()
        ), "sources changed after freeze"
        assert _identity_for_tree(Path(manifest["checkpoint"])) == manifest["checkpoint_identity"]
        assert shutil.disk_usage(output).free >= 20 * GIB
        selected = __import__("os").environ.get("CUDA_VISIBLE_DEVICES", "")
        assert selected and "," not in selected, "explicit single GPU required"
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        smi = subprocess.check_output(["nvidia-smi"], text=True)
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
        row = next(line.split(", ") for line in gpu.splitlines() if line.split(", ")[0] == selected)
        assert "A100" in row[2] and int(row[4]) < 512 and row[1] not in processes, (
            "selected A100 is not idle"
        )
        write_json(
            output / "preflight.json",
            {
                "nvidia_smi": smi,
                "gpu": gpu,
                "processes": processes,
                "df": subprocess.check_output(["df", "-B1", str(output)], text=True),
            },
        )
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, 44 * GIB / torch.cuda.get_device_properties(0).total_memory)
        )
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(manifest["seed"])
        model_a, _ = load(manifest["checkpoint"], dtype="bf16", backend="sdpa")
        model_a = model_a.to("cuda").eval()
        frozen_a_logits = {}
        for spec in manifest["fixed_cases"]:
            docs = [_ids(values) for values in spec["documents"]]
            query = _ids(spec["query"])
            frozen_a_logits[spec["name"]] = (
                _a_baseline(model_a, docs, query).logits[:, -query.shape[1] :].cpu()
            )
        del model_a
        torch.cuda.empty_cache()
        model_bc, _ = load(manifest["checkpoint"], dtype="bf16", backend="eager")
        model_bc = model_bc.to("cuda").eval()
        result = diagnose(
            None,
            model_bc,
            manifest["docs"],
            manifest["query"],
            output / "canonical_cache",
            manifest,
            natural_generation=True,
            frozen_a_logits=frozen_a_logits,
        )
        torch.cuda.synchronize()
        result["resources"] = {
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "hard_cap_gib": 45,
        }
        assert result["resources"]["peak_reserved_bytes"] <= 45 * GIB
        assert result["exact_bc_gate"], "exact B/C canonical gate failed"
        assert sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) < 2 * GIB
    except BaseException:
        result.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        result["artifact_bytes"] = sum(
            path.stat().st_size for path in output.rglob("*") if path.is_file()
        )
        write_json(output / "result.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--inputs", type=Path, default=ROOT / "results/v58_diag_bf16_20260908.json")
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.output.resolve(), args.checkpoint.resolve(), args.inputs)
    else:
        run(args.output.resolve())


if __name__ == "__main__":
    main()
