"""V61 GPU timing runner: segmented FA2 candidate versus dense SDPA reference.

Bounded timing campaign on the real Qwen2.5-7B ``ddi_qf`` restart checkpoint: grid
``N x L`` with ``N in {1, 2, 5, 10}`` documents of ``L in {128, 512}`` tokens each and a
64-token query, seed 47, fixed token streams per cell. Both arms run on IDENTICAL
tokens/positions with ``use_cache=False`` prefill only:

- reference: SDPA model under the ``ddi_qf`` additive mask (``ddi.masks``);
- candidate: eager model patched with ``patch_model_for_segmented_fa2`` and armed with
  ``set_segmented_state(..., variant="ddi_qf")``.

Numerics are NOT re-checked here: the correctness gate was ratified separately via FP64
calibration (RMS <= 0.02, ``scripts/calibrate_fa2_fp64.py``); this run measures SPEED
only. Timing gate per cell: the lower 95% paired-bootstrap bound of
``latency(SDPA-DDI) / latency(segmented-FA2-DDI)`` must exceed 1.05.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import transformers

from ddi.inference import load
from ddi.masks import build_mask, to_additive
from ddi.positions import build_positions
from ddi.segmented_fa2 import (
    clear_segmented_state,
    patch_model_for_segmented_fa2,
    set_segmented_state,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "v61_timing_v1"
DOC_COUNTS = (1, 2, 5, 10)
DOC_LENGTHS = (128, 512)
QUERY_LENGTH = 64
SEED = 47
WARMUPS = 2
REPEATS = 10
BOOTSTRAP_DRAWS = 10_000
GATE_THRESHOLD = 1.05
TOKEN_LOW, TOKEN_HIGH = 100, 30_000
DEFAULT_CHECKPOINT = (
    ROOT / "results" / "checkpoints" / "Qwen__Qwen2.5-7B-Instruct__ddi_qf__8L_restart"
)

CRITERION_NOTE = (
    "Numerics gate ratified separately via FP64 calibration (RMS<=0.02, "
    "scripts/calibrate_fa2_fp64.py); this run measures SPEED only. Timing gate: lower 95% "
    "paired-bootstrap bound of latency(SDPA-DDI)/latency(segmented-FA2-DDI) > "
    f"{GATE_THRESHOLD}."
)
GATE_RULE = (
    "lower 95% paired-bootstrap bound of latency(SDPA-DDI)/latency(segmented-FA2-DDI) > "
    f"{GATE_THRESHOLD}"
)

_REQUIRED_CELL_KEYS = (
    "status",
    "n_docs",
    "doc_length",
    "doc_lengths",
    "query_length",
    "total_tokens",
    "tokens_sha256",
    "positions_sha256",
    "reference",
    "candidate",
    "ratio",
    "gate",
)


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(Path(path).read_bytes())


def _sha_list(values: list[int]) -> str:
    return _sha_bytes(json.dumps(values).encode("utf-8"))


def _git_output(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def build_runtime() -> dict[str, Any]:
    runtime: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "unknown",
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "transformers": transformers.__version__,
    }
    try:
        import flash_attn

        runtime["flash_attn"] = flash_attn.__version__
    except Exception:  # noqa: BLE001 - provenance only, never fatal
        runtime["flash_attn"] = "unavailable"
    return runtime


def collect_source_hashes(root: Path = ROOT) -> dict[str, str]:
    """SHA-256 of every ``ddi/**/*.py`` plus this runner — the hashed implementation surface."""
    paths = sorted((root / "ddi").rglob("*.py"))
    paths.append(Path(__file__).resolve())
    resolved_root = root.resolve()
    return {
        path.resolve().relative_to(resolved_root).as_posix(): _sha_file(path)
        for path in paths
    }


def collect_checkpoint_identity(checkpoint: Path) -> dict[str, Any]:
    """Cheap identity of a local HF checkpoint: config hashes + weight file sizes."""
    config = checkpoint / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"checkpoint config not found: {config}")
    weights = [
        {"name": path.name, "bytes": path.stat().st_size}
        for path in sorted(checkpoint.glob("*.safetensors"))
    ]
    identity: dict[str, Any] = {
        "path": str(checkpoint),
        "resolved_path": str(checkpoint.resolve()),
        "config_sha256": _sha_file(config),
        "weights": weights,
        "weights_bytes": sum(weight["bytes"] for weight in weights),
    }
    for name in (
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
    ):
        path = checkpoint / name
        if path.is_file():
            identity[f"{name}_sha256"] = _sha_file(path)
    return identity


def build_cell_specs() -> list[dict[str, Any]]:
    """The bounded grid: N x L cells, each with per-document lengths and the shared query."""
    specs = []
    for n_docs in DOC_COUNTS:
        for doc_length in DOC_LENGTHS:
            specs.append(
                {
                    "n_docs": n_docs,
                    "doc_length": doc_length,
                    "doc_lengths": [doc_length] * n_docs,
                    "query_length": QUERY_LENGTH,
                }
            )
    return specs


def cell_inputs(
    n_docs: int,
    doc_length: int,
    *,
    query_length: int = QUERY_LENGTH,
    seed: int = SEED,
) -> dict[str, list[int]]:
    """Fixed token stream + restart positions for one grid cell (reseeded per cell)."""
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(
        TOKEN_LOW, TOKEN_HIGH, (n_docs * doc_length + query_length,), generator=generator
    ).tolist()
    positions = build_positions(
        [doc_length] * n_docs, query_length, 0, "restart"  # type: ignore[arg-type]
    ).tolist()
    return {"tokens": tokens, "positions": positions}


def paired_bootstrap_ratio(
    reference_seconds: list[float],
    candidate_seconds: list[float],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = SEED,
) -> dict[str, Any]:
    """Paired bootstrap over per-repeat latency ratios (reference / candidate).

    Ratios are formed pairwise on repeat index (both arms saw identical inputs), then the
    mean ratio is resampled ``draws`` times; the reported interval is the 2.5/97.5
    percentile band of that bootstrap distribution.
    """
    ref = torch.as_tensor(reference_seconds, dtype=torch.float64)
    cand = torch.as_tensor(candidate_seconds, dtype=torch.float64)
    if ref.numel() == 0 or cand.numel() == 0:
        raise ValueError("paired bootstrap needs at least one repeat per arm")
    if ref.numel() != cand.numel():
        raise ValueError(
            f"unpaired repeats: reference has {ref.numel()}, candidate has {cand.numel()}"
        )
    if bool((ref <= 0).any()) or bool((cand <= 0).any()):
        raise ValueError("latencies must be strictly positive")
    ratios = ref / cand
    generator = torch.Generator().manual_seed(seed)
    index = torch.randint(0, ratios.numel(), (draws, ratios.numel()), generator=generator)
    means = ratios[index].mean(dim=1)
    low, high = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return {
        "mean": float(ratios.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "repeats": ratios.tolist(),
        "draws": draws,
        "seed": seed,
    }


def gate_pass(ratio_stats: dict[str, Any]) -> bool:
    """Timing gate for one cell: the lower 95% bootstrap bound beats the threshold."""
    return bool(ratio_stats["ci95_low"] > GATE_THRESHOLD)


def gate_overall(cells: list[dict[str, Any]]) -> bool:
    """The campaign passes only if every grid cell passes (an errored cell fails)."""
    return bool(cells) and all(bool(cell["gate"]["pass"]) for cell in cells)


def validate_cell_result(cell: dict[str, Any]) -> None:
    """Structural + arithmetic invariants for one successful cell result."""
    missing = [key for key in _REQUIRED_CELL_KEYS if key not in cell]
    if missing:
        raise ValueError(f"cell result missing keys: {missing}")
    if cell["query_length"] != QUERY_LENGTH:
        raise ValueError(f"query_length must be {QUERY_LENGTH}")
    if cell["total_tokens"] != sum(cell["doc_lengths"]) + cell["query_length"]:
        raise ValueError("total_tokens disagrees with doc_lengths + query_length")
    for arm in ("reference", "candidate"):
        repeats = cell[arm].get("repeats_seconds")
        if not repeats:
            raise ValueError(f"{arm} arm has no timed repeats")
        if any(not isinstance(seconds, (int, float)) or seconds <= 0 for seconds in repeats):
            raise ValueError(f"{arm} arm repeats must be positive numbers")
        median = cell[arm].get("median_seconds")
        if median is None or abs(median - statistics.median(repeats)) > 1e-12:
            raise ValueError(f"{arm} median_seconds disagrees with its repeats")
    ratio = cell["ratio"]
    if ratio["ci95_low"] > ratio["mean"] + 1e-12 or ratio["mean"] > ratio["ci95_high"] + 1e-12:
        raise ValueError("bootstrap ci95 does not bracket the ratio mean")
    expected_pass = ratio["ci95_low"] > cell["gate"]["threshold"]
    if bool(cell["gate"]["pass"]) != expected_pass:
        raise ValueError("gate.pass disagrees with the threshold rule")


def _timed_forward(
    model: torch.nn.Module,
    ids: torch.Tensor,
    additive: torch.Tensor,
    positions: torch.Tensor,
) -> float:
    """One prefill forward, cuda-synchronized wall-clock seconds."""
    device = model.device
    if device.type == "cuda":
        torch.cuda.synchronize()
    ids = ids.to(device)
    additive = additive.to(device)
    positions = positions.to(device)
    started = time.perf_counter()
    with torch.no_grad():
        model(
            input_ids=ids,
            attention_mask=additive,
            position_ids=positions,
            use_cache=False,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - started


def run_arm_timings(
    model: torch.nn.Module,
    ids: torch.Tensor,
    additive: torch.Tensor,
    positions: torch.Tensor,
    *,
    warmups: int = WARMUPS,
    repeats: int = REPEATS,
) -> list[float]:
    """Warm up, then return ``repeats`` wall-clock prefill seconds for one arm."""
    for _ in range(warmups):
        _timed_forward(model, ids, additive, positions)
    return [_timed_forward(model, ids, additive, positions) for _ in range(repeats)]


def run_cell(
    reference: torch.nn.Module, candidate: torch.nn.Module, spec: dict[str, Any]
) -> dict[str, Any]:
    """Time both arms on IDENTICAL tokens/positions; returns one validated cell result."""
    doc_lengths = spec["doc_lengths"]
    inputs = cell_inputs(spec["n_docs"], spec["doc_length"])
    ids = torch.tensor([inputs["tokens"]], dtype=torch.long)
    positions = torch.tensor([inputs["positions"]], dtype=torch.long)
    additive = (
        to_additive(build_mask(doc_lengths, QUERY_LENGTH, 0, "ddi_qf"), dtype=torch.bfloat16)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    reference_seconds = run_arm_timings(reference, ids, additive, positions)

    # The patched layers ignore the additive mask while armed; passing it keeps the two
    # call sites byte-identical so only the attention path differs between arms.
    set_segmented_state(candidate, doc_lengths, QUERY_LENGTH, variant="ddi_qf")
    try:
        candidate_seconds = run_arm_timings(candidate, ids, additive, positions)
    finally:
        clear_segmented_state(candidate)

    ratio = paired_bootstrap_ratio(reference_seconds, candidate_seconds)
    cell = {
        "status": "ok",
        "n_docs": spec["n_docs"],
        "doc_length": spec["doc_length"],
        "doc_lengths": doc_lengths,
        "query_length": QUERY_LENGTH,
        "total_tokens": len(inputs["tokens"]),
        "tokens_sha256": _sha_list(inputs["tokens"]),
        "positions_sha256": _sha_list(inputs["positions"]),
        "reference": {
            "backend": "sdpa",
            "repeats_seconds": reference_seconds,
            "median_seconds": statistics.median(reference_seconds),
        },
        "candidate": {
            "backend": "segmented_fa2",
            "repeats_seconds": candidate_seconds,
            "median_seconds": statistics.median(candidate_seconds),
        },
        "ratio": ratio,
        "gate": {"rule": GATE_RULE, "threshold": GATE_THRESHOLD, "pass": gate_pass(ratio)},
    }
    validate_cell_result(cell)
    return cell


def build_manifest(checkpoint: Path) -> dict[str, Any]:
    """Schema ``v61_timing_v1`` manifest: sources, git, runtime, checkpoint, cell inputs."""
    created = datetime.now(UTC)
    diff = _git_output("diff", "HEAD")
    cells = []
    for spec in build_cell_specs():
        inputs = cell_inputs(spec["n_docs"], spec["doc_length"])
        cells.append(
            {
                **spec,
                "total_tokens": len(inputs["tokens"]),
                "tokens": {"ids": inputs["tokens"], "sha256": _sha_list(inputs["tokens"])},
                "positions": {
                    "ids": inputs["positions"],
                    "sha256": _sha_list(inputs["positions"]),
                },
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": _sha_bytes(
            json.dumps(
                {
                    "checkpoint": str(checkpoint.resolve()),
                    "seed": SEED,
                    "created": created.isoformat(),
                },
                sort_keys=True,
            ).encode("utf-8")
        ),
        "created_utc": created.isoformat(),
        "criterion_note": CRITERION_NOTE,
        "grid": {
            "doc_counts": list(DOC_COUNTS),
            "doc_lengths": list(DOC_LENGTHS),
            "query_length": QUERY_LENGTH,
            "seed": SEED,
            "warmups": WARMUPS,
            "timed_repeats": REPEATS,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": SEED,
            "gate_rule": GATE_RULE,
            "gate_threshold": GATE_THRESHOLD,
        },
        "checkpoint": collect_checkpoint_identity(checkpoint),
        "git": {
            "source_revision": _git_output("rev-parse", "HEAD").decode().strip(),
            "diff_sha256": _sha_bytes(diff),
            "diff_empty": not diff,
        },
        "runtime": build_runtime(),
        "sources": collect_source_hashes(),
        "cells": cells,
    }


def _summary_entry(cell: dict[str, Any]) -> dict[str, Any]:
    entry = {
        "n_docs": cell["n_docs"],
        "doc_length": cell["doc_length"],
        "total_tokens": cell["total_tokens"],
        "status": cell["status"],
        "gate_pass": bool(cell["gate"]["pass"]),
    }
    if cell["status"] == "ok":
        entry.update(
            {
                "reference_median_seconds": cell["reference"]["median_seconds"],
                "candidate_median_seconds": cell["candidate"]["median_seconds"],
                "ratio_mean": cell["ratio"]["mean"],
                "ratio_ci95": [cell["ratio"]["ci95_low"], cell["ratio"]["ci95_high"]],
            }
        )
    else:
        entry["error"] = cell["error"]
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.output is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        args.output = ROOT / "results" / f"v61_timing_{stamp}"
    if args.output.exists():
        raise FileExistsError(f"--output must not exist: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("the V61 timing campaign requires CUDA")
    args.output.mkdir(parents=True)
    run_started = datetime.now(UTC)

    print(f"loading SDPA reference from {args.checkpoint}", flush=True)
    reference, _ = load(str(args.checkpoint), dtype="bf16", backend="sdpa")
    reference = reference.to("cuda").eval()
    print("loading eager candidate", flush=True)
    candidate, _ = load(str(args.checkpoint), dtype="bf16", backend="eager")
    candidate = patch_model_for_segmented_fa2(candidate.to("cuda").eval())
    torch.cuda.reset_peak_memory_stats()

    manifest = build_manifest(args.checkpoint)
    _write_json(args.output / "manifest.json", manifest)
    print(f"manifest_id={manifest['manifest_id']}", flush=True)

    cells = []
    for spec in manifest["cells"]:
        label = f"N={spec['n_docs']} L={spec['doc_length']}"
        try:
            cell = run_cell(reference, candidate, spec)
            ratio = cell["ratio"]
            print(
                f"{label}: ref={cell['reference']['median_seconds']:.4f}s "
                f"cand={cell['candidate']['median_seconds']:.4f}s "
                f"ratio={ratio['mean']:.3f} "
                f"ci95=[{ratio['ci95_low']:.3f}, {ratio['ci95_high']:.3f}] "
                f"gate={'PASS' if cell['gate']['pass'] else 'FAIL'}",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001 - record-and-continue is the protocol
            torch.cuda.empty_cache()
            cell = {
                "status": "error",
                "n_docs": spec["n_docs"],
                "doc_length": spec["doc_length"],
                "doc_lengths": spec["doc_lengths"],
                "query_length": QUERY_LENGTH,
                "total_tokens": spec["total_tokens"],
                "tokens_sha256": spec["tokens"]["sha256"],
                "positions_sha256": spec["positions"]["sha256"],
                "error": f"{type(error).__name__}: {error}",
                "gate": {"rule": GATE_RULE, "threshold": GATE_THRESHOLD, "pass": False},
            }
            print(f"{label}: ERROR {cell['error']}", flush=True)
        cells.append(cell)

    resources = {
        "device": torch.cuda.get_device_name(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "wall_seconds": (datetime.now(UTC) - run_started).total_seconds(),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": manifest["manifest_id"],
        "created_utc": manifest["created_utc"],
        "criterion_note": CRITERION_NOTE,
        "runtime": manifest["runtime"],
        "resources": resources,
        "cells": cells,
    }
    _write_json(args.output / "result.json", result)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": manifest["manifest_id"],
        "criterion_note": CRITERION_NOTE,
        "gate_rule": GATE_RULE,
        "gate_threshold": GATE_THRESHOLD,
        "resources": resources,
        "cells": [_summary_entry(cell) for cell in cells],
        "overall_pass": gate_overall(cells),
    }
    _write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if not summary["overall_pass"]:
        print("TIMING GATE NOT PASSED", flush=True)


if __name__ == "__main__":
    main()
