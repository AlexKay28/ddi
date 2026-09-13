"""Executable V60 A/B/C/D development-matrix runner (CPU or a single GPU).

Per request, run four arms over identical token IDs and record per-arm
observations: A (original monolithic SDPA DDI-QF prefill), B (CanonicalCache
fresh encode, nothing persisted), C (CanonicalCache backed by a DocCache store,
in cold and all-hit scenarios), and D (full-causal reference over the same
stream with build_mask "full").  Greedy decode is teacher-forced on A's chain.
Artifacts: manifest.json (development provenance), cells.jsonl (one line per
request), result.json (TTFT medians, B/C equality flags, agreement counts).

This is development data ("data_role": "development"): it supports no
confirmation, QA, or cross-backend performance claim -- A/B timing differs by
backend and is inadmissible as a performance comparison per
scripts.validate_canonical_matrix.validate_comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import transformers

from ddi.cache import DocCache
from ddi.canonical_cache import (
    CanonicalCache,
    ExecutionContract,
    model_config_identity,
    runtime_identity,
)
from ddi.inference import load
from ddi.masks import build_mask, to_additive
from ddi.positions import build_positions
from scripts.validate_canonical_matrix import ARMS, SCHEMA_VERSION, arm_identity

ROOT = Path(__file__).resolve().parents[1]

LoadModels = Callable[[Path, str], tuple[Any, Any]]


def _sha_file(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _write_json(path: Path, value: Any) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _digest_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _git_bytes(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def _git_text(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _source_hashes() -> dict[str, str]:
    paths = sorted((ROOT / "ddi").rglob("*.py"))
    paths.append(Path(__file__).resolve())
    return {str(path.relative_to(ROOT)): _sha_file(path) for path in paths}


def _checkpoint_identity(checkpoint: Path) -> tuple[str, str]:
    path = Path(checkpoint)
    if path.is_dir():
        digest = hashlib.sha256()
        for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
            digest.update(str(file_path.relative_to(path)).encode() + b"\0")
            digest.update(_sha_file(file_path).encode() + b"\0")
        return "directory_tree", digest.hexdigest()
    return "path_only", hashlib.sha256(str(path.resolve()).encode()).hexdigest()


def _arm_identity(arm: str, dtype: str) -> str:
    return arm_identity(arm, {**ARMS[arm], "dtype": dtype, "layout": "restart"})


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_bytes(device: torch.device) -> tuple[int | None, int | None]:
    if device.type != "cuda":
        return None, None
    allocated = int(torch.cuda.max_memory_allocated(device))
    reserved = int(torch.cuda.max_memory_reserved(device))
    return allocated, reserved


def _logits_digest(row: torch.Tensor) -> str:
    data = row.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _clear_store(directory: Path) -> None:
    for path in directory.glob("*"):
        if path.is_file():
            path.unlink()


def _store_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.glob("*.pt"))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decode-length", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    args = parser.parse_args(argv)
    if args.decode_length < 1:
        parser.error("--decode-length must be >= 1")
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    return args


def _load_inputs(path: Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"seed", "docs", "query_ids"}:
        raise ValueError(
            "inputs must be an object with exactly the keys seed, docs, query_ids"
        )
    seed = raw["seed"]
    docs = raw["docs"]
    query_ids = raw["query_ids"]
    if type(seed) is not int:
        raise ValueError("inputs.seed must be an integer")
    if not isinstance(docs, list) or not docs:
        raise ValueError("inputs.docs must be a non-empty list of token id lists")
    for index, doc in enumerate(docs):
        if not isinstance(doc, list) or not doc or any(type(token) is not int for token in doc):
            raise ValueError(f"inputs.docs[{index}] must be a non-empty list of integers")
    if not isinstance(query_ids, list) or not query_ids:
        raise ValueError("inputs.query_ids must be a non-empty list of integers")
    if any(type(token) is not int for token in query_ids):
        raise ValueError("inputs.query_ids must contain only integers")
    return {"seed": seed, "docs": docs, "query_ids": query_ids}


def _validate_ids(
    config: Any, docs: list[list[int]], query_ids: list[int], decode_length: int
) -> None:
    vocab = int(config.vocab_size)
    pad = getattr(config, "pad_token_id", None)
    all_ids = [token for doc in docs for token in doc] + list(query_ids)
    if any(token < 0 or token >= vocab for token in all_ids):
        raise ValueError("token id outside vocabulary")
    if pad is not None and any(token == pad for token in all_ids):
        raise ValueError("pad token id used as document/query content is not supported")
    tail = decode_length - 1
    monolithic_total = sum(len(doc) for doc in docs) + len(query_ids) + tail
    canonical_total = max(len(doc) for doc in docs) + len(query_ids) + tail
    limit = int(config.max_position_embeddings)
    if monolithic_total > limit or canonical_total > limit:
        raise ValueError("request exceeds supported position range")


def _monolithic_row(
    model: Any,
    docs: list[torch.Tensor],
    query: torch.Tensor,
    prefix: torch.Tensor | None = None,
    *,
    variant: str,
) -> torch.Tensor:
    """Last-position logits [1, V] of the concatenated docs+query+prefix stream."""
    if prefix is None:
        prefix = torch.empty(1, 0, dtype=torch.long)
    lengths = [int(doc.shape[1]) for doc in docs]
    query_len = int(query.shape[1])
    answer_len = int(prefix.shape[1])
    tokens = torch.cat([*docs, query, prefix], dim=1).to(model.device)
    positions = build_positions(lengths, query_len, answer_len, "restart").unsqueeze(0)
    additive = to_additive(build_mask(lengths, query_len, answer_len, variant), model.dtype)
    with torch.no_grad():
        out = model(
            input_ids=tokens,
            attention_mask=additive.to(model.device)[None, None],
            position_ids=positions.to(model.device),
            use_cache=False,
        )
    return out.logits[:, -1, :]


def _mono_argmax(
    model: Any,
    docs: list[torch.Tensor],
    query: torch.Tensor,
    prefix_values: list[int],
    variant: str,
) -> int:
    row = _monolithic_row(model, docs, query, torch.tensor([prefix_values], dtype=torch.long),
                          variant=variant)
    return int(row[0].argmax())


def _canonical_argmax(
    cache: CanonicalCache,
    docs: list[torch.Tensor],
    query_values: list[int],
    prefix_values: list[int],
) -> int:
    ids = torch.tensor([[*query_values, *prefix_values]], dtype=torch.long)
    out = cache.request(docs, ids)
    return int(out.logits[0, -1].argmax())


def _teacher_chain(
    model_a: Any,
    docs: list[torch.Tensor],
    query: torch.Tensor,
    decode_length: int,
) -> list[int]:
    """A's greedy chain: t1 = A prefill argmax, then argmax given the chain so far."""
    row = _monolithic_row(model_a, docs, query, variant="ddi_qf")
    chain = [int(row[0].argmax())]
    for _step in range(1, decode_length):
        prefix = torch.tensor([chain], dtype=torch.long)
        row = _monolithic_row(model_a, docs, query, prefix, variant="ddi_qf")
        chain.append(int(row[0].argmax()))
    return chain


def _decode_arm(
    step_fn: Callable[[list[int]], int], chain: list[int]
) -> tuple[list[int], int | None]:
    """Teacher-forced decode: decode_ids[0] is the consumed chain[0]; each later entry is
    the arm's own argmax given query + the chain prefix consumed so far.  The returned
    divergence is the 0-based index of the first proposal differing from the chain."""
    decode_ids = [chain[0]]
    divergence = None
    prefix = [chain[0]]
    for step in range(1, len(chain)):
        proposal = int(step_fn(prefix))
        decode_ids.append(proposal)
        if divergence is None and proposal != chain[step]:
            divergence = step
        prefix.append(chain[step])
    return decode_ids, divergence


def _timed(
    fn: Callable[[], Any],
    repeats: int,
    device: torch.device,
    prepare: Callable[[int], None] | None = None,
) -> tuple[list[float], Any]:
    times: list[float] = []
    first: Any = None
    for index in range(repeats):
        if prepare is not None:
            prepare(index)
        _sync(device)
        started = time.perf_counter()
        result = fn()
        _sync(device)
        times.append(time.perf_counter() - started)
        if index == 0:
            first = result
    return times, first


def _record(
    arm: str,
    scenario: str,
    identity: str,
    row: torch.Tensor,
    times: list[float],
    decode_ids: list[int],
    divergence: int | None,
    hit_state: list[bool],
    cache_bytes: int,
    device: torch.device,
) -> dict[str, Any]:
    allocated, reserved = _peak_bytes(device)
    return {
        "arm": arm,
        "scenario": scenario,
        "arm_identity": identity,
        "backend": ARMS[arm]["backend"],
        "execution": ARMS[arm]["execution"],
        "cache": ARMS[arm]["cache"],
        "logits_sha256": _logits_digest(row),
        "logits_shape": list(row.shape),
        "logits_dtype": str(row.dtype),
        "prefill_argmax": int(row[0].argmax()),
        "decode_ids": decode_ids,
        "decode_divergence_step": divergence,
        "hit_state": hit_state,
        "ttft_seconds": times,
        "ttft_median_seconds": float(statistics.median(times)),
        "cache_bytes": int(cache_bytes),
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
    }


def _contract(model_bc: Any, checkpoint: Path) -> ExecutionContract:
    _mode, digest = _checkpoint_identity(checkpoint)
    return ExecutionContract(
        weights_identity=digest,
        adapters_identity="none",
        tokenizer_identity=digest,
        model_config_identity=model_config_identity(model_bc),
        runtime_identity=runtime_identity(model_bc),
        backend="eager",
        dtype=str(model_bc.dtype),
    )


def _load_models(checkpoint: Path, device: str) -> tuple[Any, Any]:
    dtype = "bf16" if device == "cuda" else "fp32"
    model_a, _ = load(str(checkpoint), dtype=dtype, backend="sdpa")
    model_a = model_a.to(device).eval()
    model_bc, _ = load(str(checkpoint), dtype=dtype, backend="eager")
    model_bc = model_bc.to(device).eval()
    return model_a, model_bc


def _check_models(model_a: Any, model_bc: Any) -> None:
    if str(model_a.dtype) != str(model_bc.dtype):
        raise ValueError("arms A/D and B/C must share one dtype")
    for name, model in (("A", model_a), ("B/C", model_bc)):
        if any(module.training for module in model.modules()):
            raise ValueError(f"model {name} must be fully eval()")


def _build_manifest(
    args: argparse.Namespace,
    inputs: dict[str, Any],
    device: torch.device,
    dtype_str: str,
    identities: dict[str, str],
) -> dict[str, Any]:
    mode, checkpoint_digest = _checkpoint_identity(args.checkpoint)
    gpu_name = torch.cuda.get_device_name() if device.type == "cuda" else None
    return {
        "schema_version": SCHEMA_VERSION,
        "data_role": "development",
        "runner": "scripts/run_canonical_matrix.py",
        "created_utc": datetime.now(UTC).isoformat(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_identity": {"mode": mode, "sha256": checkpoint_digest},
        "arms": {
            arm: {
                "backend": ARMS[arm]["backend"],
                "execution": ARMS[arm]["execution"],
                "cache": ARMS[arm]["cache"],
                "dtype": dtype_str,
                "layout": "restart",
                "arm_identity": identities[arm],
            }
            for arm in ARMS
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": args.device,
            "gpu_name": gpu_name,
            "torch_num_threads": torch.get_num_threads(),
        },
        "provenance": {
            "source_files": _source_hashes(),
            "git_base_sha256": _git_text("rev-parse", "HEAD"),
            "git_diff_head_sha256": hashlib.sha256(_git_bytes("diff", "HEAD")).hexdigest(),
            "git_diff_head_command": "git diff HEAD",
        },
        "inputs": {
            "inputs_path": str(Path(args.inputs).resolve()),
            "inputs_file_sha256": _sha_file(args.inputs),
            "seed": inputs["seed"],
            "request_count": 1,
            "doc_lengths": [len(doc) for doc in inputs["docs"]],
            "query_len": len(inputs["query_ids"]),
            "decode_length": args.decode_length,
            "repeats": args.repeats,
        },
        "notes": {
            "arm_a": (
                "monolithic concatenated prefill over docs+query with restart positions "
                "and the ddi_qf additive mask; use_cache=False"
            ),
            "arm_b": "CanonicalCache(store=None): fresh per-request encode, nothing persisted",
            "arm_c": (
                "CanonicalCache(store=DocCache): scenario 'cold' clears the store before "
                "every timed repeat; scenario 'all_hit' prepopulates the store untimed "
                "before the timed repeats"
            ),
            "arm_d": (
                "full-causal reference: the same concatenated token stream and restart "
                "positions as A with build_mask variant 'full'"
            ),
            "decode": (
                "teacher-forced on A's greedy chain: decode_ids[0] is A's prefill argmax "
                "and is consumed by every arm; later entries are the arm's own argmax "
                "given the chain prefix consumed so far; decode_divergence_step is the "
                "0-based index of the first proposal that differs from the chain"
            ),
            "ttft": (
                "wall seconds of the prefill execution only, torch.cuda.synchronize "
                "before and after on cuda"
            ),
            "peaks": (
                "torch.cuda peak allocated/reserved cover the timed repeats only; null off cuda"
            ),
            "token_digests": "sha256 over canonical JSON of token id arrays",
            "data_role": (
                "development observations only: no confirmation, QA, or cross-backend "
                "performance claim"
            ),
        },
    }


def run_matrix(
    model_a: Any,
    model_bc: Any,
    contract: ExecutionContract,
    identities: dict[str, str],
    spec: dict[str, Any],
    output: Path,
    *,
    request_index: int,
    seed: int,
    decode_length: int,
    repeats: int,
    device: torch.device,
) -> dict[str, Any]:
    """Execute all arms for one request and return its cell record."""
    docs_values = spec["docs"]
    query_values = spec["query_ids"]
    _validate_ids(model_bc.config, docs_values, query_values, decode_length)
    docs = [torch.tensor([doc], dtype=torch.long) for doc in docs_values]
    query = torch.tensor([query_values], dtype=torch.long)

    chain = _teacher_chain(model_a, docs, query, decode_length)
    no_hits = [False] * repeats
    arms: dict[str, Any] = {}

    _reset_peak(device)
    times, row = _timed(
        lambda: _monolithic_row(model_a, docs, query, variant="ddi_qf"), repeats, device
    )
    arms["A"] = _record(
        "A", "standard", identities["A"], row, times, chain, None, no_hits, 0, device
    )

    _reset_peak(device)
    times, row = _timed(
        lambda: _monolithic_row(model_a, docs, query, variant="full"), repeats, device
    )
    decode_ids, divergence = _decode_arm(
        lambda prefix: _mono_argmax(model_a, docs, query, prefix, "full"), chain
    )
    arms["D"] = _record(
        "D", "standard", identities["D"], row, times, decode_ids, divergence, no_hits, 0, device
    )

    fresh = CanonicalCache(model_bc, contract)
    _reset_peak(device)
    times, out = _timed(lambda: fresh.request(docs, query), repeats, device)
    decode_ids, divergence = _decode_arm(
        lambda prefix: _canonical_argmax(fresh, docs, query_values, prefix), chain
    )
    arms["B"] = _record(
        "B", "standard", identities["B"], out.logits[:, -1, :], times, decode_ids,
        divergence, no_hits, 0, device,
    )

    cold_dir = output / "canonical_cache" / "cold"
    cached_cold = CanonicalCache(model_bc, contract, DocCache(cold_dir))
    _reset_peak(device)
    times, out = _timed(
        lambda: cached_cold.request(docs, query),
        repeats,
        device,
        prepare=lambda _index: _clear_store(cold_dir),
    )
    cold_bytes = _store_bytes(cold_dir)
    _clear_store(cold_dir)
    decode_ids, divergence = _decode_arm(
        lambda prefix: _canonical_argmax(cached_cold, docs, query_values, prefix), chain
    )
    cold_record = _record(
        "C", "cold", identities["C"], out.logits[:, -1, :], times, decode_ids,
        divergence, no_hits, cold_bytes, device,
    )

    all_hit_dir = output / "canonical_cache" / "all_hit"
    cached_hit = CanonicalCache(model_bc, contract, DocCache(all_hit_dir))
    cached_hit.documents(docs)
    _reset_peak(device)
    times, out = _timed(lambda: cached_hit.request(docs, query), repeats, device)
    hit_bytes = _store_bytes(all_hit_dir)
    decode_ids, divergence = _decode_arm(
        lambda prefix: _canonical_argmax(cached_hit, docs, query_values, prefix), chain
    )
    hit_record = _record(
        "C", "all_hit", identities["C"], out.logits[:, -1, :], times, decode_ids,
        divergence, [True] * repeats, hit_bytes, device,
    )
    arms["C"] = {"cold": cold_record, "all_hit": hit_record}

    b_record = arms["B"]
    return {
        "seed": seed,
        "request_index": request_index,
        "input_sha256": _digest_json({"docs": docs_values, "query_ids": query_values}),
        "doc_token_sha256": [_digest_json(doc) for doc in docs_values],
        "query_token_sha256": _digest_json(query_values),
        "doc_lengths": [len(doc) for doc in docs_values],
        "query_len": len(query_values),
        "decode_length": decode_length,
        "repeats": repeats,
        "teacher_token_ids": chain,
        "arms": arms,
        "b_vs_c_equal": {
            "cold_logits": b_record["logits_sha256"] == cold_record["logits_sha256"],
            "all_hit_logits": b_record["logits_sha256"] == hit_record["logits_sha256"],
            "cold_decode": b_record["decode_ids"] == cold_record["decode_ids"],
            "all_hit_decode": b_record["decode_ids"] == hit_record["decode_ids"],
        },
        "a_vs_b_equal_logits": arms["A"]["logits_sha256"] == b_record["logits_sha256"],
    }


def _summarize(cells: list[dict[str, Any]], manifest_sha256: str) -> dict[str, Any]:
    per_request: list[dict[str, Any]] = []
    ttft: dict[str, list[float]] = {key: [] for key in ("A", "B", "C_cold", "C_all_hit", "D")}
    agreement = {
        key: {"prefill_matches": 0, "decode_step_matches": 0, "decode_steps": 0}
        for key in ("B", "C_cold", "C_all_hit", "D")
    }
    b_vs_c_all = True
    for cell in cells:
        arms = cell["arms"]
        anchor = arms["A"]
        pairs = (
            ("A", anchor),
            ("B", arms["B"]),
            ("C_cold", arms["C"]["cold"]),
            ("C_all_hit", arms["C"]["all_hit"]),
            ("D", arms["D"]),
        )
        per_request.append(
            {
                "request_index": cell["request_index"],
                "input_sha256": cell["input_sha256"],
                "ttft_median_seconds": {
                    key: record["ttft_median_seconds"] for key, record in pairs
                },
                "b_vs_c_equal": cell["b_vs_c_equal"],
                "a_vs_b_equal_logits": cell["a_vs_b_equal_logits"],
            }
        )
        for key, record in pairs:
            ttft[key].append(record["ttft_median_seconds"])
        b_vs_c_all = b_vs_c_all and all(cell["b_vs_c_equal"].values())
        for key, record in pairs[1:]:
            agreement[key]["prefill_matches"] += int(
                record["prefill_argmax"] == anchor["prefill_argmax"]
            )
            agreement[key]["decode_step_matches"] += sum(
                int(proposal == teacher)
                for proposal, teacher in zip(
                    record["decode_ids"], cell["teacher_token_ids"], strict=True
                )
            )
            agreement[key]["decode_steps"] += len(cell["teacher_token_ids"])
    return {
        "schema_version": SCHEMA_VERSION,
        "data_role": "development",
        "manifest_sha256": manifest_sha256,
        "requests": per_request,
        "aggregate": {
            "ttft_median_seconds": {
                key: float(statistics.median(values)) for key, values in ttft.items() if values
            },
            "b_vs_c_equal_all": b_vs_c_all,
            "argmax_agreement": agreement,
        },
        "timing_caveat": (
            "TTFT medians are development observations; cross-backend A/B performance "
            "comparisons are inadmissible per scripts.validate_canonical_matrix.validate_comparison"
        ),
    }


def _prepare_output(raw: Path) -> Path:
    output = Path(raw)
    if output.exists():
        raise FileExistsError(f"--output must not already exist: {output}")
    output.mkdir(parents=True)
    return output


def main(
    argv: list[str] | None = None, *, load_models: LoadModels | None = None
) -> dict[str, Any]:
    args = _parse_args(argv)
    inputs = _load_inputs(args.inputs)
    output = _prepare_output(args.output)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
    torch.manual_seed(inputs["seed"])
    if load_models is None:
        load_models = _load_models
    model_a, model_bc = load_models(args.checkpoint, args.device)
    _check_models(model_a, model_bc)
    contract = _contract(model_bc, args.checkpoint)
    dtype_str = str(model_bc.dtype)
    identities = {arm: _arm_identity(arm, dtype_str) for arm in ARMS}
    manifest = _build_manifest(args, inputs, device, dtype_str, identities)
    _write_json(output / "manifest.json", manifest)

    specs = [{"docs": inputs["docs"], "query_ids": inputs["query_ids"]}]
    cells: list[dict[str, Any]] = []
    with (output / "cells.jsonl").open("x", encoding="utf-8") as handle:
        for index, spec in enumerate(specs):
            cell = run_matrix(
                model_a,
                model_bc,
                contract,
                identities,
                spec,
                output,
                request_index=index,
                seed=inputs["seed"],
                decode_length=args.decode_length,
                repeats=args.repeats,
                device=device,
            )
            handle.write(
                json.dumps(cell, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            )
            handle.flush()
            cells.append(cell)

    summary = _summarize(cells, _sha_file(output / "manifest.json"))
    _write_json(output / "result.json", summary)
    print(
        json.dumps(
            {
                "output": str(output),
                "requests": len(cells),
                "b_vs_c_equal_all": summary["aggregate"]["b_vs_c_equal_all"],
            }
        )
    )
    return summary


if __name__ == "__main__":
    main()
