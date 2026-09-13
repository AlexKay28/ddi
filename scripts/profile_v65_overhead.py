"""V65 step 1: localize canonical-DDI prefill overhead vs the monolithic SDPA baseline.

Profiling-only harness (no fixes): three arms over the V60 confirmation workload
(the 24 frozen HotpotQA record ids with their real re-tokenized documents, shapes
verified against the frozen ``requests.jsonl``), one shared SDPA model, mirroring
the V60 ``--canonical-on-sdpa`` execution:

- ``monolithic_sdpa`` — arm-A-like plain SDPA additive-mask prefill with
  ``generate_ids`` prefill semantics (mask build excluded from the timed wall,
  one model call timed between synchronizes).
- ``fresh_canonical`` — B-like ``CanonicalCache(store=None).request(...)``.
- ``all_hit_reuse``   — C-like ``CanonicalCache(DocCache store).request(...)``
  with the store pre-populated, so every timed pass is an all-hit reuse.

Protocol per task: seed 47, 2 warmups + 10 synchronized repeats per (record, arm).
Per arm and record this harness records: total prefill wall (plus a no-final-sync
CPU wall), hook-timed CPU wall inside attention / mlp / embedding / lm_head /
final-norm modules, sub-phase CPU walls inside the canonical path (contract
re-check, id validation, token digests, disk load, KV validation, cache assembly,
mask/position building, encode model calls), and one ``torch.profiler`` pass per
(record, arm) for CUDA kernel totals, kernel launch counts, top-10 kernels by CUDA
time and ``aten::`` op counts (tensor allocs/copies).

The instrumented ``ProfiledCanonicalCache`` overrides reproduce the bodies of
``CanonicalCache.encode_document`` / ``CanonicalCache.documents`` verbatim with
phase timers added; the manifest records the sha256 of the profiled sources and
this script refuses to run if ``ddi/canonical_cache.py`` drifts from the sha the
V60 confirmation ran with. Outputs ``manifest.json`` / ``result.json`` /
``summary.md`` under the output directory; nothing else is written.
"""

from __future__ import annotations

import argparse
import functools
import json
import platform
import statistics
import subprocess
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from ddi.cache import CacheEntry, DocCache
from ddi.canonical_cache import CanonicalCache
from ddi.inference import load
from ddi.masks import build_mask, to_additive
from scripts.run_canonical_matrix import _checkpoint_identity, _contract, _sha_file

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 65
SEED = 47
WARMUPS = 2
REPEATS = 10
DEFAULT_CHECKPOINT = ROOT / "results/checkpoints/Qwen__Qwen2.5-7B-Instruct__ddi_qf__8L_restart"
DEFAULT_REQUESTS = ROOT / "results/v60_confirmation24_20260910T093500Z/requests.jsonl"
DEFAULT_SOURCE_MANIFEST = ROOT / "experiments/v60_confirmation_manifest.json"
V64_CANONICAL_CACHE_SHA256 = "ce5a93b2b5132f4fe6ff36f4e4889b25cc377a468a70a260f7319175c875e756"
ATTENTION_KERNEL_PATTERNS = ("flash", "fmha", "attention", "sdpa")
ARM_KEYS = ("monolithic_sdpa", "fresh_canonical", "all_hit_reuse")
MODULE_GROUPS = ("attention", "mlp", "embedding", "lm_head", "final_norm")
OP_SUBSET = (
    "aten::empty",
    "aten::empty_like",
    "aten::zero_",
    "aten::copy_",
    "aten::to",
    "aten::contiguous",
    "aten::clone",
    "aten::cat",
    "aten::isfinite",
    "aten::any",
    "aten::eq",
    "aten::lt",
    "aten::arange",
    "aten::item",
    "aten::_local_scalar_dense",
    "aten::slice",
    "aten::select",
    "aten::detach",
)
PHASE_REFERENCES = {
    "check_model": "CanonicalCache._check_model (ddi/canonical_cache.py:119-145)",
    "ids_validation": "CanonicalCache._ids (ddi/canonical_cache.py:147-165)",
    "metadata_digest": "CanonicalCache._metadata sha256 digest (ddi/canonical_cache.py:167-178)",
    "store_lookup": "CanonicalCache.documents path.exists (ddi/canonical_cache.py:246-247)",
    "disk_load": "CanonicalCache.documents torch.load + check (ddi/canonical_cache.py:250-259)",
    "validate_entry": "CanonicalCache._validate geometry/isfinite (ddi/canonical_cache.py:180-204)",
    "cache_assemble": "assemble_dynamic_cache per-entry to(device) + cat (ddi/cache.py:164-198)",
    "query_mask_build": "forward_query_with_doc_cache build_mask (ddi/cache.py:216)",
    "query_positions_build": "forward_query_with_doc_cache build_positions (ddi/cache.py:211-212)",
    "query_to_additive": "forward_query_with_doc_cache to_additive (ddi/cache.py:217-222)",
    "encode_mask_build": "CanonicalCache.encode_document mask (ddi/canonical_cache.py:211-212)",
    "encode_model_call": "CanonicalCache.encode_document model() (ddi/canonical_cache.py:213-220)",
    "encode_kv_extract": "encode_document KV detach/contiguous (ddi/canonical_cache.py:221-231)",
    "store_put_cpu_copy": "documents store.put CPU copies (ddi/canonical_cache.py:262-268)",
    "attention_modules_cpu": "Qwen2Attention forward-hook CPU wall inside modules",
    "mlp_modules_cpu": "Qwen2MLP forward-hook CPU wall inside modules",
    "embedding_modules_cpu": "Qwen2Embedding forward-hook CPU wall inside module",
    "lm_head_modules_cpu": "lm_head forward-hook CPU wall inside module",
    "final_norm_modules_cpu": "Qwen2RMSNorm final forward-hook CPU wall inside module",
    "query_forward_glue": "HF Qwen2Model.forward python glue outside hooked modules (residual)",
    "wrapper_glue_residual": "request/documents python glue not attributed (residual)",
}
V64_TARGETS = {
    "A_monolithic_median_ms": 124.9597,
    "C_all_hit_median_ms": 221.2411,
    "C_over_A": 1.7705,
    "source": "results/v64_audit_20260910T091755Z_TIMING_CLAIMS/CLAIMS_TABLE.md",
}

_ACTIVE: dict[str, Any] = {"trace": None}


class PhaseTrace:
    """Accumulates named CPU-wall phases (seconds + call counts) for one repeat."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, name: str, seconds: float) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + seconds
        self.counts[name] = self.counts.get(name, 0) + 1


@contextmanager
def _phase(name: str):
    trace = _ACTIVE["trace"]
    started = time.perf_counter()
    try:
        yield
    finally:
        if trace is not None:
            trace.add(name, time.perf_counter() - started)


class GroupedModuleTimer:
    """Accumulates CPU wall time inside registered module calls via forward hooks.

    Same semantics as the V61 ``ModuleCallTimer``, extended to named groups
    (attention / mlp / embedding / lm_head / final_norm). No synchronization is
    performed, so this measures the CPU-side duration of each module ``__call__``
    (python wrapper + kernel launches), never GPU completion.
    """

    def __init__(self, groups: dict[str, list[torch.nn.Module]]) -> None:
        self.groups = list(groups)
        self.seconds: dict[str, float] = {name: 0.0 for name in groups}
        self.calls: dict[str, int] = {name: 0 for name in groups}
        self._enter: dict[torch.nn.Module, float] = {}
        self._handles: list[Any] = []
        for name, modules in groups.items():
            for module in modules:
                self._handles.append(module.register_forward_pre_hook(self._make_enter(module)))
                self._handles.append(module.register_forward_hook(self._make_exit(name, module)))

    def _make_enter(self, module: torch.nn.Module):
        def hook(mod: torch.nn.Module, args: tuple) -> None:  # noqa: ARG001
            self._enter[module] = time.perf_counter()

        return hook

    def _make_exit(self, name: str, module: torch.nn.Module):
        def hook(mod: torch.nn.Module, args: tuple, output: Any) -> Any:  # noqa: ARG001
            started = self._enter.pop(module, None)
            if started is not None:
                self.seconds[name] += time.perf_counter() - started
                self.calls[name] += 1
            return output

        return hook

    def reset(self) -> None:
        self.seconds = {name: 0.0 for name in self.groups}
        self.calls = {name: 0 for name in self.groups}

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class ProfiledCanonicalCache(CanonicalCache):
    """CanonicalCache with phase timers; bodies mirror the base class verbatim."""

    def _check_model(self) -> None:
        with _phase("check_model"):
            return super()._check_model()

    def _ids(self, ids: torch.Tensor) -> torch.Tensor:
        with _phase("ids_validation"):
            return super()._ids(ids)

    def _metadata(self, ids: torch.Tensor) -> dict[str, Any]:
        with _phase("metadata_digest"):
            return super()._metadata(ids)

    def _validate(self, entry: CacheEntry, meta: dict[str, Any], device: torch.device) -> None:
        with _phase("validate_entry"):
            return super()._validate(entry, meta, device)

    def encode_document(self, token_ids: torch.Tensor) -> CacheEntry:
        self._check_model()
        ids = self._ids(token_ids)
        length = ids.shape[1]
        slots = torch.arange(length, device=self.model.device)
        with _phase("encode_mask_build"):
            mask = to_additive(build_mask([length], 0, 0, "full"), self.model.dtype)
        with torch.no_grad(), _phase("encode_model_call"):
            out = self.model(
                input_ids=ids.to(self.model.device),
                attention_mask=mask.to(self.model.device)[None, None],
                position_ids=slots[None],
                cache_position=slots,
                use_cache=True,
            )
        with _phase("encode_kv_extract"):
            past = out.past_key_values
            if hasattr(past, "key_cache"):
                layers = list(zip(past.key_cache, past.value_cache, strict=True))
            else:
                layers = list(past)
            meta = self._metadata(ids)
            entry = CacheEntry(
                **meta,
                k_per_layer=[k.detach().contiguous() for k, _ in layers],
                v_per_layer=[v.detach().contiguous() for _, v in layers],
            )
        self._validate(entry, meta, self.model.device)
        return entry

    def documents(self, documents: list[torch.Tensor]) -> list[CacheEntry]:
        self._check_model()
        if not isinstance(documents, list) or not documents:
            raise ValueError("at least one document is required")
        ids_list = [self._ids(ids) for ids in documents]
        entries = []
        for ids in ids_list:
            meta = self._metadata(ids)
            entry = None
            if self.store is not None:
                with _phase("store_lookup"):
                    path = self.store.cache_dir / f"{meta['key']}.pt"
                    hit = path.exists()
                if hit:
                    with _phase("disk_load"):
                        blob = torch.load(path, map_location="cpu", weights_only=True)
                        if (
                            not isinstance(blob, dict) or set(blob) != {"k", "v", "meta"}
                            or not isinstance(blob["meta"], dict) or set(blob["meta"]) != set(meta)
                        ):
                            raise ValueError("malformed canonical payload schema")
                        entry = CacheEntry(
                            **blob["meta"], k_per_layer=blob["k"], v_per_layer=blob["v"]
                        )
                    self._validate(entry, meta, torch.device("cpu"))
            if entry is None:
                entry = self.encode_document(ids)
                if self.store is not None:
                    with _phase("store_put_cpu_copy"):
                        self.store.put(replace(
                            entry,
                            k_per_layer=[
                                t.detach().to("cpu", copy=True) for t in entry.k_per_layer
                            ],
                            v_per_layer=[
                                t.detach().to("cpu", copy=True) for t in entry.v_per_layer
                            ],
                        ))
            entries.append(entry)
        return entries


def _is_cuda_event(event: Any) -> bool:
    return "CUDA" in str(getattr(event, "device_type", "")).upper()


def _event_micros(event: Any) -> float:
    micros = getattr(event, "self_device_time_total", None)
    if micros is None:
        micros = getattr(event, "self_cuda_time_total", 0.0)
    return float(micros or 0.0)


def classify_attention_kernel(
    name: str, patterns: tuple[str, ...] = ATTENTION_KERNEL_PATTERNS
) -> bool:
    """True when a CUDA kernel name looks like an attention kernel."""
    lowered = name.lower()
    return any(pattern in lowered for pattern in patterns)


def summarize_kernel_events(
    events: Any, patterns: tuple[str, ...] = ATTENTION_KERNEL_PATTERNS
) -> dict[str, Any]:
    """Parse profiler events into kernel totals, launch counts and per-name sums.

    Only CUDA events are considered; ``Memcpy`` transfers are reported separately
    from kernel launches. ``by_name`` maps kernel name to {count, total_us}.
    """
    kernel_ms = 0.0
    attention_ms = 0.0
    memcpy_ms = 0.0
    launches = 0
    memcpy_count = 0
    by_name: dict[str, dict[str, float]] = {}
    for event in events:
        if not _is_cuda_event(event):
            continue
        micros = _event_micros(event)
        name = str(getattr(event, "name", "") or "")
        if name.startswith("Memcpy"):
            memcpy_ms += micros / 1e3
            memcpy_count += 1
            continue
        if micros <= 0.0:
            continue
        kernel_ms += micros / 1e3
        launches += 1
        if classify_attention_kernel(name, patterns):
            attention_ms += micros / 1e3
        entry = by_name.setdefault(name, {"count": 0, "total_us": 0.0})
        entry["count"] += 1
        entry["total_us"] += micros
    return {
        "kernel_ms": kernel_ms,
        "launches": launches,
        "attention_kernel_ms": attention_ms,
        "memcpy_ms": memcpy_ms,
        "memcpy_count": memcpy_count,
        "by_name": by_name,
    }


def top_kernels(by_name: dict[str, dict[str, float]], k: int = 10) -> list[dict[str, Any]]:
    """Top-k kernel names by accumulated CUDA time, descending (pure)."""
    rows = [
        {"name": name, "count": int(stats["count"]), "total_us": stats["total_us"]}
        for name, stats in by_name.items()
    ]
    rows.sort(key=lambda row: (-row["total_us"], row["name"]))
    return rows[:k]


def aggregate_op_counts(events: Any) -> dict[str, int]:
    """Count CPU-side ``aten::`` operator invocations across profiler events (pure)."""
    counts: Counter[str] = Counter()
    for event in events:
        if _is_cuda_event(event):
            continue
        name = str(getattr(event, "name", "") or "")
        if name.startswith("aten::"):
            counts[name] += max(1, int(getattr(event, "count", 1) or 1))
    return dict(counts)


def decompose_gap(arm_a: dict[str, Any], arm_c: dict[str, Any]) -> dict[str, Any]:
    """Pure C-like minus A-like gap arithmetic with sorted canonical contributors.

    Each arm dict needs ``wall_ms``, ``cpu_wall_ms`` and a ``cuda`` dict with
    ``kernel_ms``; the C-like arm may carry a ``phase_ms`` dict whose entries are
    reported as contributors sorted by descending milliseconds.
    """
    gap_wall = arm_c["wall_ms"] - arm_a["wall_ms"]
    gap_cpu = arm_c["cpu_wall_ms"] - arm_a["cpu_wall_ms"]
    gpu_tail_a = arm_a["wall_ms"] - arm_a["cpu_wall_ms"]
    gpu_tail_c = arm_c["wall_ms"] - arm_c["cpu_wall_ms"]
    kernel_a = float(arm_a.get("cuda", {}).get("kernel_ms", 0.0))
    kernel_c = float(arm_c.get("cuda", {}).get("kernel_ms", 0.0))
    contributors = sorted(
        ((name, float(ms)) for name, ms in arm_c.get("phase_ms", {}).items()),
        key=lambda item: (-item[1], item[0]),
    )
    return {
        "gap_wall_ms": gap_wall,
        "gap_cpu_wall_ms": gap_cpu,
        "gap_gpu_tail_ms": gpu_tail_c - gpu_tail_a,
        "gap_cuda_kernel_ms": kernel_c - kernel_a,
        "contributors_ms": [{"phase": name, "ms": ms} for name, ms in contributors],
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _agg(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "min": ordered[0],
        "p25": ordered[max(0, int(0.25 * (len(ordered) - 1)))],
        "p75": ordered[int(0.75 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def _write_json(path: Path, value: Any) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _install_phase_wrappers() -> Any:
    """Phase-time the module-level helpers the canonical query forward calls."""
    import ddi.cache as cache_mod
    import ddi.canonical_cache as canonical_mod

    originals: list[tuple[Any, str, Any]] = []

    def wrap(module: Any, attr: str, name: str) -> None:
        original = getattr(module, attr)
        originals.append((module, attr, original))

        @functools.wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with _phase(name):
                return original(*args, **kwargs)

        setattr(module, attr, wrapper)

    wrap(cache_mod, "assemble_dynamic_cache", "cache_assemble")
    wrap(cache_mod, "build_positions", "query_positions_build")
    wrap(cache_mod, "build_mask", "query_mask_build")
    wrap(cache_mod, "to_additive", "query_to_additive")
    wrap(canonical_mod, "forward_query_with_doc_cache", "query_forward_total")

    def restore() -> None:
        for module, attr, original in originals:
            setattr(module, attr, original)

    return restore


def _module_groups(model: Any) -> dict[str, list[torch.nn.Module]]:
    return {
        "attention": [layer.self_attn for layer in model.model.layers],
        "mlp": [layer.mlp for layer in model.model.layers],
        "embedding": [model.model.embed_tokens],
        "lm_head": [model.lm_head],
        "final_norm": [model.model.norm],
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--requests", type=Path, default=DEFAULT_REQUESTS)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--record-limit", type=int, default=None)
    parser.add_argument("--warmups", type=int, default=WARMUPS)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    if args.warmups < 0 or args.repeats < 1:
        parser.error("--warmups must be >= 0 and --repeats >= 1")
    return args


def _load_frozen_requests(path: Path, limit: int | None) -> list[dict[str, Any]]:
    cells = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if limit is not None:
        cells = cells[:limit]
    if not cells:
        raise ValueError("no frozen request cells found")
    return cells


def _run_passes(
    pass_fn: Any,
    *,
    module_timer: GroupedModuleTimer,
    warmups: int,
    repeats: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Warm up, then run synchronized timed passes with per-repeat phase traces."""
    for _ in range(warmups):
        _ACTIVE["trace"] = None
        module_timer.reset()
        _sync(device)
        pass_fn()
        _sync(device)
    rows: list[dict[str, Any]] = []
    for _ in range(repeats):
        trace = PhaseTrace()
        _ACTIVE["trace"] = trace
        module_timer.reset()
        _sync(device)
        started = time.perf_counter()
        payload = pass_fn()
        cpu_wall = time.perf_counter() - started
        _sync(device)
        wall = time.perf_counter() - started
        _ACTIVE["trace"] = None
        token = int(payload["result"].logits[0, -1].argmax())
        rows.append({
            "wall_ms": wall * 1e3,
            "cpu_wall_ms": cpu_wall * 1e3,
            "extras": payload["extras"],
            "phases": dict(trace.totals),
            "phase_counts": dict(trace.counts),
            "modules": dict(module_timer.seconds),
            "module_calls": dict(module_timer.calls),
            "first_token": token,
        })
    return rows


def _profiler_pass(pass_fn: Any, device: torch.device) -> tuple[Any, Any]:
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities) as prof:
        payload = pass_fn()
        _sync(device)
    return payload, prof


def _per_record_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    walls = [row["wall_ms"] for row in rows]
    cpu_walls = [row["cpu_wall_ms"] for row in rows]
    phase_keys = sorted({key for row in rows for key in row["phases"]})
    module_keys = sorted({key for row in rows for key in row["modules"]})
    phases = {key: statistics.median([row["phases"].get(key, 0.0) for row in rows]) * 1e3
              for key in phase_keys}
    modules = {key: statistics.median([row["modules"].get(key, 0.0) for row in rows]) * 1e3
               for key in module_keys}
    extras: dict[str, Any] = {}
    for key in ("forward_wall_ms", "forward_cpu_wall_ms", "mask_wall_ms"):
        if all(key in row["extras"] for row in rows):
            extras[key] = statistics.median([row["extras"][key] for row in rows])
    return {
        "wall_ms": statistics.median(walls),
        "cpu_wall_ms": statistics.median(cpu_walls),
        "wall_repeats_ms": walls,
        "cpu_wall_repeats_ms": cpu_walls,
        "phase_ms": phases,
        "phase_counts_per_repeat": rows[0]["phase_counts"],
        "modules_ms": modules,
        "module_calls_per_repeat": rows[0]["module_calls"],
        "extras": extras,
        "first_token": rows[0]["first_token"],
        "first_token_all_repeats_equal": len({row["first_token"] for row in rows}) == 1,
    }


def _aggregate_arm(per_record: list[dict[str, Any]], num_layers: int) -> dict[str, Any]:
    walls = [record["wall_ms"] for record in per_record]
    cpu_walls = [record["cpu_wall_ms"] for record in per_record]
    gpu_tails = [w - c for w, c in zip(walls, cpu_walls, strict=True)]
    phase_keys = sorted({key for record in per_record for key in record["phase_ms"]})
    module_keys = sorted({key for record in per_record for key in record["modules_ms"]})
    phase_ms = {
        key: statistics.median([record["phase_ms"].get(key, 0.0) for record in per_record])
        for key in phase_keys
    }
    modules_ms = {
        key: statistics.median([record["modules_ms"].get(key, 0.0) for record in per_record])
        for key in module_keys
    }
    kernel_totals: dict[str, dict[str, float]] = {}
    op_totals: Counter[str] = Counter()
    kernel_ms = 0.0
    attention_kernel_ms = 0.0
    launches = 0
    memcpy_ms = 0.0
    passes = 0
    for record in per_record:
        cuda = record.get("cuda")
        if not cuda:
            continue
        passes += 1
        kernel_ms += cuda["kernel_ms"]
        attention_kernel_ms += cuda["attention_kernel_ms"]
        launches += cuda["launches"]
        memcpy_ms += cuda["memcpy_ms"]
        for name, stats in cuda["by_name"].items():
            entry = kernel_totals.setdefault(name, {"count": 0.0, "total_us": 0.0})
            entry["count"] += stats["count"]
            entry["total_us"] += stats["total_us"]
        op_totals.update(cuda["op_counts"])
    divisor = max(1, passes)
    cuda_summary: dict[str, Any] = {
        "profiled_passes": passes,
        "kernel_ms_per_pass": kernel_ms / divisor,
        "attention_kernel_ms_per_pass": attention_kernel_ms / divisor,
        "launches_per_pass": launches / divisor,
        "memcpy_ms_per_pass": memcpy_ms / divisor,
        "top_kernels": [
            {
                "name": row["name"],
                "count_per_pass": row["count"] / divisor,
                "total_ms_per_pass": row["total_us"] / 1e3 / divisor,
                "attention": classify_attention_kernel(row["name"]),
            }
            for row in top_kernels(kernel_totals, k=10)
        ],
        "op_counts_per_pass": {name: op_totals[name] / divisor for name in sorted(op_totals)},
        "op_subset_per_pass": {name: op_totals[name] / divisor for name in OP_SUBSET
                               if name in op_totals},
        "op_subset_per_layer_per_pass": {
            name: op_totals[name] / divisor / num_layers for name in OP_SUBSET
            if name in op_totals
        },
    }
    return {
        "wall_ms": _agg(walls),
        "cpu_wall_ms": _agg(cpu_walls),
        "gpu_tail_ms": _agg(gpu_tails),
        "phase_ms": phase_ms,
        "modules_ms": modules_ms,
        "cuda": cuda_summary,
    }


def _derived_glue(record: dict[str, Any]) -> None:
    """Split query-forward / wrapper residuals so contributors do not double count."""
    phases = record["phase_ms"]
    modules = record["modules_ms"]
    module_sum = sum(modules.get(key, 0.0) for key in MODULE_GROUPS)
    forward_parts = (
        phases.get("cache_assemble", 0.0)
        + phases.get("query_mask_build", 0.0)
        + phases.get("query_positions_build", 0.0)
        + phases.get("query_to_additive", 0.0)
    )
    if "query_forward_total" in phases:
        phases["query_forward_glue"] = max(
            0.0, phases["query_forward_total"] - forward_parts - module_sum
        )
        leaves = (
            sum(phases.get(key, 0.0) for key in (
                "check_model",
                "ids_validation",
                "metadata_digest",
                "store_lookup",
                "disk_load",
                "store_put_cpu_copy",
                "validate_entry",
                "encode_mask_build",
                "encode_model_call",
                "encode_kv_extract",
            ))
            + phases["query_forward_total"]
        )
        phases["wrapper_glue_residual"] = max(0.0, record["cpu_wall_ms"] - leaves)
    if "forward_cpu_wall_ms" in record["extras"]:
        record.setdefault("derived", {})["monolithic_model_glue_ms"] = max(
            0.0, record["extras"]["forward_cpu_wall_ms"] - module_sum
        )


def _summary_markdown(payload: dict[str, Any]) -> str:
    arms = payload["arms"]
    gap = payload["gap_decomposition"]
    lines = [
        "# V65 step 1 — canonical-path overhead profile vs monolithic SDPA baseline",
        "",
        f"- Created (UTC): {payload['created_utc']}",
        f"- Checkpoint: `{payload['checkpoint']}`",
        f"- Workload: V60 confirmation 24 frozen ids, real tokens "
        f"(`{payload['requests']['path']}`), shapes verified per record",
        f"- Protocol: seed {payload['protocol']['seed']}, warmups "
        f"{payload['protocol']['warmups']}, repeats {payload['protocol']['repeats']}, "
        f"synchronized; one SDPA model (V60 --canonical-on-sdpa execution)",
        f"- Corrected V64 targets: A ~{V64_TARGETS['A_monolithic_median_ms']} ms, "
        f"C_all_hit ~{V64_TARGETS['C_all_hit_median_ms']} ms "
        f"(C/A ~{V64_TARGETS['C_over_A']}x)",
        "",
        "## Per-arm table (medians across 24 records of per-record repeat medians)",
        "",
        "| arm | like | wall_ms | cpu_wall_ms | gpu_tail_ms | cuda_kernel_ms/pass | "
        "launches/pass | attention_cpu_ms | non_attention_cpu_ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key, like in (("monolithic_sdpa", "A"), ("fresh_canonical", "B"),
                      ("all_hit_reuse", "C_all_hit")):
        arm = arms[key]
        attention = arm["modules_ms"].get("attention", 0.0)
        non_attention = arm["cpu_wall_ms"]["median"] - attention
        lines.append(
            f"| {key} | {like} | {arm['wall_ms']['median']:.2f} | "
            f"{arm['cpu_wall_ms']['median']:.2f} | {arm['gpu_tail_ms']['median']:.2f} | "
            f"{arm['cuda']['kernel_ms_per_pass']:.2f} | "
            f"{arm['cuda']['launches_per_pass']:.0f} | {attention:.2f} | "
            f"{non_attention:.2f} |"
        )
    lines += [
        "",
        "## C_all_hit − A gap decomposition (medians)",
        "",
        f"- wall gap: {gap['gap_wall_ms']:.2f} ms "
        f"(target ~{V64_TARGETS['C_all_hit_median_ms'] - V64_TARGETS['A_monolithic_median_ms']:.2f}"
        " ms)",
        f"- CPU-wall (python/launch) component: {gap['gap_cpu_wall_ms']:.2f} ms",
        f"- GPU-tail component: {gap['gap_gpu_tail_ms']:.2f} ms",
        f"- CUDA-kernel-time component (profiler): {gap['gap_cuda_kernel_ms']:.2f} ms",
        "",
        "Top canonical-path contributors (C arm phase medians):",
        "",
        "| rank | phase | ms | reference |",
        "|---|---|---|---|",
    ]
    for rank, item in enumerate(gap["contributors_ms"][:10], start=1):
        lines.append(
            f"| {rank} | {item['phase']} | {item['ms']:.2f} | "
            f"{PHASE_REFERENCES.get(item['phase'], '—')} |"
        )
    a_median = arms["monolithic_sdpa"]["wall_ms"]["median"]
    c_median = arms["all_hit_reuse"]["wall_ms"]["median"]
    lines += [
        "",
        "## Comparison against corrected V64 targets",
        "",
        "| arm | this run (ms) | V64 target (ms) | ratio |",
        "|---|---|---|---|",
        f"| monolithic_sdpa | {a_median:.4f} | {V64_TARGETS['A_monolithic_median_ms']} | "
        f"{a_median / V64_TARGETS['A_monolithic_median_ms']:.4f}x |",
        f"| all_hit_reuse | {c_median:.4f} | {V64_TARGETS['C_all_hit_median_ms']} | "
        f"{c_median / V64_TARGETS['C_all_hit_median_ms']:.4f}x |",
        "",
        "## Notes",
        "",
        "- Hook-timed module CPU walls carry the same ~2 hooks/module overhead in every arm; "
        "they measure CPU-side module duration (wrapper + launches), never GPU completion.",
        "- torch.profiler passes add overhead; kernel stats are per-pass means over the "
        "profiled passes and are not used for the wall table.",
        "- Profiled sources: ddi/canonical_cache.py sha256 "
        f"`{payload['sources']['ddi/canonical_cache.py']}` (must equal the V60 run sha "
        f"`{V64_CANONICAL_CACHE_SHA256[:16]}…`).",
    ]
    return "\n".join(lines) + "\n"


def _profile_record(
    cell: dict[str, Any],
    batch: Any,
    *,
    model: Any,
    contract: Any,
    cache_root: Path,
    module_timer: GroupedModuleTimer,
    warmups: int,
    repeats: int,
    device: torch.device,
) -> dict[str, Any]:
    docs, offset = [], 0
    for length in batch.doc_lengths:
        docs.append(batch.input_ids[0][offset:offset + length][None])
        offset += length
    query_ids = batch.input_ids[0][offset:][None]

    prefix_ids = batch.input_ids.to(device)
    positions = batch.position_ids.to(device)
    prefix_len = int(prefix_ids.shape[-1])
    dtype = model.dtype

    def pass_monolithic() -> dict[str, Any]:
        t_mask = time.perf_counter()
        with _phase("A_mask_build"):
            allow = build_mask(batch.doc_lengths, batch.query_len, 1, "ddi_qf")
        with _phase("A_to_additive"):
            additive = to_additive(allow, dtype)
        with _phase("A_mask_to_device"):
            additive = additive.to(device)
        with _phase("A_mask_slice_clone"):
            prefill_mask = additive[:prefix_len, :prefix_len].clone()[None, None]
        _sync(device)
        mask_wall = time.perf_counter() - t_mask
        t_fwd = time.perf_counter()
        out = model(
            input_ids=prefix_ids,
            attention_mask=prefill_mask,
            position_ids=positions,
            use_cache=True,
        )
        fwd_cpu = time.perf_counter() - t_fwd
        _sync(device)
        fwd_wall = time.perf_counter() - t_fwd
        return {
            "result": out,
            "extras": {
                "mask_wall_ms": mask_wall * 1e3,
                "forward_wall_ms": fwd_wall * 1e3,
                "forward_cpu_wall_ms": fwd_cpu * 1e3,
            },
        }

    fresh = ProfiledCanonicalCache(model, contract)
    hit_store = DocCache(cache_root)
    cached_hit = ProfiledCanonicalCache(model, contract, hit_store)
    _ACTIVE["trace"] = None
    cached_hit.documents([doc.cpu() for doc in docs])

    def pass_fresh() -> dict[str, Any]:
        return {"result": fresh.request(docs, query_ids), "extras": {}}

    def pass_hit() -> dict[str, Any]:
        return {"result": cached_hit.request(docs, query_ids), "extras": {}}

    record_arms: dict[str, dict[str, Any]] = {}
    for key, pass_fn in (
        ("monolithic_sdpa", pass_monolithic),
        ("fresh_canonical", pass_fresh),
        ("all_hit_reuse", pass_hit),
    ):
        rows = _run_passes(
            pass_fn,
            module_timer=module_timer,
            warmups=warmups,
            repeats=repeats,
            device=device,
        )
        summary = _per_record_arm(rows)
        _derived_glue(summary)
        _, prof = _profiler_pass(pass_fn, device)
        events = prof.events()
        cuda = summarize_kernel_events(events)
        cuda["op_counts"] = aggregate_op_counts(events)
        summary["cuda"] = cuda
        record_arms[key] = summary
    return record_arms


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    canonical_sha = _sha_file(ROOT / "ddi" / "canonical_cache.py")
    if canonical_sha != V64_CANONICAL_CACHE_SHA256:
        raise RuntimeError(
            "ddi/canonical_cache.py drifted from the sha the V60 confirmation ran with; "
            "phase-instrumented overrides must be re-verified before profiling"
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    source_manifest = json.loads(args.source_manifest.read_text())
    source_config = source_manifest["config"]
    cells = _load_frozen_requests(args.requests, args.record_limit)
    record_ids = [cell["record_id"] for cell in cells]

    model, tokenizer = load(str(args.checkpoint), dtype="bf16", backend="sdpa")
    model = model.to(device).eval()
    contract = replace(_contract(model, args.checkpoint), backend="sdpa")
    num_layers = int(model.config.num_hidden_layers)

    module_timer = GroupedModuleTimer(_module_groups(model))
    restore_wrappers = _install_phase_wrappers()
    try:
        from ddi import inference
        from ddi.data import load as load_dataset
        from ddi.prompt import build_query_text

        records = {record["id"]: record for record in load_dataset(
            source_config["dataset"], "val", source_config["seed"]
        )}
        encoded = []
        for cell in cells:
            record = records[cell["record_id"]]
            batch = inference.encode(
                tokenizer, list(record["info_contexts"]), build_query_text(tokenizer, record)
            )
            if batch.doc_lengths != cell["doc_lengths"] or batch.query_len != cell["query_length"]:
                raise ValueError(
                    f"re-tokenized shape drift for {cell['record_id']}: "
                    f"{batch.doc_lengths}/{batch.query_len} vs frozen "
                    f"{cell['doc_lengths']}/{cell['query_length']}"
                )
            encoded.append((cell, batch))

        cache_root = output / "canonical_cache" / "all_hit"
        per_record: dict[str, list[dict[str, Any]]] = {key: [] for key in ARM_KEYS}
        for index, (cell, batch) in enumerate(encoded):
            record_arms = _profile_record(
                cell,
                batch,
                model=model,
                contract=contract,
                cache_root=cache_root,
                module_timer=module_timer,
                warmups=args.warmups,
                repeats=args.repeats,
                device=device,
            )
            for key, summary in record_arms.items():
                per_record[key].append(summary)
            tokens = {key: record_arms[key]["first_token"] for key in ARM_KEYS}
            print(
                json.dumps({
                    "request_index": index,
                    "record_id": cell["record_id"],
                    "doc_lengths": cell["doc_lengths"],
                    "query_length": cell["query_length"],
                    "wall_ms": {key: round(record_arms[key]["wall_ms"], 2) for key in ARM_KEYS},
                    "first_token_agreement": (
                        tokens["fresh_canonical"] == tokens["monolithic_sdpa"]
                        and tokens["all_hit_reuse"] == tokens["monolithic_sdpa"]
                    ),
                })
            )
    finally:
        restore_wrappers()
        module_timer.remove()

    arms = {key: _aggregate_arm(per_record[key], num_layers) for key in ARM_KEYS}
    arms["monolithic_sdpa"]["extras"] = {
        "mask_wall_ms": statistics.median(
            [record["extras"].get("mask_wall_ms", 0.0) for record in per_record["monolithic_sdpa"]]
        ),
        "monolithic_model_glue_ms": statistics.median(
            [record.get("derived", {}).get("monolithic_model_glue_ms", 0.0)
             for record in per_record["monolithic_sdpa"]]
        ),
    }
    contributors_source = dict(arms["all_hit_reuse"]["phase_ms"])
    contributors_source.update({
        f"{key}_modules_cpu": value
        for key, value in arms["all_hit_reuse"]["modules_ms"].items()
    })
    # query_forward_total is a container: its children (assemble/mask/modules/glue) already
    # appear as contributors, so drop it to avoid double counting.
    contributors_source.pop("query_forward_total", None)
    gap = decompose_gap(
        {
            "wall_ms": arms["monolithic_sdpa"]["wall_ms"]["median"],
            "cpu_wall_ms": arms["monolithic_sdpa"]["cpu_wall_ms"]["median"],
            "cuda": {"kernel_ms": arms["monolithic_sdpa"]["cuda"]["kernel_ms_per_pass"]},
        },
        {
            "wall_ms": arms["all_hit_reuse"]["wall_ms"]["median"],
            "cpu_wall_ms": arms["all_hit_reuse"]["cpu_wall_ms"]["median"],
            "cuda": {"kernel_ms": arms["all_hit_reuse"]["cuda"]["kernel_ms_per_pass"]},
            "phase_ms": contributors_source,
        },
    )
    gap["fresh_canonical_minus_monolithic_wall_ms"] = (
        arms["fresh_canonical"]["wall_ms"]["median"]
        - arms["monolithic_sdpa"]["wall_ms"]["median"]
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "data_role": "development",
        "runner": "scripts/profile_v65_overhead.py",
        "created_utc": datetime.now(UTC).isoformat(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_identity": dict(
            zip(("mode", "sha256"), _checkpoint_identity(args.checkpoint), strict=True)
        ),
        "requests": {
            "path": str(args.requests.resolve()),
            "sha256": _sha_file(args.requests),
            "record_ids": record_ids,
            "source_manifest": {
                "path": str(args.source_manifest.resolve()),
                "sha256": _sha_file(args.source_manifest),
                "dataset": source_config["dataset"],
                "seed": source_config["seed"],
            },
        },
        "protocol": {
            "seed": args.seed,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "synchronize": True,
            "canonical_backend": "sdpa",
            "device": args.device,
            "gpu_name": torch.cuda.get_device_name() if device.type == "cuda" else None,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "num_layers": num_layers,
        },
        "targets": V64_TARGETS,
        "sources": {
            "ddi/canonical_cache.py": canonical_sha,
            "ddi/cache.py": _sha_file(ROOT / "ddi" / "cache.py"),
            "scripts/profile_v65_overhead.py": _sha_file(Path(__file__).resolve()),
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
    }
    _write_json(output / "manifest.json", manifest)

    result = {
        "schema_version": SCHEMA_VERSION,
        "data_role": "development",
        "manifest_sha256": _sha_file(output / "manifest.json"),
        "arms": arms,
        "gap_decomposition": gap,
        "records": {
            cell["record_id"]: {
                "doc_lengths": cell["doc_lengths"],
                "query_length": cell["query_length"],
                "arms": {
                    key: {
                        "wall_ms": summary["wall_ms"],
                        "cpu_wall_ms": summary["cpu_wall_ms"],
                        "phase_ms": summary["phase_ms"],
                        "modules_ms": summary["modules_ms"],
                        "extras": summary["extras"],
                        "first_token": summary["first_token"],
                        "cuda_kernel_ms": summary["cuda"]["kernel_ms"],
                        "cuda_launches": summary["cuda"]["launches"],
                    }
                    for key, summary in zip(
                        ARM_KEYS,
                        (per_record[key][index] for key in ARM_KEYS),
                        strict=True,
                    )
                },
            }
            for index, (cell, _) in enumerate(encoded)
        },
    }
    _write_json(output / "result.json", result)
    (output / "summary.md").write_text(_summary_markdown({**manifest, "arms": arms,
                                                           "gap_decomposition": gap}),
                                        encoding="utf-8")
    print(json.dumps({"output": str(output), "records": len(encoded)}))
    return result


if __name__ == "__main__":
    main()
