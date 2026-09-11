"""V61 step — timing-scope decomposition: where does the segmented-FA2 overhead live?

The prior artifact ``results/v61_timing_20260910T071010Z`` measured prefill WALL time only
and found segmented-FA2 winning only at ``N >= 5, L=512``. This run decomposes that wall
time into four scopes on the SAME checkpoint and token streams, on the reduced grid
``N in {1, 2, 5, 10} x L in {512}`` (the L=128 column is dropped PREREGISTERED: every
L=128 cell there was overhead-dominated, ratios 0.65-1.01):

- ``kernel``:  sum of attention-kernel device time via ``torch.profiler``
  (activities=[CUDA], ``with_stack=False``) over the prefill forward;
- ``module``:  CPU wall time inside the per-layer attention-module calls (hook-timed);
- ``prefill``: wall time of the full prefill forward (identical protocol to the prior
  artifact);
- ``request``: prefill (with this arm's own KV cache) + greedy decode of 16 new tokens.

Arms: SDPA-DDI reference vs segmented-FA2-DDI candidate, identical tokens/positions,
checkpoint ``Qwen__Qwen2.5-7B-Instruct__ddi_qf__8L_restart``, seed 47, 2 warmups,
10 timed repeats, ``torch.cuda.synchronize()`` between repeats. Per scope the ratio
``SDPA / segmented-FA2`` is reported as the median of paired per-repeat ratios plus a
paired bootstrap 95% CI (10k draws, seed 47, percentile method). Both arms are COLD
single-prefill (no KV reuse); warm-reuse is out of scope.

``manifest.json`` — including all scope/ratio definitions — is written BEFORE any cell is
measured. This run measures SPEED decomposition only; numerics were ratified separately
(FP64 calibration, ``scripts/calibrate_fa2_fp64.py``).

Run: ``<python> scripts/run_v61_timing_scopes.py`` (CUDA + flash-attn required).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ddi.inference import load  # noqa: E402
from ddi.masks import build_mask, to_additive  # noqa: E402
from ddi.segmented_fa2 import (  # noqa: E402
    clear_segmented_state,
    patch_model_for_segmented_fa2,
    set_segmented_state,
)
from scripts.run_segmented_fa2_timing import (  # noqa: E402
    BOOTSTRAP_DRAWS,
    QUERY_LENGTH,
    REPEATS,
    SEED,
    WARMUPS,
    _git_output,
    _sha_bytes,
    _sha_file,
    _sha_list,
    _write_json,
    build_runtime,
    cell_inputs,
    collect_checkpoint_identity,
    paired_bootstrap_ratio,
)

SCHEMA_VERSION = "v61_timing_scopes_v1"
DOC_COUNTS = (1, 2, 5, 10)
DOC_LENGTHS = (512,)  # preregistered reduction: L=128 dropped (overhead-dominated)
DECODE_NEW_TOKENS = 16
DECODE_STEPS = 16  # decode forwards per request (ddi.inference.generate loop shape)
SCOPES = ("kernel", "module", "prefill", "request")
KERNEL_PATTERNS = ("flash", "fmha", "attention")
PRIOR_ARTIFACT = "results/v61_timing_20260910T071010Z"
DEFAULT_CHECKPOINT = (
    ROOT / "results" / "checkpoints" / "Qwen__Qwen2.5-7B-Instruct__ddi_qf__8L_restart"
)
TOUCHED_DDI_FILES = (
    "ddi/inference.py",
    "ddi/masks.py",
    "ddi/model.py",
    "ddi/positions.py",
    "ddi/prompt.py",
    "ddi/segmented_fa2.py",
)

# ---- preregistration (stated here BEFORE the run; manifest.json is written first) -------
SCOPE_REDUCTION_NOTE = (
    "Preregistered scope reduction vs results/v61_timing_20260910T071010Z: the L=128 column "
    "is dropped. There every L=128 cell was overhead-dominated (ratios 0.65-1.01, gate fails) "
    "and the segmented-FA2 win appears only at N>=5, L=512; this decomposition therefore "
    "measures N={1, 2, 5, 10} x L={512} only."
)
SCOPE_DEFINITIONS = {
    "kernel": (
        "Sum of self-device-time of CUDA kernel events classified as attention (kernel name "
        "contains any of ('flash', 'fmha', 'attention')) emitted during one prefill forward "
        "(use_cache=False). Measured in a separate profiled pass: torch.profiler.profile("
        "activities=[ProfilerActivity.CUDA], with_stack=False), one profiler session per "
        "repeat, preceded by 2 unprofiled warmups plus 1 throwaway profiled warmup that "
        "absorbs CUPTI lazy initialization. Matched kernel names (+counts, total us) and the "
        "all-CUDA-kernel total are recorded per arm as diagnostics; profiler overhead never "
        "enters the wall-time scopes."
    ),
    "module": (
        "Sum over all per-layer self_attn modules of CPU wall time (time.perf_counter) "
        "between a forward-pre-hook and the forward hook of the same call, NO per-call "
        "synchronization, accumulated during the same use_cache=False prefill forwards that "
        "produce the prefill scope. This is the CPU-side wrapper/launch cost inside the "
        "attention modules; both arms carry identical hook instrumentation."
    ),
    "prefill": (
        "Wall time of the full prefill forward with use_cache=False, "
        "torch.cuda.synchronize()-bracketed — byte-identical protocol to "
        "results/v61_timing_20260910T071010Z (2 warmups, 10 timed repeats, sync between "
        "repeats)."
    ),
    "request": (
        "Wall time of one full request: prefill with use_cache=True (this arm's own KV cache) "
        "+ greedy decode producing 16 new tokens with use_cache=True, same decode-loop shape "
        "as ddi.inference.generate (16 decode forwards; the 16th forward's argmax is dropped; "
        "kept tokens = t1..t16; no EOS early stop so the step count is identical across "
        "arms). Per-step additive masks are the answer rows of ddi.masks.build_mask("
        "doc_lengths, query_len=64, answer_len=16, variant='ddi_qf'); decode positions "
        "continue the restart layout from the last prefill position. Candidate arm: "
        "segmented state is armed immediately before and cleared immediately after the "
        "prefill forward (a dict assignment), so decode falls back to SDPA."
    ),
}
RATIO_DEFINITIONS = (
    "Per scope, per repeat index i: ratio_i = seconds_reference[i] / seconds_candidate[i] "
    "(paired — both arms saw identical tokens/positions). Reported per cell and scope: the "
    "median and the mean of the ratio_i sample, plus a paired bootstrap 95% CI — resample "
    "repeat indices with replacement 10000 times (seed 47), statistic = mean of the "
    "resampled ratios, interval = 2.5/97.5 percentiles (identical bootstrap to the prior "
    "artifact so numbers stay comparable). ratio > 1 means the segmented-FA2 arm is faster "
    "in that scope."
)
TEMPERATURE_LABELS = (
    "Cold/warm labels: BOTH arms are cold single-prefill — every repeat re-prefills the full "
    "request from scratch and no KV is reused across repeats or cells; the decode inside the "
    "request scope reads only the KV cache produced by that same repeat's own prefill. "
    "Warm-reuse (per-document KV reuse across requests) is OUT OF SCOPE for this run."
)
MEASUREMENT_ORDER = (
    "Per cell: reference arm first (prefill pass, then kernel pass, then request pass), then "
    "the candidate arm in the same order; arms see identical tokens/positions. Peak CUDA "
    "memory is captured per arm: torch.cuda.reset_peak_memory_stats() immediately before the "
    "arm's first prefill warmup, max_memory_allocated/reserved read after the arm's request "
    "pass. Both 7B models are co-resident, so peaks include the shared residency and differ "
    "only in transient allocations."
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
    "scopes",
    "decode_tokens_match",
)


def attention_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    """The per-layer attention modules hooks attach to (one entry per decoder layer)."""
    return [layer.self_attn for layer in model.model.layers]


class ModuleCallTimer:
    """Accumulates CPU wall time inside registered module calls via forward hooks.

    ``seconds`` is the sum over calls of (post-hook perf_counter - pre-hook perf_counter);
    no synchronization is performed, so this measures the CPU-side duration of the module
    ``__call__`` (wrapper + kernel launches), which is the preregistered ``module`` scope.
    """

    def __init__(self, modules: list[torch.nn.Module]) -> None:
        self.module_count = len(modules)
        self.seconds = 0.0
        self.calls = 0
        self._enter = 0.0
        self._handles = []
        for module in modules:
            self._handles.append(module.register_forward_pre_hook(self._on_enter))
            self._handles.append(module.register_forward_hook(self._on_exit))

    def _on_enter(self, module: torch.nn.Module, args: tuple) -> None:  # noqa: ARG002
        self._enter = time.perf_counter()

    def _on_exit(self, module: torch.nn.Module, args: tuple, output: Any) -> Any:  # noqa: ARG002
        self.seconds += time.perf_counter() - self._enter
        self.calls += 1
        return output

    def reset(self) -> None:
        self.seconds = 0.0
        self.calls = 0

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _is_cuda_event(event: Any) -> bool:
    return "CUDA" in str(getattr(event, "device_type", "")).upper()


def classify_attention_kernels(
    events: Any, patterns: tuple[str, ...] = KERNEL_PATTERNS
) -> tuple[float, float, dict[str, dict[str, float]]]:
    """Split profiler events into (attention-kernel seconds, all-CUDA-kernel seconds, names).

    Only events whose ``device_type`` is CUDA are considered (CPU-side ops that launched
    kernels carry device_type CPU and are excluded, so nothing is double-counted). An
    attention kernel is a CUDA event whose name contains any of ``patterns``. The returned
    name map ({name: {count, total_us}}) is recorded as an audit diagnostic.
    """
    attention_seconds = 0.0
    all_seconds = 0.0
    matched: dict[str, dict[str, float]] = {}
    for event in events:
        if not _is_cuda_event(event):
            continue
        micros = float(getattr(event, "self_device_time_total", 0.0))
        all_seconds += micros / 1e6
        name = str(getattr(event, "name", "") or "")
        if any(pattern in name for pattern in patterns):
            attention_seconds += micros / 1e6
            entry = matched.setdefault(name, {"count": 0, "total_us": 0.0})
            entry["count"] += 1
            entry["total_us"] += micros
    return attention_seconds, all_seconds, matched


def paired_ratio_stats(
    reference_seconds: list[float],
    candidate_seconds: list[float],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = SEED,
) -> dict[str, Any]:
    """Paired per-repeat-index ratios plus the prior artifact's bootstrap, + median."""
    stats = paired_bootstrap_ratio(reference_seconds, candidate_seconds, draws=draws, seed=seed)
    ratios = torch.as_tensor(reference_seconds, dtype=torch.float64) / torch.as_tensor(
        candidate_seconds, dtype=torch.float64
    )
    # statistics.median (interpolated), NOT torch.median (lower-middle for even samples).
    stats["median"] = statistics.median(ratios.tolist())
    return stats


def build_decode_positions(last_position: int, count: int) -> torch.Tensor:
    """[1, count] decode positions continuing the restart layout past ``last_position``."""
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    if last_position < 0:
        raise ValueError(f"last_position must be >= 0, got {last_position}")
    return torch.arange(
        last_position + 1, last_position + 1 + count, dtype=torch.long
    ).unsqueeze(0)


def decode_step_mask(
    full_additive: torch.Tensor, prefix_len: int, step: int, kv_len: int
) -> torch.Tensor:
    """The [1, 1, 1, kv_len] additive mask for decode step ``step`` (answer row ``step``).

    Mirrors ``ddi.inference.generate``: row ``prefix_len + step`` of the full answer-extended
    additive matrix, sliced to the ``kv_len`` keys in the cache, cloned off the view so the
    base pointer stays aligned (see the alignment note in ``ddi/inference.py``).
    """
    if kv_len != prefix_len + step + 1:
        raise ValueError(
            f"kv_len must be prefix_len + step + 1 = {prefix_len + step + 1}, got {kv_len}"
        )
    row = prefix_len + step
    if row >= full_additive.shape[0]:
        raise ValueError(
            f"decode step {step} exceeds the answer-extended mask rows ({full_additive.shape[0]})"
        )
    return full_additive[row, :kv_len].clone().reshape(1, 1, 1, kv_len)


def _forward_prefill(
    model: torch.nn.Module, ids: torch.Tensor, additive: torch.Tensor, positions: torch.Tensor
) -> None:
    with torch.no_grad():
        model(
            input_ids=ids,
            attention_mask=additive,
            position_ids=positions,
            use_cache=False,
        )


def _timed_prefill(
    model: torch.nn.Module, ids: torch.Tensor, additive: torch.Tensor, positions: torch.Tensor
) -> float:
    """One prefill forward, synchronize-bracketed wall-clock seconds."""
    device = model.device
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    _forward_prefill(model, ids, additive, positions)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - started


def run_prefill_scopes(
    model: torch.nn.Module,
    timer: ModuleCallTimer,
    ids: torch.Tensor,
    additive: torch.Tensor,
    positions: torch.Tensor,
    *,
    warmups: int = WARMUPS,
    repeats: int = REPEATS,
) -> tuple[list[float], list[float], list[int]]:
    """Prefill (wall) + module (hook) scopes: the same forwards feed both scopes."""
    for _ in range(warmups):
        _timed_prefill(model, ids, additive, positions)
    prefill_seconds: list[float] = []
    module_seconds: list[float] = []
    module_calls: list[int] = []
    for _ in range(repeats):
        timer.reset()
        seconds = _timed_prefill(model, ids, additive, positions)
        if timer.calls != timer.module_count:
            raise RuntimeError(
                f"module hooks fired {timer.calls} times in one prefill, "
                f"expected {timer.module_count}"
            )
        prefill_seconds.append(seconds)
        module_seconds.append(timer.seconds)
        module_calls.append(timer.calls)
    return prefill_seconds, module_seconds, module_calls


def run_kernel_scope(
    model: torch.nn.Module,
    ids: torch.Tensor,
    additive: torch.Tensor,
    positions: torch.Tensor,
    *,
    warmups: int = WARMUPS,
    repeats: int = REPEATS,
) -> tuple[list[float], list[float], dict[str, dict[str, float]]]:
    """Attention-kernel scope: one CUDA profiler session per repeat, summed per repeat."""
    for _ in range(warmups):
        _timed_prefill(model, ids, additive, positions)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:  # CUPTI warm-up, discarded
        _forward_prefill(model, ids, additive, positions)
    attention_repeats: list[float] = []
    all_repeats: list[float] = []
    matched_names: dict[str, dict[str, float]] = {}
    for _ in range(repeats):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            _forward_prefill(model, ids, additive, positions)
        attention_seconds, all_seconds, matched = classify_attention_kernels(prof.events())
        if attention_seconds <= 0.0:
            raise RuntimeError(
                "no attention kernels classified in the profiled prefill — the preregistered "
                f"patterns {KERNEL_PATTERNS} matched nothing"
            )
        attention_repeats.append(attention_seconds)
        all_repeats.append(all_seconds)
        for name, agg in matched.items():
            entry = matched_names.setdefault(name, {"count": 0, "total_us": 0.0})
            entry["count"] += int(agg["count"])
            entry["total_us"] += agg["total_us"]
    return attention_repeats, all_repeats, matched_names


def _request_once(
    model: torch.nn.Module,
    ids: torch.Tensor,
    prefill_additive: torch.Tensor,
    positions: torch.Tensor,
    full_additive: torch.Tensor,
    decode_positions: torch.Tensor,
    *,
    prefix_len: int,
    decode_steps: int,
    arm: Any,
    disarm: Any,
) -> tuple[float, list[int]]:
    """One full request (prefill + greedy decode), synchronize-bracketed; returns (s, tokens)."""
    device = model.device
    if device.type == "cuda":
        torch.cuda.synchronize()
    try:
        if arm is not None:
            arm()
        started = time.perf_counter()
        with torch.no_grad():
            out = model(
                input_ids=ids,
                attention_mask=prefill_additive,
                position_ids=positions,
                use_cache=True,
            )
        if disarm is not None:
            disarm()
        past = out.past_key_values
        next_token = out.logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)
        kept: list[int] = []
        for step in range(decode_steps):
            kept.append(int(next_token.item()))
            step_mask = decode_step_mask(full_additive, prefix_len, step, prefix_len + step + 1)
            with torch.no_grad():
                step_out = model(
                    input_ids=next_token,
                    attention_mask=step_mask,
                    position_ids=decode_positions[:, step : step + 1].clone(),
                    past_key_values=past,
                    use_cache=True,
                )
            past = step_out.past_key_values
            next_token = step_out.logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - started, kept
    finally:
        if disarm is not None:
            disarm()


def run_request_scope(
    model: torch.nn.Module,
    ids: torch.Tensor,
    prefill_additive: torch.Tensor,
    positions: torch.Tensor,
    full_additive: torch.Tensor,
    decode_positions: torch.Tensor,
    *,
    prefix_len: int,
    decode_steps: int = DECODE_STEPS,
    warmups: int = WARMUPS,
    repeats: int = REPEATS,
    arm: Any = None,
    disarm: Any = None,
) -> tuple[list[float], list[int]]:
    """Request scope: prefill (this arm's own cache) + greedy decode, warmups then repeats."""
    for _ in range(warmups):
        _request_once(
            model,
            ids,
            prefill_additive,
            positions,
            full_additive,
            decode_positions,
            prefix_len=prefix_len,
            decode_steps=decode_steps,
            arm=arm,
            disarm=disarm,
        )
    request_seconds: list[float] = []
    kept_tokens: list[int] | None = None
    for _ in range(repeats):
        seconds, kept = _request_once(
            model,
            ids,
            prefill_additive,
            positions,
            full_additive,
            decode_positions,
            prefix_len=prefix_len,
            decode_steps=decode_steps,
            arm=arm,
            disarm=disarm,
        )
        if kept_tokens is None:
            kept_tokens = kept
        request_seconds.append(seconds)
    assert kept_tokens is not None
    return request_seconds, kept_tokens


def _median_entry(repeats: list[float]) -> dict[str, Any]:
    return {"repeats_seconds": repeats, "median_seconds": statistics.median(repeats)}


def run_cell(
    reference: torch.nn.Module,
    candidate: torch.nn.Module,
    spec: dict[str, Any],
    *,
    reference_timer: ModuleCallTimer,
    candidate_timer: ModuleCallTimer,
) -> dict[str, Any]:
    """Measure all four scopes on both arms for one grid cell; returns one validated cell."""
    doc_lengths = spec["doc_lengths"]
    inputs = cell_inputs(spec["n_docs"], spec["doc_length"])
    device = reference.device
    ids = torch.tensor([inputs["tokens"]], dtype=torch.long).to(device)
    positions = torch.tensor([inputs["positions"]], dtype=torch.long).to(device)
    prefix_len = len(inputs["tokens"])

    # Standalone prefill mask (answer_len=0): byte-identical to the prior artifact's protocol.
    prefill_additive = (
        to_additive(build_mask(doc_lengths, QUERY_LENGTH, 0, "ddi_qf"), dtype=reference.dtype)
        .to(device)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    # Answer-extended mask for the request scope (prefill block + 16 answer rows).
    full_additive = to_additive(
        build_mask(doc_lengths, QUERY_LENGTH, DECODE_NEW_TOKENS, "ddi_qf"),
        dtype=reference.dtype,
    ).to(device)
    # 4D like ddi.inference.generate's prefill mask: a 2D mask would fall into HF's
    # dense-mask-building path and materialize a [T, 1, T, T] tensor (hundreds of GiB).
    request_prefill_additive = (
        full_additive[:prefix_len, :prefix_len].clone().unsqueeze(0).unsqueeze(0)
    )
    decode_positions = build_decode_positions(int(positions[0, -1].item()), DECODE_STEPS).to(
        device
    )

    arm_specs = (
        (
            "reference",
            reference,
            reference_timer,
            None,
            None,
        ),
        (
            "candidate",
            candidate,
            candidate_timer,
            lambda: set_segmented_state(candidate, doc_lengths, QUERY_LENGTH, variant="ddi_qf"),
            lambda: clear_segmented_state(candidate),
        ),
    )

    arm_results: dict[str, dict[str, Any]] = {}
    decode_tokens: dict[str, list[int]] = {}
    for name, model, timer, arm, disarm in arm_specs:
        is_cuda = device.type == "cuda"
        if is_cuda:
            torch.cuda.reset_peak_memory_stats()

        set_fn = (lambda: None) if arm is None else arm
        clear_fn = (lambda: None) if disarm is None else disarm
        set_fn()
        try:
            prefill, module, module_calls = run_prefill_scopes(
                model, timer, ids, prefill_additive, positions
            )
            kernel, kernel_all, matched = run_kernel_scope(model, ids, prefill_additive, positions)
        finally:
            clear_fn()
        request, kept = run_request_scope(
            model,
            ids,
            request_prefill_additive,
            positions,
            full_additive,
            decode_positions,
            prefix_len=prefix_len,
            arm=arm,
            disarm=disarm,
        )

        scopes = {
            "kernel": {
                **_median_entry(kernel),
                "all_cuda_kernel_repeats_seconds": kernel_all,
                "matched_kernel_names": matched,
            },
            "module": {**_median_entry(module), "module_calls": module_calls},
            "prefill": _median_entry(prefill),
            "request": _median_entry(request),
        }
        peak = (
            {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            if is_cuda
            else {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
        )
        arm_results[name] = {
            "backend": "sdpa" if name == "reference" else "segmented_fa2",
            "scopes": scopes,
            "peak_memory": peak,
            "decode_tokens_sha256": _sha_list(kept),
        }
        decode_tokens[name] = kept

    cell = {
        "status": "ok",
        "n_docs": spec["n_docs"],
        "doc_length": spec["doc_length"],
        "doc_lengths": doc_lengths,
        "query_length": QUERY_LENGTH,
        "total_tokens": prefix_len,
        "tokens_sha256": _sha_list(inputs["tokens"]),
        "positions_sha256": _sha_list(inputs["positions"]),
        "reference": arm_results["reference"],
        "candidate": arm_results["candidate"],
        "scopes": {
            scope: paired_ratio_stats(
                arm_results["reference"]["scopes"][scope]["repeats_seconds"],
                arm_results["candidate"]["scopes"][scope]["repeats_seconds"],
            )
            for scope in SCOPES
        },
        "decode_tokens_match": (
            arm_results["reference"]["decode_tokens_sha256"]
            == arm_results["candidate"]["decode_tokens_sha256"]
        ),
    }
    validate_scope_cell(cell)
    return cell


def validate_scope_cell(cell: dict[str, Any]) -> None:
    """Structural + arithmetic invariants for one successful scope-decomposition cell."""
    missing = [key for key in _REQUIRED_CELL_KEYS if key not in cell]
    if missing:
        raise ValueError(f"cell result missing keys: {missing}")
    if cell["query_length"] != QUERY_LENGTH:
        raise ValueError(f"query_length must be {QUERY_LENGTH}")
    if cell["total_tokens"] != sum(cell["doc_lengths"]) + cell["query_length"]:
        raise ValueError("total_tokens disagrees with doc_lengths + query_length")
    for arm in ("reference", "candidate"):
        scopes = cell[arm]["scopes"]
        if set(scopes) != set(SCOPES):
            raise ValueError(f"{arm} arm scopes must be exactly {SCOPES}")
        for scope in SCOPES:
            entry = scopes[scope]
            repeats = entry["repeats_seconds"]
            if len(repeats) != REPEATS:
                raise ValueError(f"{arm}/{scope} must have {REPEATS} repeats")
            if any(not isinstance(seconds, (int, float)) or seconds <= 0 for seconds in repeats):
                raise ValueError(f"{arm}/{scope} repeats must be positive numbers")
            median = entry["median_seconds"]
            if median is None or abs(median - statistics.median(repeats)) > 1e-12:
                raise ValueError(f"{arm}/{scope} median_seconds disagrees with its repeats")
            if scope == "kernel":
                all_repeats = entry["all_cuda_kernel_repeats_seconds"]
                covered = all(a >= b - 1e-12 for a, b in zip(all_repeats, repeats, strict=True))
                if len(all_repeats) != REPEATS or not covered:
                    raise ValueError(
                        f"{arm}/kernel all-CUDA totals must cover attention kernels"
                    )
        peak = cell[arm]["peak_memory"]
        if set(peak) != {"peak_allocated_bytes", "peak_reserved_bytes"}:
            raise ValueError(f"{arm} peak_memory must carry allocated/reserved byte counts")
        if any(not isinstance(value, int) or value < 0 for value in peak.values()):
            raise ValueError(f"{arm} peak_memory values must be non-negative integers")
        if not isinstance(cell[arm]["decode_tokens_sha256"], str):
            raise ValueError(f"{arm} decode_tokens_sha256 must be a string")
    for scope in SCOPES:
        ratio = cell["scopes"][scope]
        ref_reps = cell["reference"]["scopes"][scope]["repeats_seconds"]
        cand_reps = cell["candidate"]["scopes"][scope]["repeats_seconds"]
        expected = [ref / cand for ref, cand in zip(ref_reps, cand_reps, strict=True)]
        if len(ratio["repeats"]) != REPEATS or any(
            abs(got - want) > 1e-9
            for got, want in zip(ratio["repeats"], expected, strict=True)
        ):
            raise ValueError(f"{scope} ratio repeats disagree with the paired arm repeats")
        if ratio["ci95_low"] > ratio["mean"] + 1e-12 or ratio["mean"] > ratio["ci95_high"] + 1e-12:
            raise ValueError(f"{scope} bootstrap ci95 does not bracket the ratio mean")
        if abs(ratio["median"] - statistics.median(expected)) > 1e-9:
            raise ValueError(f"{scope} ratio median disagrees with the paired ratio repeats")
        if ratio["draws"] != BOOTSTRAP_DRAWS or ratio["seed"] != SEED:
            raise ValueError(f"{scope} bootstrap must use {BOOTSTRAP_DRAWS} draws and seed {SEED}")
    if not isinstance(cell["decode_tokens_match"], bool):
        raise ValueError("decode_tokens_match must be a boolean")


def build_cell_specs() -> list[dict[str, Any]]:
    """The reduced grid: N x L=512 cells with per-document lengths and the shared query."""
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


def build_manifest(checkpoint: Path) -> dict[str, Any]:
    """Schema ``v61_timing_scopes_v1`` manifest — written BEFORE any cell is measured."""
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
    touched = {name: _sha_file(ROOT / name) for name in TOUCHED_DDI_FILES}
    touched["scripts/run_v61_timing_scopes.py"] = _sha_file(Path(__file__).resolve())
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": _sha_bytes(
            json.dumps(
                {
                    "schema": SCHEMA_VERSION,
                    "checkpoint": str(checkpoint.resolve()),
                    "seed": SEED,
                    "created": created.isoformat(),
                },
                sort_keys=True,
            ).encode("utf-8")
        ),
        "created_utc": created.isoformat(),
        "prior_artifact": PRIOR_ARTIFACT,
        "preregistration": {
            "note": (
                "This manifest (scope definitions, ratio definitions, grid, temperature labels) "
                "is written to the output directory BEFORE any cell is measured."
            ),
            "scope_reduction": SCOPE_REDUCTION_NOTE,
            "grid": {
                "doc_counts": list(DOC_COUNTS),
                "doc_lengths": list(DOC_LENGTHS),
                "query_length": QUERY_LENGTH,
                "seed": SEED,
                "warmups": WARMUPS,
                "timed_repeats": REPEATS,
                "sync_between_repeats": True,
                "decode_new_tokens": DECODE_NEW_TOKENS,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "bootstrap_seed": SEED,
            },
            "scope_definitions": SCOPE_DEFINITIONS,
            "ratio_definitions": RATIO_DEFINITIONS,
            "temperature_labels": TEMPERATURE_LABELS,
            "measurement_order": MEASUREMENT_ORDER,
        },
        "criterion_note": (
            "Numerics ratified separately via FP64 calibration (RMS<=0.02, "
            "scripts/calibrate_fa2_fp64.py); this run is a SPEED DECOMPOSITION with no "
            "pass/fail gate — every scope ratio is reported with its paired bootstrap CI."
        ),
        "checkpoint": collect_checkpoint_identity(checkpoint),
        "git": {
            "source_revision": _git_output("rev-parse", "HEAD").decode().strip(),
            "diff_sha256": _sha_bytes(diff),
            "diff_empty": not diff,
        },
        "runtime": build_runtime(),
        "touched_sources": touched,
        "cells": cells,
    }


def _summary_scope(cell: dict[str, Any], scope: str) -> dict[str, Any]:
    ratio = cell["scopes"][scope]
    return {
        "reference_median_seconds": cell["reference"]["scopes"][scope]["median_seconds"],
        "candidate_median_seconds": cell["candidate"]["scopes"][scope]["median_seconds"],
        "ratio_median": ratio["median"],
        "ratio_mean": ratio["mean"],
        "ratio_ci95": [ratio["ci95_low"], ratio["ci95_high"]],
    }


def _summary_entry(cell: dict[str, Any]) -> dict[str, Any]:
    entry = {
        "n_docs": cell["n_docs"],
        "doc_length": cell["doc_length"],
        "total_tokens": cell["total_tokens"],
        "status": cell["status"],
    }
    if cell["status"] == "ok":
        entry.update(
            {
                "scopes": {scope: _summary_scope(cell, scope) for scope in SCOPES},
                "peak_memory": {
                    "reference": cell["reference"]["peak_memory"],
                    "candidate": cell["candidate"]["peak_memory"],
                },
                "decode_tokens_match": cell["decode_tokens_match"],
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
        args.output = ROOT / "results" / f"v61_timing_scopes_{stamp}"
    if args.output.exists():
        raise FileExistsError(f"--output must not exist: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("the V61 timing-scope campaign requires CUDA")
    args.output.mkdir(parents=True)
    run_started = datetime.now(UTC)

    print(f"loading SDPA reference from {args.checkpoint}", flush=True)
    reference, _ = load(str(args.checkpoint), dtype="bf16", backend="sdpa")
    reference = reference.to("cuda").eval()
    print("loading eager candidate", flush=True)
    candidate, _ = load(str(args.checkpoint), dtype="bf16", backend="eager")
    candidate = patch_model_for_segmented_fa2(candidate.to("cuda").eval())
    timers = (
        ModuleCallTimer(attention_modules(reference)),
        ModuleCallTimer(attention_modules(candidate)),
    )
    torch.cuda.reset_peak_memory_stats()

    manifest = build_manifest(args.checkpoint)
    _write_json(args.output / "manifest.json", manifest)
    print(f"manifest_id={manifest['manifest_id']}", flush=True)

    cells = []
    for spec in manifest["cells"]:
        label = f"N={spec['n_docs']} L={spec['doc_length']}"
        try:
            cell = run_cell(
                reference,
                candidate,
                spec,
                reference_timer=timers[0],
                candidate_timer=timers[1],
            )
            for scope in SCOPES:
                ratio = cell["scopes"][scope]
                print(
                    f"{label} {scope}: "
                    f"ref={cell['reference']['scopes'][scope]['median_seconds']:.6f}s "
                    f"cand={cell['candidate']['scopes'][scope]['median_seconds']:.6f}s "
                    f"ratio_median={ratio['median']:.3f} "
                    f"ci95=[{ratio['ci95_low']:.3f}, {ratio['ci95_high']:.3f}]",
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
            }
            print(f"{label}: ERROR {cell['error']}", flush=True)
        cells.append(cell)

    resources = {
        "device": torch.cuda.get_device_name(),
        "wall_seconds": (datetime.now(UTC) - run_started).total_seconds(),
        "process_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "process_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": manifest["manifest_id"],
        "created_utc": manifest["created_utc"],
        "prior_artifact": PRIOR_ARTIFACT,
        "runtime": manifest["runtime"],
        "resources": resources,
        "cells": cells,
    }
    _write_json(args.output / "result.json", result)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": manifest["manifest_id"],
        "prior_artifact": PRIOR_ARTIFACT,
        "scope_definitions": SCOPE_DEFINITIONS,
        "ratio_definitions": RATIO_DEFINITIONS,
        "temperature_labels": TEMPERATURE_LABELS,
        "resources": resources,
        "cells": [_summary_entry(cell) for cell in cells],
    }
    _write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
