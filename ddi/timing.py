"""Timing harness — FTT / TPOT with warmup, repeats, median + IQR (issue #17, Phase 3.1).

Ground rule 4: every efficiency number ≥5 repeats, discarded warmup, reported variance. Every
Phase 3 sweep (#18 doc-count, #19 doc-length, #20 backends, #21 realization ratio) builds on
this primitive. One `measure` + two thin wrappers (`measure_prefill`, `measure_decode`) is the
whole surface; no "measure everything at once" convenience wrapper by design.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import median

import torch

from ddi.model import forward_with_mask
from ddi.provenance import Provenance
from ddi.store import DEFAULT_STORE, append


def _quantile(sorted_xs: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile over a pre-sorted sequence.

    Inlined from the deleted ``ddi.report._quantile`` in #89 so the IQR
    computation in the Timing dataclass keeps its bit-identical semantics.
    """
    n = len(sorted_xs)
    if n == 1:
        return float(sorted_xs[0])
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * frac)


@dataclass(frozen=True)
class Timing:
    name: str
    n_repeats: int
    warmup: int
    median_ms: float
    iqr_low_ms: float
    iqr_high_ms: float
    per_repeat_ms: list[float]


def _time_once_cpu(fn: Callable[[], object]) -> float:
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000.0


def _time_once_cuda(fn: Callable[[], object]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def measure(
    fn: Callable[[], object],
    *,
    n_repeats: int = 5,
    warmup: int = 3,
    name: str = "",
    use_cuda_events: bool | None = None,
) -> Timing:
    """Run `fn` `warmup + n_repeats` times, drop the first `warmup` calls, time each remaining call.

    If `use_cuda_events` is None the timer is auto-selected: `torch.cuda.Event` when CUDA is
    available, else `time.perf_counter`. Between repeats we call `torch.cuda.empty_cache()` so
    each timed call reflects a cold-cache run of the same shape, and `torch.cuda.synchronize()`
    after the fn so the reading captures the full compute rather than just the kernel launch.

    This function does NOT wrap `fn` in `torch.no_grad()` — the caller decides. Measuring a
    gradient-tracked step is a legitimate Phase 3.5 use case, and forcing no-grad here would
    silently change what the caller asked to be timed.
    """
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    # WHY branch here: CUDA-event timing measures device-side elapsed time and needs a synchronize
    # to be meaningful; perf_counter is the only sensible choice with no CUDA context. Auto-select
    # so the caller does not have to know which environment they are on; explicit bool overrides
    # for tests (force CPU path) and cases where a CUDA build has no visible device.
    on_cuda = torch.cuda.is_available()
    if use_cuda_events is None:
        use_cuda_events = on_cuda
    if use_cuda_events and not on_cuda:
        raise RuntimeError("use_cuda_events=True but torch.cuda.is_available() is False")

    time_once = _time_once_cuda if use_cuda_events else _time_once_cpu

    samples: list[float] = []
    for i in range(warmup + n_repeats):
        if on_cuda:
            torch.cuda.empty_cache()
        ms = time_once(fn)
        if on_cuda:
            torch.cuda.synchronize()
        if i >= warmup:
            samples.append(ms)

    xs = sorted(samples)
    med = float(median(xs))
    q1 = _quantile(xs, 0.25)
    q3 = _quantile(xs, 0.75)

    return Timing(
        name=name,
        n_repeats=n_repeats,
        warmup=warmup,
        median_ms=med,
        iqr_low_ms=q1,
        iqr_high_ms=q3,
        per_repeat_ms=samples,
    )


def measure_prefill(
    model,
    input_ids: torch.Tensor,
    additive_mask: torch.Tensor,
    *,
    n_repeats: int = 5,
    warmup: int = 3,
) -> Timing:
    """FTT (first-token time) for a single masked prefill via `forward_with_mask`."""

    def _fn() -> object:
        with torch.no_grad():
            return forward_with_mask(model, input_ids, additive_mask)

    return measure(_fn, n_repeats=n_repeats, warmup=warmup, name="prefill")


def measure_decode(
    model,
    prompt_ids: torch.Tensor,
    additive_mask: torch.Tensor,
    n_new_tokens: int = 16,
    *,
    n_repeats: int = 5,
    warmup: int = 3,
) -> Timing:
    """TPOT (time per output token). Each repeat: DDI-masked prefill (untimed), then autoregress
    `n_new_tokens` with cached KVs (timed). Reported ms is per-token (total / n_new_tokens)."""
    if n_new_tokens < 1:
        raise ValueError(f"n_new_tokens must be >= 1, got {n_new_tokens}")

    def _fn() -> object:
        with torch.no_grad():
            out = model(
                input_ids=prompt_ids,
                attention_mask=additive_mask,
                use_cache=True,
            )
            past = out.past_key_values
            next_token = out.logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)
            for _ in range(n_new_tokens):
                step = model(
                    input_ids=next_token,
                    past_key_values=past,
                    use_cache=True,
                )
                past = step.past_key_values
                next_token = step.logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)
            return next_token

    # Time prefill + n_new_tokens decode steps together, then divide — this is a two-step process
    # inside one fn because a cold KV cache is a precondition for the decode loop.
    total = measure(_fn, n_repeats=n_repeats, warmup=warmup, name="decode_total")
    per_token = [x / n_new_tokens for x in total.per_repeat_ms]
    xs = sorted(per_token)
    return Timing(
        name="decode",
        n_repeats=n_repeats,
        warmup=warmup,
        median_ms=float(median(xs)),
        iqr_low_ms=_quantile(xs, 0.25),
        iqr_high_ms=_quantile(xs, 0.75),
        per_repeat_ms=per_token,
    )


def append_timing(
    store,
    run_id: str,
    metric: str,
    timing: Timing,
    provenance: Provenance,
) -> None:
    """Append median, IQR bounds, and per-repeat samples to the store as four rows."""
    store_path = store if store is not None else DEFAULT_STORE
    append(run_id, f"{metric}_median_ms", timing.median_ms, provenance, store=store_path)
    append(run_id, f"{metric}_iqr_low_ms", timing.iqr_low_ms, provenance, store=store_path)
    append(run_id, f"{metric}_iqr_high_ms", timing.iqr_high_ms, provenance, store=store_path)
    append(
        run_id,
        f"{metric}_repeats_ms",
        json.dumps(timing.per_repeat_ms),
        provenance,
        store=store_path,
    )
