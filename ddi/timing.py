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


def _timing_from_samples(
    samples: Sequence[float],
    *,
    name: str,
    n_repeats: int,
    warmup: int,
) -> Timing:
    """Median + IQR over `samples`, preserving their original (unsorted) order in the record."""
    xs = sorted(samples)
    return Timing(
        name=name,
        n_repeats=n_repeats,
        warmup=warmup,
        median_ms=float(median(xs)),
        iqr_low_ms=_quantile(xs, 0.25),
        iqr_high_ms=_quantile(xs, 0.75),
        per_repeat_ms=list(samples),
    )


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
    setup: Callable[[], object] | None = None,
    n_repeats: int = 5,
    warmup: int = 3,
    name: str = "",
    use_cuda_events: bool | None = None,
) -> Timing:
    """Run `fn` `warmup + n_repeats` times, drop the first `warmup` calls, time each remaining call.

    `setup`, when given, runs once per iteration — warmup iterations included — *before* the timer
    starts, and its cost is excluded from the reading. It exists for the case where the thing being
    measured has a precondition that is itself expensive and separately reported: `measure_decode`
    needs a warm KV cache, so its prefill has to run every iteration but must not land in `tpot`.
    Do not use it to hoist work out of a measurement that legitimately owns that work.

    If `use_cuda_events` is None the timer is auto-selected: `torch.cuda.Event` when CUDA is
    available, else `time.perf_counter`. Between repeats we call `torch.cuda.empty_cache()` so
    each timed call reflects a cold-cache run of the same shape, and `torch.cuda.synchronize()`
    after `setup` and after `fn` so the reading brackets exactly the timed region.

    The allocator flush follows the *resolved* `use_cuda_events` rather than bare
    `torch.cuda.is_available()`: it is the expensive one — a device-wide flush plus an implicit
    sync — and a CPU-resident `fn` on a GPU host was paying for it every iteration for nothing. The
    two synchronizes stay on mere availability, because they are microseconds on an idle stream and
    a missed barrier is a wrong number rather than a slow one. Note that neither has ever been able
    to bias a reading: both sit outside the timed bracket. They cost sweep wall-clock, not accuracy.

    `measure_prefill` / `measure_decode` resolve `use_cuda_events` from the device their tensors
    actually live on, not from availability — auto-selecting CUDA events for a CPU-resident model
    on a GPU host would report ~0 ms, since events only see device-side work.

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
        if use_cuda_events:
            torch.cuda.empty_cache()
        if setup is not None:
            setup()
            # Drain setup before the timer opens, or its kernels finish inside `fn`'s window —
            # which is the very contamination `setup` exists to remove. Keyed on availability, not
            # on the timer choice: a caller who forces the CPU timer over CUDA tensors still needs
            # the barrier, and it is free when the stream is idle.
            if on_cuda:
                torch.cuda.synchronize()
        ms = time_once(fn)
        if on_cuda:
            torch.cuda.synchronize()
        if i >= warmup:
            samples.append(ms)

    return _timing_from_samples(samples, name=name, n_repeats=n_repeats, warmup=warmup)


def _cuda_events_for(reference: torch.Tensor, override: bool | None) -> bool:
    """Whether to time this workload with CUDA events, judged by where its tensors live.

    `measure`'s own auto-select can only ask whether a CUDA device *exists*. That is the wrong
    question for a model pinned to CPU on a GPU host: CUDA events would bracket a region with no
    device work in it and report ~0 ms. The workload's own tensors answer it correctly.
    """
    if override is not None:
        return override
    return bool(reference.is_cuda)


def measure_prefill(
    model,
    input_ids: torch.Tensor,
    additive_mask: torch.Tensor,
    *,
    position_ids: torch.Tensor | None = None,
    n_repeats: int = 5,
    warmup: int = 3,
    use_cuda_events: bool | None = None,
) -> Timing:
    """FTT (first-token time) for a single masked prefill via `inference.forward`.

    `position_ids` is the optional `[1, T]` long tensor from `ddi.positions.build_positions`; it is
    forwarded verbatim so a timing sweep can hold the mask fixed and vary the layout axis
    (`sequential` vs `restart`). Omitted, HF's default sequential handling applies — the shape of
    the computation is identical either way, so this changes no existing number.
    """
    # Route through inference.forward with mask_variant=None because callers
    # already built the additive mask; we pass it straight through.
    from ddi.inference import forward

    def _fn() -> object:
        mask_for_call = additive_mask  # [1, 1, T, T] or [T, T]
        with torch.no_grad():
            return forward(
                model,
                input_ids,
                mask_variant=None,
                additive_mask=mask_for_call,
                position_ids=position_ids,
            )

    return measure(
        _fn,
        n_repeats=n_repeats,
        warmup=warmup,
        name="prefill",
        use_cuda_events=_cuda_events_for(input_ids, use_cuda_events),
    )


def _as_2d(mask: torch.Tensor) -> torch.Tensor:
    """Strip leading singleton (batch, head) dims off a mask so we can index it by row."""
    m = mask
    while m.dim() > 2:
        if m.shape[0] != 1:
            raise ValueError(
                f"expected leading singleton dims on the mask, got shape {tuple(mask.shape)}"
            )
        m = m[0]
    if m.dim() != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"expected a square [T, T] mask, got shape {tuple(mask.shape)}")
    return m


def _decode_matrix(
    additive_mask: torch.Tensor,
    prefix_len: int,
    n_new_tokens: int,
) -> torch.Tensor:
    """Return the additive matrix covering prefill *and* the whole decode budget, [T+n, T+n].

    The caller may hand us either matrix. Both call shapes are load-bearing, so neither is a
    deprecation: `ddi.run._timing_cell` already builds a prefill-only mask for `measure_prefill`
    and passes the same tensor here, while a caller who wants the answer rows to be exactly what
    `ddi.masks.build_mask` says can pass the `answer_len=n_new_tokens` matrix and skip the
    reconstruction below.

    Reconstruction is exact, not an approximation: for every variant `build_mask` emits, an
    answer row over the prefill columns is identical to the *last* prefill row (the answer sees
    the same documents the query block sees as a whole, plus the whole query — masks.py:99-106),
    and over the answer columns it is plain causal. Do not "simplify" this by leaving the answer
    rows unmasked — that is the bug this function exists to close.
    """
    m = _as_2d(additive_mask)
    total = prefix_len + n_new_tokens

    if m.shape[0] == total:
        return m
    if m.shape[0] != prefix_len:
        raise ValueError(
            f"mask width {m.shape[0]} matches neither the prompt ({prefix_len}) nor "
            f"prompt + n_new_tokens ({total})"
        )

    out = torch.full((total, total), float("-inf"), dtype=m.dtype, device=m.device)
    out[:prefix_len, :prefix_len] = m
    out[prefix_len:, :prefix_len] = m[prefix_len - 1]
    causal = torch.ones(n_new_tokens, n_new_tokens, dtype=torch.bool, device=m.device).tril()
    out[prefix_len:, prefix_len:].masked_fill_(causal, 0.0)
    return out


def _decode_positions(
    position_ids: torch.Tensor,
    prefix_len: int,
    n_new_tokens: int,
) -> torch.Tensor:
    """Return `[1, prefix_len + n_new_tokens]` position ids spanning prefill *and* the decode run.

    Mirrors `_decode_matrix`'s contract: the caller may hand over either the prompt-only vector or
    the full one, in `[T]` or `[1, T]` form. Extension is exact rather than an approximation for
    both layouts `ddi.positions.build_positions` emits — it writes the query+answer tail as a
    single contiguous `arange` (positions.py:66-68), so the answer positions are always the
    continuation of the last prompt position, whether the docs restarted at 0 or not.
    """
    p = position_ids
    if p.dim() == 1:
        p = p.unsqueeze(0)
    if p.dim() != 2 or p.shape[0] != 1:
        raise ValueError(
            f"position_ids must be [T] or [1, T]; got shape {tuple(position_ids.shape)}"
        )
    if p.dtype != torch.long:
        raise ValueError(f"position_ids must be torch.long; got {p.dtype}")

    total = prefix_len + n_new_tokens
    width = int(p.shape[-1])
    if width == total:
        return p
    if width != prefix_len:
        raise ValueError(
            f"position_ids width {width} matches neither the prompt ({prefix_len}) nor "
            f"prompt + n_new_tokens ({total})"
        )

    tail_start = int(p[0, -1].item()) + 1
    tail = torch.arange(
        tail_start, tail_start + n_new_tokens, dtype=torch.long, device=p.device
    ).unsqueeze(0)
    return torch.cat([p, tail], dim=-1)


def measure_decode(
    model,
    prompt_ids: torch.Tensor,
    additive_mask: torch.Tensor,
    n_new_tokens: int = 16,
    *,
    position_ids: torch.Tensor | None = None,
    n_repeats: int = 5,
    warmup: int = 3,
    use_cuda_events: bool | None = None,
) -> Timing:
    """TPOT (time per output token). Each repeat: DDI-masked prefill (untimed), then autoregress
    `n_new_tokens` with cached KVs (timed). Reported ms is per-token (decode total / n_new_tokens).

    `additive_mask` is either the prefill-only [T, T] mask (optionally with leading singleton
    dims) or the full [T + n_new_tokens, T + n_new_tokens] one; see `_decode_matrix`. Optional
    `position_ids` follows the same either-width contract; see `_decode_positions`.

    The prefill is a *precondition*, not part of the measurement, and it is bracketed out via
    `measure(setup=...)`. It used to sit inside the same timed region as the decode loop, with the
    combined total divided by `n_new_tokens` — so the number reported as `tpot` was really
    `(FTT + n·TPOT) / n`, overstated by `FTT / n`. At the shipped default of 16 new tokens against
    an `L=2048, N=25` prefill that error term is the dominant one.

    Deriving it out afterwards would not have been enough either, because the contamination
    correlates with the variable under test: prefill is exactly where mask sparsity pays off, so a
    sparse arm looked faster per *decode* token partly because its *prefill* was cheaper, and the
    error ran in the flattering direction. `realization` in `experiments/performance.yaml` is
    derived from `tpot` and inherited it. Timing the two regions separately is what makes the
    decode number a decode number.

    Every decode step is masked. Running the decode loop with no `attention_mask` — as this
    function used to — hands HF's default full-causal mask over the whole KV cache to every
    step, so `full`, `ddi_qf` and `blind` decode the *same* computation and their `tpot_*` rows
    differ only by noise. That silently makes the decode half of every speed claim, and the whole
    `masks:` axis of `experiments/performance.yaml`, measure nothing. This is the same defect
    already post-mortemed in `ddi.inference.generate`; it was fixed there and left standing here.
    """
    if n_new_tokens < 1:
        raise ValueError(f"n_new_tokens must be >= 1, got {n_new_tokens}")

    prefix_len = int(prompt_ids.shape[-1])
    additive_full = _decode_matrix(additive_mask, prefix_len, n_new_tokens)

    # `.clone()` on every slice is load-bearing — read the post-mortem in `ddi.inference.generate`
    # (inference.py:220-233) before touching it. A row of `additive_full` is contiguous by stride
    # but carries a storage offset, `.contiguous()` returns it unchanged, and an unaligned bf16
    # operand is what took down all 15 runs on 2026-08-06.
    #
    # Built out here rather than in `_prefill`/`_decode` on purpose: this is a timing harness, and
    # slicing the mask inside the timed region measures the harness instead of the model. The
    # per-step kwargs are assembled out here for the same reason.
    prefill_additive = additive_full[:prefix_len, :prefix_len].clone().unsqueeze(0).unsqueeze(0)
    prefill_kwargs: dict[str, torch.Tensor] = {"attention_mask": prefill_additive}
    # Step i processes the token at absolute position prefix_len + i, attending over the
    # prefix_len + i + 1 entries already in the cache — hence one mask row per step.
    step_kwargs: list[dict[str, torch.Tensor]] = [
        {
            "attention_mask": additive_full[prefix_len + i, : prefix_len + i + 1]
            .clone()
            .reshape(1, 1, 1, prefix_len + i + 1)
        }
        for i in range(n_new_tokens)
    ]

    # Only populated when the caller asked for explicit positions: the mocks and stand-ins that
    # cover this loop mirror HF's signature loosely, and passing `position_ids=None` to one that
    # does not declare the kwarg is a TypeError rather than a no-op.
    if position_ids is not None:
        positions_full = _decode_positions(position_ids, prefix_len, n_new_tokens)
        prefill_kwargs["position_ids"] = positions_full[:, :prefix_len].clone()
        for i, kwargs in enumerate(step_kwargs):
            kwargs["position_ids"] = positions_full[
                :, prefix_len + i : prefix_len + i + 1
            ].clone()

    past: object = None
    first_token: torch.Tensor | None = None

    def _prefill() -> None:
        """Untimed precondition: the decode loop cannot start without a warm KV cache."""
        nonlocal past, first_token
        with torch.no_grad():
            out = model(input_ids=prompt_ids, use_cache=True, **prefill_kwargs)
        past = out.past_key_values
        first_token = out.logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)

    def _decode() -> object:
        # Local copies so a repeat always resumes from the cache `_prefill` just built rather
        # than from wherever the previous repeat left off.
        step_past, next_token = past, first_token
        with torch.no_grad():
            for i in range(n_new_tokens):
                step = model(
                    input_ids=next_token,
                    past_key_values=step_past,
                    use_cache=True,
                    **step_kwargs[i],
                )
                step_past = step.past_key_values
                next_token = step.logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)
        return next_token

    decode_total = measure(
        _decode,
        setup=_prefill,
        n_repeats=n_repeats,
        warmup=warmup,
        name="decode_total",
        use_cuda_events=_cuda_events_for(prompt_ids, use_cuda_events),
    )
    per_token = [x / n_new_tokens for x in decode_total.per_repeat_ms]
    return _timing_from_samples(per_token, name="decode", n_repeats=n_repeats, warmup=warmup)


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
