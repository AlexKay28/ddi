"""Tests for the timing harness (issue #17, Phase 3.1)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch

from ddi import timing
from ddi.provenance import Provenance
from ddi.timing import Timing, append_timing, measure, measure_decode
from tests.fake_models import RecordingCausalLM, SlowCausalLM


@pytest.fixture
def prov() -> Provenance:
    return Provenance(
        git_sha="0" * 40,
        dirty=False,
        config_hash="cfg-hash",
        seed=7,
        model_revision="model@rev",
        dataset_revision="data@rev",
        gpu_model="cpu",
        driver_version="unknown",
        cuda_version="cpu",
        torch_version="2.4.0",
        timestamp_utc="2026-07-27T14:33:12+00:00",
    )


# ---- measure primitive ---------------------------------------------------------------------


def test_measure_basic() -> None:
    t = measure(lambda: sum(range(1000)), n_repeats=5, warmup=3, use_cuda_events=False)
    assert isinstance(t, Timing)
    assert t.n_repeats == 5
    assert t.warmup == 3
    assert len(t.per_repeat_ms) == 5
    assert t.median_ms > 0


def test_measure_median_iqr(monkeypatch: pytest.MonkeyPatch) -> None:
    samples = iter([1.0, 2.0, 3.0, 4.0, 5.0])
    monkeypatch.setattr(timing, "_time_once_cpu", lambda fn: next(samples))
    t = measure(lambda: None, n_repeats=5, warmup=0, use_cuda_events=False)
    assert t.median_ms == 3.0
    assert t.iqr_low_ms == 2.0
    assert t.iqr_high_ms == 4.0
    assert t.per_repeat_ms == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_measure_warmup_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    samples = iter([10.0, 10.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    monkeypatch.setattr(timing, "_time_once_cpu", lambda fn: next(samples))
    t = measure(lambda: None, n_repeats=5, warmup=3, use_cuda_events=False)
    assert t.per_repeat_ms == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert t.median_ms == 3.0


def test_measure_zero_repeats_raises() -> None:
    with pytest.raises(ValueError, match="n_repeats"):
        measure(lambda: None, n_repeats=0, warmup=0, use_cuda_events=False)


def test_measure_negative_warmup_raises() -> None:
    with pytest.raises(ValueError, match="warmup"):
        measure(lambda: None, n_repeats=1, warmup=-1, use_cuda_events=False)


def test_measure_cpu_fallback() -> None:
    """With `use_cuda_events=False` no CUDA is needed and a valid Timing is returned."""
    calls = 0

    def _fn() -> None:
        nonlocal calls
        calls += 1

    t = measure(_fn, n_repeats=4, warmup=2, use_cuda_events=False, name="cpu-only")
    assert t.name == "cpu-only"
    assert calls == 6  # warmup + n_repeats
    assert len(t.per_repeat_ms) == 4
    assert all(x >= 0 for x in t.per_repeat_ms)
    assert t.iqr_low_ms <= t.median_ms <= t.iqr_high_ms


def test_measure_setup_runs_every_iteration_but_is_not_timed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`setup` runs on warmup iterations too, and only `fn` sits inside the timer."""
    order: list[str] = []
    monkeypatch.setattr(
        timing, "_time_once_cpu", lambda fn: (order.append("time"), fn(), 1.0)[-1]
    )

    t = measure(
        lambda: order.append("fn"),
        setup=lambda: order.append("setup"),
        n_repeats=2,
        warmup=1,
        use_cuda_events=False,
    )
    assert order == ["setup", "time", "fn"] * 3
    assert t.per_repeat_ms == [1.0, 1.0]


def test_measure_setup_cost_stays_out_of_the_reading() -> None:
    """A setup an order of magnitude more expensive than `fn` must not move the number."""
    fast = measure(
        lambda: time.sleep(0.002), setup=None, n_repeats=3, warmup=1, use_cuda_events=False
    )
    with_setup = measure(
        lambda: time.sleep(0.002),
        setup=lambda: time.sleep(0.05),
        n_repeats=3,
        warmup=1,
        use_cuda_events=False,
    )
    # 50 ms of setup against a ~2 ms fn: if it leaked in, the medians would differ by ~25x.
    assert with_setup.median_ms < fast.median_ms * 2.0


def test_measure_cuda_events_missing_device_raises() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA present; this test only meaningful on CPU-only")
    with pytest.raises(RuntimeError, match="cuda"):
        measure(lambda: None, n_repeats=1, warmup=0, use_cuda_events=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_measure_cuda_events() -> None:
    a = torch.randn(256, 256, device="cuda")
    b = torch.randn(256, 256, device="cuda")

    def _fn() -> torch.Tensor:
        return a @ b

    t = measure(_fn, n_repeats=5, warmup=3, use_cuda_events=True, name="matmul")
    assert t.name == "matmul"
    assert len(t.per_repeat_ms) == 5
    assert t.median_ms > 0
    assert all(x > 0 for x in t.per_repeat_ms)


# ---- append_timing --------------------------------------------------------------------------


def test_append_timing_writes_median_and_repeats(tmp_path: Path, prov: Provenance) -> None:
    store = tmp_path / "runs.jsonl"
    t = Timing(
        name="prefill",
        n_repeats=5,
        warmup=3,
        median_ms=3.0,
        iqr_low_ms=2.0,
        iqr_high_ms=4.0,
        per_repeat_ms=[1.0, 2.0, 3.0, 4.0, 5.0],
    )
    append_timing(store, "run-1", "prefill", t, prov)

    rows = [json.loads(line) for line in store.read_text().splitlines() if line]
    assert len(rows) == 4

    by_metric = {r["metric"]: r for r in rows}
    assert by_metric["prefill_median_ms"]["value"] == 3.0
    assert by_metric["prefill_iqr_low_ms"]["value"] == 2.0
    assert by_metric["prefill_iqr_high_ms"]["value"] == 4.0
    assert json.loads(by_metric["prefill_repeats_ms"]["value"]) == [1.0, 2.0, 3.0, 4.0, 5.0]

    for r in rows:
        assert r["run_id"] == "run-1"
        assert r["provenance"]["git_sha"] == prov.git_sha
        assert r["provenance"]["seed"] == 7


# ---- measure_prefill / measure_decode integration on a mock -------------------------------


class _MockPrefillModel:
    """Just enough surface for `measure_prefill` — records the number of forward calls."""

    def __init__(self) -> None:
        self.calls = 0
        # Satisfy _find_patch_target through model wrapping: not needed here because we bypass
        # patch_model and call forward_with_mask directly on this stand-in.

    def __call__(self, input_ids, attention_mask, use_cache, position_ids=None):  # noqa: ARG002
        self.calls += 1
        from types import SimpleNamespace

        vocab = 10
        return SimpleNamespace(logits=torch.zeros(input_ids.shape + (vocab,)))


def test_measure_prefill_counts_calls() -> None:
    from ddi.timing import measure_prefill

    model = _MockPrefillModel()
    input_ids = torch.zeros(1, 8, dtype=torch.long)
    mask = torch.zeros(1, 1, 8, 8, dtype=torch.float32)

    t = measure_prefill(model, input_ids, mask, n_repeats=3, warmup=2)
    assert model.calls == 5
    assert len(t.per_repeat_ms) == 3
    assert t.name == "prefill"


class _MockDecodeModel:
    """Mock that returns a `past_key_values` tuple and deterministic argmax logits.

    Counts calls only. What each call was *given* — the mask per decode step — is pinned in
    `tests/test_decode_mask_per_step.py` against a recording model; an unmasked decode step is
    invisible to a call count.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.masks: list[object] = []

    def __call__(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):  # noqa: ARG002
        self.calls += 1
        self.masks.append(attention_mask)
        from types import SimpleNamespace

        vocab = 4
        logits = torch.zeros(input_ids.shape + (vocab,))
        # Argmax at position 0 for determinism (no ties matter here).
        logits[..., -1, 0] = 1.0
        return SimpleNamespace(logits=logits, past_key_values=("cached",))


def test_measure_decode_per_token_reporting() -> None:
    from ddi.timing import measure_decode

    model = _MockDecodeModel()
    prompt = torch.zeros(1, 6, dtype=torch.long)
    mask = torch.zeros(1, 1, 6, 6, dtype=torch.float32)
    n_new = 4

    t = measure_decode(model, prompt, mask, n_new_tokens=n_new, n_repeats=3, warmup=1)
    # Each iteration: 1 prefill + n_new decode steps. Iterations = warmup + n_repeats.
    assert model.calls == (1 + n_new) * (1 + 3)
    assert len(t.per_repeat_ms) == 3
    assert t.name == "decode"
    assert t.iqr_low_ms <= t.median_ms <= t.iqr_high_ms
    assert all(m is not None for m in model.masks), "a decode step ran unmasked"


def test_measure_decode_zero_new_tokens_raises() -> None:
    from ddi.timing import measure_decode

    model = _MockDecodeModel()
    prompt = torch.zeros(1, 6, dtype=torch.long)
    mask = torch.zeros(1, 1, 6, 6, dtype=torch.float32)
    with pytest.raises(ValueError, match="n_new_tokens"):
        measure_decode(model, prompt, mask, n_new_tokens=0)


# ---- tpot means time-per-output-token, not (FTT + n*TPOT) / n -------------------------------
#
# `measure_decode` used to run the prefill and the whole decode loop inside ONE timed region and
# divide the total by `n_new_tokens`, so the reported `tpot` carried an extra `FTT / n`. At the
# shipped default of 16 new tokens against an L=2048, N=25 prefill that term is the larger one.
#
# The reason it is a correctness bug rather than an imprecision is that the contamination is
# CORRELATED with the variable under test: prefill is exactly where mask sparsity pays off, so a
# sparse arm's `tpot` was flattered by its own cheaper prefill, and Phase 4 would have read part of
# a prefill speedup as a decode speedup. `realization` in experiments/performance.yaml is derived
# from `tpot` and inherited it.
#
# `SlowCausalLM` sleeps `ms_per_input_token` per input token, so a prompt of T tokens costs T ms to
# prefill while every decode step costs 1 ms regardless of T. Sweeping T therefore moves the
# prefill cost and nothing else.

_STEP_MS = 1.0
_DECODE_BUDGET = 4


def _tpot_ms(prefix_len: int) -> float:
    model = SlowCausalLM(ms_per_input_token=_STEP_MS)
    prompt = torch.zeros(1, prefix_len, dtype=torch.long)
    mask = torch.zeros(1, 1, prefix_len, prefix_len, dtype=torch.float32)
    t = measure_decode(
        model, prompt, mask, n_new_tokens=_DECODE_BUDGET, n_repeats=2, warmup=1
    )
    return t.median_ms


def test_tpot_does_not_scale_with_prefill_cost() -> None:
    cheap_prefill, dear_prefill = 16, 128
    tpot_cheap = _tpot_ms(cheap_prefill)
    tpot_dear = _tpot_ms(dear_prefill)

    # What the pre-fix code would have added to the gap: the whole difference in prefill cost,
    # amortised over the decode budget. 112 ms / 4 = 28 ms per token, against a true 1 ms.
    leak_if_broken = (dear_prefill - cheap_prefill) * _STEP_MS / _DECODE_BUDGET
    gap = abs(tpot_dear - tpot_cheap)
    assert gap < leak_if_broken / 4, (
        f"tpot tracks prefill cost: {tpot_cheap:.2f} ms at prefix_len={cheap_prefill} vs "
        f"{tpot_dear:.2f} ms at prefix_len={dear_prefill} (gap {gap:.2f} ms). The prefill is "
        f"back inside the timed region."
    )


def test_tpot_is_the_per_step_cost_not_the_amortised_total() -> None:
    """The absolute value, not just its flatness: 8x the prefill must not shift it off 1 ms."""
    prefix_len = 128
    tpot = _tpot_ms(prefix_len)
    amortised_total = (prefix_len * _STEP_MS + _DECODE_BUDGET * _STEP_MS) / _DECODE_BUDGET
    assert tpot < amortised_total / 4, (
        f"tpot={tpot:.2f} ms is close to (FTT + n*TPOT)/n = {amortised_total:.2f} ms rather than "
        f"to the {_STEP_MS:.2f} ms a decode step actually costs"
    )
    # Generous upper bound rather than an equality: sleep wakeup granularity and the recording
    # model's own bookkeeping ride along, and both are small next to the 33 ms leak above.
    assert tpot < 4 * _STEP_MS


def test_measure_decode_still_prefills_every_iteration() -> None:
    """The prefill is untimed, not skipped — the decode loop needs a warm cache to start from."""
    model = RecordingCausalLM()
    prompt = torch.zeros(1, 6, dtype=torch.long)
    mask = torch.zeros(1, 1, 6, 6, dtype=torch.float32)
    measure_decode(model, prompt, mask, n_new_tokens=3, n_repeats=2, warmup=2)
    assert len(model.prefill_calls) == 4  # warmup + n_repeats
    assert len(model.decode_calls) == 12
    # Each repeat resumes from the cache its own prefill built, not from the previous repeat's.
    assert [c.past_len for c in model.decode_calls] == [6, 7, 8] * 4


# ---- position_ids reach the timing path (Phase 4.1 layout axis) -----------------------------


def test_measure_prefill_forwards_position_ids() -> None:
    from ddi.timing import measure_prefill

    model = RecordingCausalLM()
    input_ids = torch.zeros(1, 8, dtype=torch.long)
    mask = torch.zeros(1, 1, 8, 8, dtype=torch.float32)
    positions = torch.arange(8, dtype=torch.long).unsqueeze(0)

    measure_prefill(model, input_ids, mask, position_ids=positions, n_repeats=1, warmup=0)
    call = model.calls[0]
    assert call.position_ids is not None
    assert torch.equal(call.position_ids.values, positions)


def test_measure_prefill_without_positions_forwards_none() -> None:
    from ddi.timing import measure_prefill

    model = RecordingCausalLM()
    measure_prefill(
        model,
        torch.zeros(1, 8, dtype=torch.long),
        torch.zeros(1, 1, 8, 8, dtype=torch.float32),
        n_repeats=1,
        warmup=0,
    )
    assert model.calls[0].position_ids is None


@pytest.mark.parametrize(
    ("positions", "match"),
    [
        (torch.zeros(2, 6, dtype=torch.long), "must be"),
        (torch.zeros(1, 5, dtype=torch.long), "matches neither"),
        (torch.zeros(1, 6, dtype=torch.float32), "torch.long"),
    ],
)
def test_measure_decode_rejects_bad_position_ids(
    positions: torch.Tensor, match: str
) -> None:
    model = RecordingCausalLM()
    with pytest.raises(ValueError, match=match):
        measure_decode(
            model,
            torch.zeros(1, 6, dtype=torch.long),
            torch.zeros(1, 1, 6, 6, dtype=torch.float32),
            n_new_tokens=3,
            position_ids=positions,
            n_repeats=1,
            warmup=0,
        )
