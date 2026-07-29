"""Tests for the timing harness (issue #17, Phase 3.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ddi import timing
from ddi.provenance import Provenance
from ddi.timing import Timing, append_timing, measure


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
    """Mock that returns a `past_key_values` tuple and deterministic argmax logits."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):  # noqa: ARG002
        self.calls += 1
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


def test_measure_decode_zero_new_tokens_raises() -> None:
    from ddi.timing import measure_decode

    model = _MockDecodeModel()
    prompt = torch.zeros(1, 6, dtype=torch.long)
    mask = torch.zeros(1, 1, 6, 6, dtype=torch.float32)
    with pytest.raises(ValueError, match="n_new_tokens"):
        measure_decode(model, prompt, mask, n_new_tokens=0)
