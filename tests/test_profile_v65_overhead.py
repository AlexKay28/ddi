"""CPU tests for the V65 profiling harness: synthetic profiler-event parsing and gap
decomposition only — no GPU, no model, no profiler."""

from __future__ import annotations

from types import SimpleNamespace

from scripts.profile_v65_overhead import (
    _ACTIVE,
    ATTENTION_KERNEL_PATTERNS,
    PhaseTrace,
    _phase,
    aggregate_op_counts,
    classify_attention_kernel,
    decompose_gap,
    summarize_kernel_events,
    top_kernels,
)


def _cuda_event(name: str, us: float) -> SimpleNamespace:
    return SimpleNamespace(
        name=name, device_type="DeviceType.CUDA", self_device_time_total=us, count=1
    )


def _cpu_event(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=name, device_type="DeviceType.CPU", self_device_time_total=0.0, count=1
    )


def test_classify_attention_kernel() -> None:
    assert classify_attention_kernel("void flash::compute_attn_kernel")
    assert classify_attention_kernel("at::native::(anonymous namespace)::sdpa_dense_backward")
    assert classify_attention_kernel("cutlass::Kernel2<fmha")
    assert not classify_attention_kernel("void at::native::vectorized_elementwise_kernel")
    assert not classify_attention_kernel("flash", patterns=())


def test_summarize_kernel_events_splits_memcpies_and_attention() -> None:
    events = [
        _cpu_event("aten::mm"),
        _cuda_event("Memcpy HtoD (Device -> Pageable)", 500.0),
        _cuda_event("void flash::attn_kernel<float>", 400.0),
        _cuda_event("void flash::attn_kernel<float>", 100.0),
        _cuda_event("ampere_bf16_s16816gemm", 250.0),
    ]
    summary = summarize_kernel_events(events, ATTENTION_KERNEL_PATTERNS)
    assert summary["kernel_ms"] == 0.75
    assert summary["launches"] == 3
    assert summary["attention_kernel_ms"] == 0.5
    assert summary["memcpy_ms"] == 0.5
    assert summary["memcpy_count"] == 1
    assert summary["by_name"]["void flash::attn_kernel<float>"] == {
        "count": 2,
        "total_us": 500.0,
    }


def test_top_kernels_orders_by_cuda_time_desc() -> None:
    by_name = {
        "b_small": {"count": 5, "total_us": 10.0},
        "a_big": {"count": 1, "total_us": 900.0},
        "c_mid": {"count": 3, "total_us": 100.0},
    }
    rows = top_kernels(by_name, k=2)
    assert [row["name"] for row in rows] == ["a_big", "c_mid"]
    assert rows[0]["total_us"] == 900.0
    assert rows[1]["count"] == 3


def test_aggregate_op_counts_only_counts_cpu_aten_ops() -> None:
    events = [
        _cpu_event("aten::to"),
        _cpu_event("aten::to"),
        _cpu_event("aten::copy_"),
        _cpu_event("ProfilerStep#1"),
        _cuda_event("aten::mm_kernel", 10.0),
    ]
    counts = aggregate_op_counts(events)
    assert counts == {"aten::to": 2, "aten::copy_": 1}


def test_decompose_gap_arithmetic_and_contributor_order() -> None:
    arm_a = {
        "wall_ms": 125.0,
        "cpu_wall_ms": 40.0,
        "cuda": {"kernel_ms": 100.0},
    }
    arm_c = {
        "wall_ms": 221.0,
        "cpu_wall_ms": 150.0,
        "cuda": {"kernel_ms": 20.0},
        "phase_ms": {
            "cache_assemble": 20.0,
            "disk_load": 30.0,
            "check_model": 5.0,
        },
    }
    gap = decompose_gap(arm_a, arm_c)
    assert gap["gap_wall_ms"] == 96.0
    assert gap["gap_cpu_wall_ms"] == 110.0
    assert gap["gap_gpu_tail_ms"] == (221.0 - 150.0) - (125.0 - 40.0)
    assert gap["gap_cuda_kernel_ms"] == -80.0
    names = [item["phase"] for item in gap["contributors_ms"]]
    assert names == ["disk_load", "cache_assemble", "check_model"]
    assert gap["contributors_ms"][0]["ms"] == 30.0


def test_phase_trace_accumulates_nested_phases() -> None:
    trace = PhaseTrace()
    _ACTIVE["trace"] = trace
    try:
        with _phase("outer"):
            with _phase("inner"):
                pass
    finally:
        _ACTIVE["trace"] = None
    assert trace.counts == {"outer": 1, "inner": 1}
    assert set(trace.totals) == {"outer", "inner"}
    assert trace.totals["outer"] >= trace.totals["inner"]
